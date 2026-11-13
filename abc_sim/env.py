"""MuJoCo simulation environment for bimanual YAM robot."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import string
from typing import Any, Optional

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

from abc_sim.config import RobotSystemConfig
from abc_sim.rendering.live import LiveRenderBackend, create_live_camera_provider
from abc_sim.task_eval import TaskEvalResult, make_task_evaluator
from abc_sim.task_registry import DEFAULT_SCENE_XML, SCENE_XMLS, get_task_randomizer, resolve_task
from abc_sim.task_runtime import make_task_runtime
from abc_sim.task_specs import SimTaskSpec, maybe_get_task_spec

# Backwards-compatible aliases for older imports.
_SCENE_XMLS = dict(SCENE_XMLS)
_DEFAULT_SCENE_XML = DEFAULT_SCENE_XML

# Gripper actuator ctrl range max (from menagerie model)
_GRIPPER_CTRL_MAX = 0.0475
_OBS_PROMPT_CHARSET = string.ascii_letters + string.digits + string.punctuation + " "


def project_policy_state(
    qpos: np.ndarray,
    qpos_indices: list[int],
    gripper_indices: list[int],
    *,
    dtype=np.float32,
) -> np.ndarray:
    """Project MuJoCo qpos into the 14D policy-space state vector."""
    state = np.zeros(len(qpos_indices), dtype=dtype)
    gripper_set = set(gripper_indices)
    for i, qpos_idx in enumerate(qpos_indices):
        val = qpos[qpos_idx]
        if i in gripper_set:
            val = np.clip(val / _GRIPPER_CTRL_MAX, 0.0, 1.0)
        state[i] = val
    return state


def project_policy_state_batch(
    qpos_batch: np.ndarray,
    qpos_indices: list[int],
    gripper_indices: list[int],
    *,
    dtype=np.float32,
) -> np.ndarray:
    """Project a batch of MuJoCo qpos vectors into policy-space states."""
    qpos_batch = np.asarray(qpos_batch)
    if qpos_batch.ndim != 2:
        raise ValueError(f"Expected batched qpos with shape (B, nq), got {qpos_batch.shape}")

    state = np.asarray(qpos_batch[:, qpos_indices], dtype=dtype)
    if gripper_indices:
        state[:, gripper_indices] = np.clip(
            state[:, gripper_indices] / _GRIPPER_CTRL_MAX,
            0.0,
            1.0,
        )
    return state


class MuJoCoYAMEnv(gym.Env):
    """MuJoCo-based simulation of the bimanual YAM robot.

    Actions and observations use the same 14D format as the real robot:
    [left_j1..6, left_grip, right_j1..6, right_grip].
    """

    def __init__(
        self,
        config: RobotSystemConfig,
        chunk_dim: int = 30,
        prompt: str = "fold the towel",
        render_cameras: bool = True,
        camera_backend: LiveRenderBackend = "mujoco",
        camera_gpu_id: int | None = None,
        camera_height: int = 480,
        camera_width: int = 640,
        physics_dt: float = 0.002,
        control_decimation: int = 17,  # 0.002 * 17 ≈ 0.034s ≈ 30Hz
        scene_xml: str | Path | None = None,
        max_episode_steps: int | None = None,
        terminate_on_success: bool = False,
        disable_arm_control_limits: bool = False,
        disable_all_arm_limits: bool = False,
        disable_arm_joint_limits: bool = False,
    ):
        super().__init__()
        self.config = config
        self.chunk_dim = chunk_dim
        self.prompt = prompt
        self._render_cameras_flag = render_cameras
        self._camera_backend: LiveRenderBackend = camera_backend
        self._camera_gpu_id = camera_gpu_id
        self._camera_height = camera_height
        self._camera_width = camera_width
        self._physics_dt = physics_dt
        self._control_decimation = control_decimation
        self._disable_arm_control_limits = bool(disable_arm_control_limits)
        self._disable_all_arm_limits = bool(disable_all_arm_limits)
        self._disable_arm_joint_limits = bool(disable_arm_joint_limits)
        self._scene_xml = Path(scene_xml) if scene_xml else _DEFAULT_SCENE_XML
        if max_episode_steps is not None and max_episode_steps < 1:
            raise ValueError(
                f"max_episode_steps must be >= 1 or None, got {max_episode_steps}"
            )
        self.max_episode_steps = max_episode_steps
        self.terminate_on_success = bool(terminate_on_success)
        self._scene_xml_string: str | None = None  # set directly for in-memory XML
        self._task_request: str = ""
        self._task: str = ""  # set by make_env after construction
        self._task_spec: SimTaskSpec | None = None
        self._task_evaluator = None
        self._task_runtime = None
        self._arm_state_initialized = False

        # Populated by reset() when randomize=True; readable by callers.
        self._last_randomization: Any = None

        self.camera_names = list(config.cameras.keys())
        self.robot_names = list(config.robots.keys())
        self.single_timestep_action_dim = 7 * len(config.robots)  # 14
        self.action_dim = self.single_timestep_action_dim
        self.chunk_action_dim = self.single_timestep_action_dim * self.chunk_dim

        self.observation_space = spaces.Dict(
            {
                "images": spaces.Dict(
                    {
                        name: spaces.Box(
                            low=0,
                            high=255,
                            shape=(
                                3,
                                self._camera_height,
                                self._camera_width,
                            ),
                            dtype=np.uint8,
                        )
                        for name in self.camera_names
                    }
                ),
                "state": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.single_timestep_action_dim,),
                    dtype=np.float32,
                ),
                "masks": spaces.Dict(
                    {name: spaces.Discrete(2) for name in self.camera_names}
                ),
                "prompt": spaces.Text(
                    max_length=4096,
                    min_length=0,
                    charset=_OBS_PROMPT_CHARSET,
                ),
                "camera_timestamps": spaces.Dict(
                    {
                        name: spaces.Box(
                            low=-np.inf,
                            high=np.inf,
                            shape=(),
                            dtype=np.float64,
                        )
                        for name in self.camera_names
                    }
                ),
            }
        )

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.single_timestep_action_dim,),
            dtype=np.float32,
        )
        self.chunk_action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.chunk_dim, self.single_timestep_action_dim),
            dtype=np.float32,
        )

        self.setup_model()
        self.cur_step = 0

    def setup_model(self):
        if self._scene_xml_string is not None:
            # Composed scene XML references assets relative to the bundled
            # models directory, but MuJoCo resolves them against the process
            # CWD — anchor it during compilation.
            prev_cwd = os.getcwd()
            os.chdir(Path(__file__).parent / "models")
            try:
                model = mujoco.MjModel.from_xml_string(self._scene_xml_string)
            finally:
                os.chdir(prev_cwd)
        else:
            model = mujoco.MjModel.from_xml_path(str(self._scene_xml))
        self._bind_model(model)

    def _bind_model(self, model: mujoco.MjModel) -> None:
        self.model = model
        self.model.opt.timestep = self._physics_dt
        if self._disable_arm_control_limits or self._disable_all_arm_limits:
            # Apply to every newly compiled/cached model, before creating any
            # MJWarp runtime. Gripper settings are always kept intact.
            for side in self.robot_names:
                for joint in range(1, 7):
                    actuator = self.model.actuator(f"{side}_joint{joint}").id
                    self.model.actuator_ctrllimited[actuator] = False
                    if self._disable_all_arm_limits:
                        joint_id = self.model.joint(f"{side}_joint{joint}").id
                        self.model.jnt_limited[joint_id] = False
                        self.model.jnt_actfrclimited[joint_id] = False
                        self.model.actuator_forcelimited[actuator] = False
        if self._disable_arm_joint_limits:
            for side in self.robot_names:
                for joint in range(1, 7):
                    joint_id = self.model.joint(f"{side}_joint{joint}").id
                    self.model.jnt_limited[joint_id] = False
        if self._disable_all_arm_limits or self._disable_arm_joint_limits:
            names = [f"{side}_joint{joint}" for side in self.robot_names for joint in range(1, 7)]
            joints = [self.model.joint(name).id for name in names]
            actuators = [self.model.actuator(name).id for name in names]
            limits = {
                "joint_position_limited": self.model.jnt_limited[joints].astype(bool).tolist(),
                "pd_target_limited": self.model.actuator_ctrllimited[actuators].astype(bool).tolist(),
                "actuator_force_limited": self.model.actuator_forcelimited[actuators].astype(bool).tolist(),
                "joint_actuator_force_limited": self.model.jnt_actfrclimited[joints].astype(bool).tolist(),
            }
            print("[arm limits] " + json.dumps(limits), flush=True)
        self.data = mujoco.MjData(self.model)
        self._arm_state_initialized = False
        if self._task_runtime is not None:
            self._task_runtime.bind(self.model, self.data)
        self._camera_provider = None

        if self._render_cameras_flag:
            if self._camera_backend == "mujoco":
                self.renderer = mujoco.Renderer(
                    self.model,
                    height=self._camera_height,
                    width=self._camera_width,
                )
            else:
                self._camera_provider = create_live_camera_provider(
                    model=self.model,
                    data=self.data,
                    backend=self._camera_backend,
                    width=self._camera_width,
                    height=self._camera_height,
                    gpu_id=self._camera_gpu_id,
                    camera_names=tuple(self.camera_names),
                )

        self._build_index_maps()

    def reload_from_xml(self, xml_string: str) -> None:
        """Swap out the MuJoCo model in-place from an XML string.

        Closes the existing renderer, loads a new model/data/renderer from
        the given XML string, and rebuilds index maps.  Does NOT call
        mj_resetData — the caller is responsible for resetting state afterward.
        """
        self._scene_xml_string = xml_string
        self._close_camera_backend()
        self.setup_model()
        self._task_evaluator = make_task_evaluator(self.model, self._task_spec)

    def reload_from_model(self, model: mujoco.MjModel) -> None:
        """Swap in a copy of an already-compiled MuJoCo model."""
        self._scene_xml_string = None
        self._close_camera_backend()
        self._bind_model(copy.deepcopy(model))
        self._task_evaluator = make_task_evaluator(self.model, self._task_spec)

    def _reset_camera_backend(self) -> None:
        if self._camera_provider is not None:
            self._camera_provider.reset()

    def _close_camera_backend(self) -> None:
        if self._camera_provider is not None:
            self._camera_provider.close()
            self._camera_provider = None
        if hasattr(self, "renderer"):
            self.renderer.close()
            del self.renderer

    def _build_index_maps(self):
        """Build mappings from 14D state/action to MuJoCo qpos/ctrl indices."""
        self._qpos_indices: list[int] = []
        self._ctrl_indices: list[int] = []
        self._gripper_indices: list[int] = []  # which of the 14 are grippers

        idx = 0
        for robot_name in self.robot_names:
            for j in range(1, 7):
                joint_name = f"{robot_name}_joint{j}"
                jnt_id = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
                )
                self._qpos_indices.append(self.model.jnt_qposadr[jnt_id])

                act_name = f"{robot_name}_joint{j}"
                act_id = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name
                )
                self._ctrl_indices.append(act_id)
                idx += 1

            finger_name = f"{robot_name}_left_finger"
            finger_jnt_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, finger_name
            )
            self._qpos_indices.append(self.model.jnt_qposadr[finger_jnt_id])

            grip_act_name = f"{robot_name}_gripper"
            grip_act_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, grip_act_name
            )
            self._ctrl_indices.append(grip_act_id)
            self._gripper_indices.append(idx)
            idx += 1

        self._gripper_set = set(self._gripper_indices)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
        randomize: bool = True,
    ):
        super().reset(seed=seed)
        reset_arm_state = self._get_reset_arm_state()
        mujoco.mj_resetData(self.model, self.data)
        self._set_qpos_from_state(reset_arm_state)
        mujoco.mj_forward(self.model, self.data)

        # _task_randomizer (set by make_env) takes priority over TASK_RANDOMIZERS
        # so that model-swapping tasks (e.g. inhand_transfer) work correctly.
        self._last_randomization = None
        if randomize:
            randomization_request = None
            if options is not None:
                randomization_request = options.get("randomization")
            randomizer = getattr(self, "_task_randomizer", None)
            if randomizer is None and self._task:
                randomizer = get_task_randomizer(self._task)
            if randomizer is not None:
                self._last_randomization = randomizer.randomize(
                    self.model,
                    self.data,
                    seed=seed,
                    request=randomization_request,
                )
                prompt = getattr(self._last_randomization, "metadata", {}).get("prompt")
                if prompt is not None:
                    self.prompt = str(prompt)

        if self._task_runtime is not None:
            self._task_runtime.after_reset(
                self.model,
                self.data,
                randomization=self._last_randomization,
            )
            self._sync_prompt_from_task_runtime()

        self._reset_camera_backend()
        if self._task_evaluator is not None:
            self._task_evaluator.reset(nworld=1)
            if self._last_randomization is not None and hasattr(
                self._task_evaluator,
                "configure_from_randomization",
            ):
                self._task_evaluator.configure_from_randomization(
                    self.model,
                    self._last_randomization,
                )
            if self._last_randomization is not None and hasattr(
                self._task_evaluator, "set_active_trash_joints"
            ):
                trash_joints = getattr(self._last_randomization, "metadata", {}).get(
                    "trash_joints"
                )
                if trash_joints is not None:
                    self._task_evaluator.set_active_trash_joints(trash_joints)
            if self._last_randomization is not None and hasattr(
                self._task_evaluator, "set_active_object_joints"
            ):
                active_joints = getattr(self._last_randomization, "metadata", {}).get(
                    "active_object_joints"
                )
                if active_joints is not None:
                    self._task_evaluator.set_active_object_joints(active_joints)

        self.cur_step = 0
        info: dict[str, Any] = {}
        if self._last_randomization is not None:
            info["randomization"] = self._last_randomization
        return self.get_obs(), info

    def step(self, action: np.ndarray, *, render_obs: bool = True):
        """Apply one 14D policy-space action.

        Returns the Gymnasium 5-tuple ``(obs, reward, terminated, truncated, info)``.
        If ``render_obs`` is false, ``obs`` is ``None`` and only physics/reward
        state is advanced.
        """

        obs = self._apply_single_action(action, render_obs=render_obs)
        reward, info = self._task_step_info()
        terminated, truncated = self._episode_status(info)
        return obs, reward, terminated, truncated, info

    def step_chunk(self, action_chunk: np.ndarray):
        """Apply a ``(chunk_dim, 14)`` action chunk and return chunk history."""

        action_chunk = self._as_action_chunk(action_chunk)
        all_obs = []
        reward = 0.0
        terminated = False
        truncated = False
        info: dict[str, Any] = {}
        for i in range(self.chunk_dim):
            obs, reward, terminated, truncated, info = self.step(
                action_chunk[i],
                render_obs=True,
            )
            if obs is None:
                raise RuntimeError(
                    "step_chunk(render_obs=True) returned no observation"
                )
            all_obs.append(obs)
            if terminated or truncated:
                break

        final_obs = all_obs[-1]
        chunk_history = self._stack_obs(all_obs)
        return final_obs, chunk_history, reward, terminated, truncated, info

    def _apply_single_action(
        self,
        action_14d: np.ndarray,
        *,
        render_obs: bool,
    ) -> dict[str, Any] | None:
        action = self._as_single_action(action_14d)
        self._step_single(action)
        self.cur_step += 1
        if not render_obs:
            return None
        return self.get_obs()

    def _as_single_action(self, action_14d: np.ndarray) -> np.ndarray:
        action = np.asarray(action_14d, dtype=np.float32)
        if action.size != self.single_timestep_action_dim:
            raise ValueError(
                "step() expects one 14D policy action; "
                "use step_chunk() for a (chunk_dim, 14) action chunk"
            )
        return action.reshape(self.single_timestep_action_dim)

    def _as_action_chunk(self, action_chunk: np.ndarray) -> np.ndarray:
        action = np.asarray(action_chunk, dtype=np.float32)
        expected_size = self.chunk_dim * self.single_timestep_action_dim
        if action.size != expected_size:
            raise ValueError(
                f"step_chunk() expects {self.chunk_dim} actions of size "
                f"{self.single_timestep_action_dim}; got shape {action.shape}"
            )
        return action.reshape(self.chunk_dim, self.single_timestep_action_dim)

    def _task_step_info(self) -> tuple[float, dict[str, Any]]:
        task_eval = self.evaluate_task()
        reward = task_eval.scalar_reward() if task_eval is not None else 0.0
        info: dict[str, Any] = {}
        if task_eval is not None:
            info["task_reward"] = reward
            info["task_success"] = task_eval.scalar_success()
            info["task_eval"] = task_eval.to_info(squeeze=True)
        return float(reward), info

    def _episode_status(self, info: dict[str, Any]) -> tuple[bool, bool]:
        task_success = bool(info.get("task_success", False))
        terminated = bool(self.terminate_on_success and task_success)
        time_limit_reached = (
            self.max_episode_steps is not None
            and self.cur_step >= self.max_episode_steps
        )
        truncated = bool(time_limit_reached and not terminated)
        info["episode_step"] = self.cur_step
        info["terminate_on_success"] = self.terminate_on_success
        info["time_limit_reached"] = bool(time_limit_reached)
        if self.max_episode_steps is not None:
            info["max_episode_steps"] = self.max_episode_steps
        return terminated, truncated

    def capture_state(self) -> dict[str, Any]:
        """Return a copy of the current MuJoCo state for tracing/replay."""

        return {
            "qpos": np.asarray(self.data.qpos, dtype=np.float32).copy(),
            "qvel": np.asarray(self.data.qvel, dtype=np.float32).copy(),
            "time": np.float32(self.data.time),
        }

    def close(self):
        self._close_camera_backend()

    def _set_qpos_from_state(self, state: np.ndarray):
        """Write a 14D state vector into MuJoCo qpos."""
        for i, qpos_idx in enumerate(self._qpos_indices):
            val = state[i]
            if i in self._gripper_set:
                # Scale [0, 1] policy space → MuJoCo joint space
                val = val * _GRIPPER_CTRL_MAX
            self.data.qpos[qpos_idx] = val
        self._arm_state_initialized = True

    def forget_arm_state(self) -> None:
        """Drop the carried arm pose so the next reset starts from init_q."""
        self._arm_state_initialized = False

    def _get_reset_arm_state(self) -> np.ndarray:
        """Return the arm state to preserve across scene resets."""
        if self._arm_state_initialized:
            return project_policy_state(
                np.asarray(self.data.qpos, dtype=np.float64),
                self._qpos_indices,
                self._gripper_indices,
            )
        return self.get_init_q().astype(np.float32, copy=True)

    def _step_single(self, action_14d: np.ndarray):
        """Apply a single 14D action and step physics."""
        ctrl = np.zeros(self.model.nu)
        for i in range(self.single_timestep_action_dim):
            val = action_14d[i]
            if i in self._gripper_set:
                # Scale [0, 1] policy space → actuator ctrl range
                val = val * _GRIPPER_CTRL_MAX
            ctrl[self._ctrl_indices[i]] = val
        self.data.ctrl[:] = ctrl
        self.advance_physics()

    def advance_physics(self, steps: int | None = None) -> None:
        """Advance MuJoCo physics while applying any task runtime controls."""

        num_steps = self._control_decimation if steps is None else steps
        if num_steps < 0:
            raise ValueError(f"steps must be non-negative, got {num_steps}")
        for _ in range(num_steps):
            if self._task_runtime is not None:
                self._task_runtime.before_step(self.model, self.data)
            mujoco.mj_step(self.model, self.data)
            if self._task_runtime is not None:
                after_step = getattr(self._task_runtime, "after_step", None)
                if after_step is not None:
                    after_step(self.model, self.data)
                self._sync_prompt_from_task_runtime()

    def _sync_prompt_from_task_runtime(self) -> None:
        """Adopt a runtime-owned prompt when the task exposes one."""

        if self._task_runtime is None:
            return
        observation_prompt = getattr(self._task_runtime, "observation_prompt", None)
        if observation_prompt is not None:
            self.prompt = str(observation_prompt())

    def project_state_from_qpos(self, qpos: np.ndarray) -> np.ndarray:
        """Project an arbitrary qpos vector into the 14D policy-space state."""
        return project_policy_state(
            qpos,
            self._qpos_indices,
            self._gripper_indices,
            dtype=np.float32,
        )

    def get_state(self) -> np.ndarray:
        """Return the current 14D policy-space state from MuJoCo qpos."""
        return self.project_state_from_qpos(self.data.qpos)

    def set_task(self, task: str) -> None:
        """Configure the task name and attach any registered evaluator."""

        resolved = resolve_task(task)
        self._task_request = task
        self._task = resolved.env_task or task
        self._task_spec = resolved.task_spec or maybe_get_task_spec(task)
        self._task_evaluator = make_task_evaluator(self.model, self._task_spec)
        self._task_runtime = make_task_runtime(
            resolved.task_spec or self._task,
            self.model,
            self.data,
        )

    def task_runtime_debug_state(self) -> dict[str, Any] | None:
        """Return task runtime diagnostics, when a runtime is attached."""

        if self._task_runtime is None:
            return None
        return self._task_runtime.debug_state()

    def evaluate_task(self) -> TaskEvalResult | None:
        """Compute the current task reward/success from the current simulation state."""

        if self._task_evaluator is None:
            return None
        evaluate_model_data = getattr(self._task_evaluator, "evaluate_model_data", None)
        if evaluate_model_data is not None:
            return evaluate_model_data(self.model, self.data)
        qpos_batch = np.asarray(self.data.qpos, dtype=np.float32)[None, :]
        return self._task_evaluator.evaluate_qpos_batch(qpos_batch)

    def get_obs(self) -> dict[str, Any]:
        """Return observation dict with state, images, and metadata."""
        # mj_step leaves derived poses before integration; refresh before RGB.
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_camlight(self.model, self.data)
        state = self.get_state()

        if self._render_cameras_flag:
            images = self._render_cameras()
        else:
            images = {
                name: np.zeros(
                    (3, self._camera_height, self._camera_width), dtype=np.uint8
                )
                for name in self.camera_names
            }

        masks = {name: True for name in self.camera_names}
        sim_time = np.asarray(self.data.time, dtype=np.float64)
        timestamps = {name: sim_time.copy() for name in self.camera_names}

        return {
            "images": images,
            "masks": masks,
            "state": state,
            "prompt": self.prompt,
            "camera_timestamps": timestamps,
        }

    def _render_cameras(self) -> dict[str, np.ndarray]:
        """Render all cameras and return (3, H, W) uint8 images."""
        if self._camera_provider is not None:
            frames = self._camera_provider.frames_for_step(self.cur_step, self.data.time)
            return {
                name: frame.transpose(2, 0, 1).copy()
                for name, frame in frames.items()
            }

        images = {}
        for name in self.camera_names:
            self.renderer.update_scene(self.data, camera=name)
            img = self.renderer.render()  # (H, W, 3) uint8
            images[name] = img.transpose(2, 0, 1).copy()  # → (3, H, W)
        return images

    def get_init_q(self) -> np.ndarray:
        """Return concatenated init_q across all robots as a flat 14D array."""
        return np.concatenate(
            [np.array(self.config.robots[name].init_q) for name in self.robot_names]
        )

    def _stack_obs(self, obses: list[dict]) -> dict:
        """Stack a list of observations into chunked arrays."""
        stacked: dict[str, Any] = {}
        for key in obses[0]:
            if isinstance(obses[0][key], str):
                stacked[key] = obses[0][key]
            elif isinstance(obses[0][key], dict):
                stacked[key] = {
                    k: np.stack([obs[key][k] for obs in obses])
                    for k in obses[0][key]
                }
            else:
                stacked[key] = np.stack([obs[key] for obs in obses])
        return stacked
