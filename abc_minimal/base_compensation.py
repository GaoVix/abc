from __future__ import annotations

import json
import time
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from deploy.robot.control.mink_ik import MinkArmController


def _pose(entity):
    transform = np.eye(4)
    transform[:3, :3] = entity.xmat.reshape(3, 3)
    transform[:3, 3] = entity.xpos
    return transform


class BaseCompensation:
    """Retarget physical joints through TCP poses in each world's top camera frame."""

    def __init__(self, ik_backend="pink", reference="xml"):
        if ik_backend not in ("pink", "mink"):
            raise ValueError(f"Unknown base compensation IK backend: {ik_backend}")
        if reference not in ("xml", "median"):
            raise ValueError(f"Unknown base compensation reference: {reference}")
        self.ik_backend = ik_backend
        self.reference = reference
        if ik_backend == "pink":
            try:
                from deploy.robot.control.pinocchio_ik import PinocchioArmController
            except ImportError as exc:
                raise RuntimeError(
                    "Pinocchio/Pink testing IK requires `uv sync --extra inference-ik`"
                ) from exc
        path = Path(__file__).resolve().parents[1] / "abc_sim/models/yam_dishrack_base.xml"
        self.model = mujoco.MjModel.from_xml_path(str(path))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.sides = ("left", "right")
        self.qpos = np.array([
            [self.model.joint(f"{side}_joint{i}").qposadr[0] for i in range(1, 7)]
            for side in self.sides
        ])
        self.world_from_camera = _pose(self.data.camera("top"))
        self.world_from_camera[:3, :3] @= np.diag([1., -1., -1.])
        self.camera_from_world = np.linalg.inv(self.world_from_camera)
        self.nominal_bases = [_pose(self.data.body(f"{side}_arm")) for side in self.sides]
        if reference == "median":
            poses = json.loads(Path(__file__).with_name(
                "dishrack_median_base_poses.json"
            ).read_text())["poses"]
            self.camera_from_reference = [
                np.asarray(poses[side]["T_camera_from_base"], dtype=np.float64)
                for side in self.sides
            ]
        else:
            self.camera_from_reference = [
                self.camera_from_world @ base for base in self.nominal_bases
            ]
        self.tcp_from_link = [
            np.linalg.inv(_pose(self.data.site(f"{side}_tcp_site")))
            @ _pose(self.data.body(f"{side}_link_6"))
            for side in self.sides
        ]
        self.controllers = []
        for arm, side in enumerate(self.sides):
            spec = mujoco.MjSpec.from_file(str(path))
            for field in ("sensors", "equalities", "actuators", "tendons", "keys"):
                for element in list(getattr(spec, field)):
                    spec.delete(element)
            for joint in list(spec.joints):
                if joint.name not in {f"{side}_joint{i}" for i in range(1, 7)}:
                    spec.delete(joint)
            if ik_backend == "pink":
                home = self.model.key("home").qpos[self.qpos[arm]]
                controller = PinocchioArmController(
                    home, model=spec.compile(), target_link=f"{side}_link_6",
                )
            else:
                # Keep the original inference position, velocity and step limits.
                controller = MinkArmController(
                    np.zeros(6), model=spec.compile(), target_link=f"{side}_link_6",
                )
            self.controllers.append(controller)
        self.stats = {
            name: {"arm_solves": 0, "solve_calls": 0, "not_converged": 0,
                   "max_position_error_mm": 0., "max_rotation_error_deg": 0., "seconds": 0.}
            for name in ("state", "action")
        }

    def reset(self, randomizations, state):
        self.state_seed = np.asarray(state, dtype=np.float64).reshape(-1, 14).copy()
        self.action_seed = self.state_seed.copy()
        transforms = []
        for record in randomizations:
            metadata = record.metadata if record is not None else {}
            poses = metadata.get("arm_base_poses", {})
            arms = []
            for arm, side in enumerate(self.sides):
                actual = self.nominal_bases[arm].copy()
                if f"{side}_arm" in poses:
                    pose = poses[f"{side}_arm"]
                    actual[:3, 3] = pose["pos"]
                    rotation = np.empty(9)
                    mujoco.mju_quat2Mat(rotation, np.asarray(pose["quat"]))
                    actual[:3, :3] = rotation.reshape(3, 3)
                camera_from_actual = self.camera_from_world @ actual
                camera_from_nominal = self.camera_from_world @ self.nominal_bases[arm]
                # The FK/IK model stays at XML base N. Represent the reference
                # base M in its coordinates: N M^-1 A FK_base(q_physical).
                # Inverting this transform gives N A^-1 M FK_base(q_policy).
                arms.append(
                    camera_from_nominal @ np.linalg.inv(self.camera_from_reference[arm])
                    @ camera_from_actual @ np.linalg.inv(camera_from_nominal)
                )
            transforms.append(arms)
        self.input_transform = np.asarray(transforms)
        self.output_transform = np.linalg.inv(self.input_transform)

    def _fk(self, joints):
        self.data.qpos[self.qpos] = joints.reshape(2, 7)[:, :6]
        mujoco.mj_kinematics(self.model, self.data)
        return [self.camera_from_world @ _pose(self.data.site(f"{side}_tcp_site"))
                for side in self.sides]

    def _solve(self, arm, target, seed):
        target = self.world_from_camera @ target
        link_target = target @ self.tcp_from_link[arm]
        quaternion = Rotation.from_matrix(link_target[:3, :3]).as_quat()[[3, 0, 1, 2]]
        joints = seed.copy()
        # Pink owns the five fixed iterations from robot(1).py. Do not wrap
        # those in the legacy Mink convergence loop (which would run 50 steps).
        max_calls = 1 if self.ik_backend == "pink" else 10
        for calls in range(1, max_calls + 1):
            joints = self.controllers[arm].solve(link_target[:3, 3], quaternion, joints)
            self.data.qpos[self.qpos[arm]] = joints
            mujoco.mj_kinematics(self.model, self.data)
            current = _pose(self.data.site(f"{self.sides[arm]}_tcp_site"))
            position_error = np.linalg.norm(current[:3, 3] - target[:3, 3]) * 1000
            rotation_error = np.rad2deg(Rotation.from_matrix(
                current[:3, :3] @ target[:3, :3].T).magnitude())
            if position_error <= 0.1 and rotation_error <= 0.1:
                break
        return joints, position_error, rotation_error, calls

    def convert(self, joints, *, state=False, physical_state=None):
        """Convert joints; Pink init/smoothness use the supplied physical state.

        Observations already contain that state. Action conversion requires the
        current measured state separately from the policy's target joints.
        """
        started = time.perf_counter()
        values = np.asarray(joints, dtype=np.float64)
        result = values.reshape(-1, 14).copy()
        transforms = self.input_transform if state else self.output_transform
        seeds = self.state_seed if state else self.action_seed
        if self.ik_backend == "pink":
            if physical_state is None:
                if not state:
                    raise ValueError("Pink action IK requires the current physical_state")
                physical_state = values
            seeds = np.asarray(physical_state, dtype=np.float64).reshape(-1, 14).copy()
            if seeds.shape != result.shape:
                raise ValueError("physical_state must match the action/state batch shape")
        stats = self.stats["state" if state else "action"]
        if np.allclose(transforms, np.eye(4), atol=1e-12):
            seeds[:] = result
            return result.reshape(values.shape).astype(np.float32)
        for world, row in enumerate(result):
            targets = self._fk(row)
            for arm in range(2):
                segment = slice(7 * arm, 7 * arm + 6)
                target = transforms[world, arm] @ targets[arm]
                solution, position, rotation, calls = self._solve(arm, target, seeds[world, segment])
                row[segment] = solution
                seeds[world, segment] = solution
                stats["arm_solves"] += 1
                stats["solve_calls"] += calls
                stats["not_converged"] += int(position > 0.1 or rotation > 0.1)
                stats["max_position_error_mm"] = max(stats["max_position_error_mm"], position)
                stats["max_rotation_error_deg"] = max(stats["max_rotation_error_deg"], rotation)
        stats["seconds"] += time.perf_counter() - started
        return result.reshape(values.shape).astype(np.float32)


class BaseCompensatedEnv:
    def __init__(self, env, output_dir, *, ik_backend="pink", reference="xml"):
        self._env = env
        self._output_dir = Path(output_dir)
        self._conversion = BaseCompensation(ik_backend=ik_backend, reference=reference)
        print(f"base compensation ({ik_backend} IK, {reference} reference): physical joints -> camera TCP -> reference joints; inverse for control", flush=True)

    def __getattr__(self, name):
        return getattr(self._env, name)

    def _obs(self, obs):
        return {**obs, "state": self._conversion.convert(obs["state"], state=True)}

    def reset(self, *args, **kwargs):
        obs = self._env.reset(*args, **kwargs)
        records = self._env.randomization
        self._conversion.reset(records if isinstance(records, list) else [records], obs["state"])
        return self._obs(obs)

    def get_obs(self):
        obs = self._env.get_obs() if hasattr(self._env, "base_env") else self._env.obs()
        return self._obs(obs)

    obs = get_obs
    obs_vanilla_state = get_obs

    def _physical_state(self):
        # Read at every control step, including inside a chunk. These accessors
        # read joints only and do not render cameras or convert observations.
        if hasattr(self._env, "base_env"):
            return self._env.state()
        return self._env.env.get_state()

    def step(self, actions):
        return self._env.step(self._conversion.convert(
            actions, physical_state=self._physical_state(),
        ))

    def step_one(self, action):
        return self._env.step_one(self._conversion.convert(
            action, physical_state=self._physical_state(),
        ))

    step_one_vanilla = step_one

    def close(self):
        self._env.close()
        path = self._output_dir / "base_compensation.json"
        path.write_text(json.dumps(self._conversion.stats, indent=2) + "\n")
        print(f"base compensation IK: {self._conversion.stats}", flush=True)
