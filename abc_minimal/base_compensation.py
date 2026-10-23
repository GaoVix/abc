from __future__ import annotations

import json
import time
from pathlib import Path

import mujoco
import numpy as np

from deploy.robot.control.mink_ik import MinkArmController


def _pose(entity):
    transform = np.eye(4)
    transform[:3, :3] = entity.xmat.reshape(3, 3)
    transform[:3, 3] = entity.xpos
    return transform


class BaseCompensation:
    """Retarget physical joints through TCP poses in the fixed top camera frame."""

    def __init__(self):
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
        self.tcp_from_link = [
            np.linalg.inv(_pose(self.data.site(f"{side}_tcp_site")))
            @ _pose(self.data.body(f"{side}_link_6"))
            for side in self.sides
        ]
        self.controllers = []
        for side in self.sides:
            spec = mujoco.MjSpec.from_file(str(path))
            for field in ("sensors", "equalities", "actuators", "tendons", "keys"):
                for element in list(getattr(spec, field)):
                    spec.delete(element)
            for joint in list(spec.joints):
                if joint.name not in {f"{side}_joint{i}" for i in range(1, 7)}:
                    spec.delete(joint)
            self.controllers.append(MinkArmController(
                np.zeros(6), model=spec.compile(), target_link=f"{side}_link_6",
                max_velocity=0, max_joint_delta=np.inf,
            ))
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
            poses = record.metadata.get("arm_base_poses", {}) if record is not None else {}
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
                arms.append(camera_from_actual @ np.linalg.inv(camera_from_nominal))
            transforms.append(arms)
        self.input_transform = np.asarray(transforms)
        self.output_transform = np.linalg.inv(self.input_transform)

    def _fk(self, joints):
        self.data.qpos[self.qpos] = joints.reshape(2, 7)[:, :6]
        mujoco.mj_kinematics(self.model, self.data)
        return [self.camera_from_world @ _pose(self.data.site(f"{side}_tcp_site"))
                for side in self.sides]

    def _solve(self, arm, target, seed):
        link_target = self.world_from_camera @ target @ self.tcp_from_link[arm]
        quaternion = np.empty(4)
        mujoco.mju_mat2Quat(quaternion, link_target[:3, :3].ravel())
        joints = seed.copy()
        for calls in range(11):
            self.data.qpos[self.qpos[arm]] = joints
            mujoco.mj_kinematics(self.model, self.data)
            current = self.camera_from_world @ _pose(self.data.site(f"{self.sides[arm]}_tcp_site"))
            position_error = np.linalg.norm(current[:3, 3] - target[:3, 3]) * 1000
            cosine = (np.trace(current[:3, :3].T @ target[:3, :3]) - 1) / 2
            rotation_error = np.rad2deg(np.arccos(np.clip(cosine, -1, 1)))
            if (position_error <= 0.1 and rotation_error <= 0.1) or calls == 10:
                break
            joints = self.controllers[arm].solve(link_target[:3, 3], quaternion, joints)
        return joints, position_error, rotation_error, calls

    def convert(self, joints, *, state=False):
        started = time.perf_counter()
        values = np.asarray(joints, dtype=np.float64)
        result = values.reshape(-1, 14).copy()
        transforms = self.input_transform if state else self.output_transform
        seeds = self.state_seed if state else self.action_seed
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
    def __init__(self, env, output_dir):
        self._env = env
        self._output_dir = Path(output_dir)
        self._conversion = BaseCompensation()
        print("base compensation: noisy joints -> camera TCP -> nominal joints; inverse for control", flush=True)

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

    def step(self, actions):
        return self._env.step(self._conversion.convert(actions))

    def step_one(self, action):
        return self._env.step_one(self._conversion.convert(action))

    step_one_vanilla = step_one

    def close(self):
        self._env.close()
        path = self._output_dir / "base_compensation.json"
        path.write_text(json.dumps(self._conversion.stats, indent=2) + "\n")
        print(f"base compensation IK: {self._conversion.stats}", flush=True)
