"""Small YAM-specific FK/IK wrapper using Mink and MuJoCo."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def find_yam_model() -> Path:
    import i2rt

    path = Path(i2rt.__file__).resolve().parent / "robot_models" / "yam" / "yam.xml"
    if not path.exists():
        raise FileNotFoundError(f"Could not find the i2rt YAM model at {path}")
    return path


class MinkArmController:
    def __init__(
        self,
        home: np.ndarray,
        *,
        target_link: str = "link_6",
        dt: float = 1.0 / 60.0,
        iterations: int = 4,
        orientation_cost: float = 5.0,
        gain: float = 0.6,
        max_velocity: float = 3.0,
        limit_margin: float = 0.1,
        limit_gain: float = 0.5,
        posture_cost: float = 5e-3,
        max_joint_delta: float = 0.12,
        solver: str = "daqp",
    ) -> None:
        try:
            import mink
            import mujoco
        except ImportError as exc:
            raise RuntimeError(
                "Mink IK is required for GELLO DAgger; run `uv sync --extra deploy`"
            ) from exc

        self.mink = mink
        self.mujoco = mujoco
        self.dt = float(dt)
        self.iterations = max(1, int(iterations))
        self.max_joint_delta = float(max_joint_delta)
        self.solver = solver
        self.model = mujoco.MjModel.from_xml_path(str(find_yam_model()))
        if self.model.nq != 6 or self.model.nv != 6:
            raise RuntimeError(
                f"Expected a six-DoF YAM model, got nq={self.model.nq}, nv={self.model.nv}"
            )
        body_names = {
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, index)
            for index in range(self.model.nbody)
        }
        if target_link not in body_names:
            raise RuntimeError(f"YAM target link {target_link!r} was not found")
        self.target_link = target_link
        self.configuration = mink.Configuration(self.model)
        self.lower, self.upper = self._joint_limits(limit_margin)
        self.home = np.clip(
            np.asarray(home, dtype=float).reshape(6), self.lower, self.upper
        )
        self.configuration.update(q=self.home)
        self.frame_task = mink.FrameTask(
            frame_name=target_link,
            frame_type="body",
            position_cost=50.0,
            orientation_cost=float(orientation_cost),
            gain=float(gain),
            lm_damping=1e-6,
        )
        posture_target = np.zeros(6)
        finite = np.isfinite(self.lower) & np.isfinite(self.upper)
        posture_target[finite] = 0.5 * (self.lower[finite] + self.upper[finite])
        posture = mink.PostureTask(self.model, cost=float(posture_cost))
        posture.set_target(posture_target)
        self.tasks = [self.frame_task, posture]
        self.limits = [
            mink.ConfigurationLimit(
                self.model,
                gain=float(limit_gain),
                min_distance_from_limits=self._effective_margin(limit_margin),
            )
        ]
        if max_velocity > 0:
            velocities = {
                mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, index): float(
                    max_velocity
                )
                for index in range(self.model.njnt)
            }
            self.limits.append(mink.VelocityLimit(self.model, velocities))

    def _effective_margin(self, requested: float) -> float:
        margin = max(0.0, float(requested))
        for index in range(self.model.njnt):
            if self.model.jnt_limited[index]:
                low, high = self.model.jnt_range[index]
                margin = min(margin, 0.49 * float(high - low))
        return margin

    def _joint_limits(self, requested_margin: float) -> tuple[np.ndarray, np.ndarray]:
        margin = self._effective_margin(requested_margin)
        lower = np.full(6, -np.inf)
        upper = np.full(6, np.inf)
        for index in range(self.model.njnt):
            if not self.model.jnt_limited[index]:
                continue
            address = self.model.jnt_qposadr[index]
            low, high = self.model.jnt_range[index]
            if high - low > 2 * margin:
                low += margin
                high -= margin
            lower[address], upper[address] = low, high
        return lower, upper

    def fk(self, joints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        joints = np.clip(
            np.asarray(joints, dtype=float).reshape(6), self.lower, self.upper
        )
        self.configuration.update(q=joints)
        pose = self.configuration.get_transform_frame_to_world(
            self.target_link, "body"
        ).parameters()
        return np.asarray(pose[4:], dtype=float), np.asarray(pose[:4], dtype=float)

    def solve(
        self, target_pos: np.ndarray, target_wxyz: np.ndarray, seed: np.ndarray
    ) -> np.ndarray:
        seed = np.clip(np.asarray(seed, dtype=float).reshape(6), self.lower, self.upper)
        self.configuration.update(q=seed)
        target = self.mink.SE3.from_rotation_and_translation(
            self.mink.SO3(np.asarray(target_wxyz, dtype=float).reshape(4)),
            np.asarray(target_pos, dtype=float).reshape(3),
        )
        self.frame_task.set_target(target)
        iteration_dt = self.dt / self.iterations
        try:
            for _ in range(self.iterations):
                velocity = self.mink.solve_ik(
                    self.configuration,
                    self.tasks,
                    iteration_dt,
                    self.solver,
                    damping=1e-6,
                    limits=self.limits,
                )
                self.configuration.integrate_inplace(velocity, iteration_dt)
        except self.mink.NoSolutionFound:
            return seed.copy()
        result = np.clip(np.asarray(self.configuration.q), self.lower, self.upper)
        result = seed + np.clip(
            result - seed, -self.max_joint_delta, self.max_joint_delta
        )
        return result if np.isfinite(result).all() else seed.copy()
