"""Single-frame equality-constrained IK for testing-time base compensation."""

import mujoco
import numpy as np
from scipy.optimize import Bounds, minimize
from scipy.spatial.transform import Rotation

from .pinocchio_ik import PinocchioArmController


class SLSQPArmController:
    """Refine a Pink debug seed; keep measured joints as the posture reference."""

    def __init__(self, home, *, model, target_link):
        if model.nq != 6 or model.nv != 6:
            raise ValueError("Expected a six-DoF arm model")
        self.model = model
        self.data = mujoco.MjData(model)
        self.body_id = model.body(target_link).id
        self.lower = np.where(model.jnt_limited, model.jnt_range[:, 0], -np.inf)
        self.upper = np.where(model.jnt_limited, model.jnt_range[:, 1], np.inf)
        self.initializer = PinocchioArmController(
            home, model=model, target_link=target_link, debug=True,
        )
        self.last_result = None

    def pose_error(self, q, target):
        self.data.qpos[:] = q
        mujoco.mj_kinematics(self.model, self.data)
        rotation = self.data.xmat[self.body_id].reshape(3, 3)
        return np.r_[
            self.data.xpos[self.body_id] - target[:3, 3],
            Rotation.from_matrix(rotation @ target[:3, :3].T).as_rotvec(),
        ]

    def pose_jacobian(self, q, target):
        phi = self.pose_error(q, target)[3:]
        mujoco.mj_comPos(self.model, self.data)
        linear, angular = np.zeros((3, 6)), np.zeros((3, 6))
        mujoco.mj_jacBody(self.model, self.data, linear, angular, self.body_id)
        x, y, z = phi
        cross = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
        angle = np.linalg.norm(phi)
        coefficient = (1 / 12 + angle**2 / 720 if angle < 1e-5 else
                       1 / angle**2 - 1 / (2 * angle * np.tan(angle / 2)))
        rotation_jacobian = np.eye(3) - .5 * cross + coefficient * cross @ cross
        return np.concatenate((linear, rotation_jacobian @ angular), axis=0)

    def solve(self, target_pos, target_wxyz, state):
        reference = np.asarray(state, dtype=np.float64).reshape(6).copy()
        position = np.asarray(target_pos, dtype=np.float64).reshape(3)
        quaternion = np.asarray(target_wxyz, dtype=np.float64).reshape(4)
        if (not np.isfinite(reference).all() or not np.isfinite(position).all()
                or not np.isfinite(quaternion).all() or np.linalg.norm(quaternion) == 0):
            raise ValueError("SLSQP IK requires finite joints and a valid target pose")
        target = np.eye(4)
        target[:3, :3] = Rotation.from_quat(quaternion[[1, 2, 3, 0]]).as_matrix()
        target[:3, 3] = position
        initial = self.initializer.solve(position, quaternion, reference)
        self.last_result = minimize(
            lambda q: .5 * np.dot(q - reference, q - reference),
            np.clip(initial, self.lower, self.upper),
            jac=lambda q: q - reference,
            method="SLSQP", bounds=Bounds(self.lower, self.upper),
            constraints={"type": "eq",
                         "fun": lambda q: self.pose_error(q, target),
                         "jac": lambda q: self.pose_jacobian(q, target)},
            options={"ftol": 1e-10, "maxiter": 500},
        )
        if not np.isfinite(self.last_result.x).all():
            raise RuntimeError("SLSQP IK returned non-finite joint angles")
        return np.clip(self.last_result.x, self.lower, self.upper)
