"""YAM testing-time IK, using the Pinocchio/Pink tasks from robot(1).py."""

from __future__ import annotations

import mujoco
import numpy as np
import pinocchio as pin
from pink import Configuration, solve_ik
from pink.tasks import FrameTask, PostureTask


def _body_pose(data, body_id):
    return pin.SE3(data.xmat[body_id].reshape(3, 3), data.xpos[body_id])


def _pinocchio_model(source, target_link):
    """Copy a six-hinge MJCF chain exactly, including joint axes and anchors.

    MuJoCo is used only here to read the existing YAM geometry. All subsequent
    controller FK, Jacobians and IK use Pinocchio. Pinocchio coordinates are
    physical joint angles minus the MJCF reference angles (qpos0).
    """
    if source.nq != 6 or source.nv != 6 or source.njnt != 6:
        raise ValueError("Expected a six-DoF YAM arm model")
    if np.any(source.jnt_type != mujoco.mjtJoint.mjJNT_HINGE):
        raise ValueError("YAM IK requires six scalar hinge joints")
    if not np.array_equal(source.jnt_qposadr, np.arange(6)):
        raise ValueError("Unexpected YAM joint order")
    target_id = source.body(target_link).id
    data = mujoco.MjData(source)
    mujoco.mj_kinematics(source, data)
    model = pin.Model()
    parent_joint = 0
    world_from_parent = pin.SE3.Identity()
    previous_body = 0
    for index in range(6):
        body_id = int(source.jnt_bodyid[index])
        ancestor = int(source.body_parentid[body_id])
        while ancestor != previous_body and ancestor != 0:
            ancestor = int(source.body_parentid[ancestor])
        if ancestor != previous_body:
            raise ValueError("YAM joints must form a single serial chain")
        world_from_joint = _body_pose(data, body_id) * pin.SE3(
            np.eye(3), source.jnt_pos[index]
        )
        parent_joint = model.addJoint(
            parent_joint,
            pin.JointModelRevoluteUnaligned(source.jnt_axis[index]),
            world_from_parent.inverse() * world_from_joint,
            source.joint(index).name,
        )
        world_from_parent = world_from_joint
        previous_body = body_id
    ancestor = target_id
    while ancestor != previous_body and ancestor != 0:
        ancestor = int(source.body_parentid[ancestor])
    if ancestor != previous_body:
        raise ValueError("Target link must belong to the final YAM joint")
    model.addFrame(pin.Frame(
        target_link, parent_joint, 0,
        world_from_parent.inverse() * _body_pose(data, target_id),
        pin.FrameType.BODY,
    ))
    reference = source.qpos0.copy()
    model.lowerPositionLimit[:] = np.where(
        source.jnt_limited, source.jnt_range[:, 0] - reference, -np.inf
    )
    model.upperPositionLimit[:] = np.where(
        source.jnt_limited, source.jnt_range[:, 1] - reference, np.inf
    )
    # MJCF has no joint velocity limits. Do not invent an online speed cap.
    model.velocityLimit[:] = np.inf
    return model, reference


class PinocchioArmController:
    """Five Pink/DAQP steps with task costs tuned for YAM base compensation.

    The physical state is the smoothness target; IK starts from its bounded projection.
    The fixed YAM home posture replaces the humanoid's default_joint_positions.json. Position
    limits come directly from the MJCF, without an additional safety margin.
    """

    dt = 0.05
    iterations = 5

    def __init__(self, home, *, model, target_link):
        self.model, self.reference = _pinocchio_model(model, target_link)
        self.data = self.model.createData()
        self.target_link = target_link
        self.frame_id = self.model.getFrameId(target_link)
        self.home = np.asarray(home, dtype=np.float64).reshape(6).copy()
        if not np.isfinite(self.home).all():
            raise ValueError("YAM home posture must contain finite joint angles")
        self.frame_task = FrameTask(
            frame=target_link, position_cost=100.0, orientation_cost=8.0,
            lm_damping=0.0, gain=1.0,
        )
        self.smoothness_task = PostureTask(cost=0.2, lm_damping=0.0, gain=0.2)
        self.regularization_task = PostureTask(cost=0.05, lm_damping=0.0, gain=0.2)
        self.regularization_task.set_target(self.home - self.reference)
        self.tasks = [self.frame_task, self.smoothness_task, self.regularization_task]

    def fk(self, joints):
        """World-frame link position and quaternion (w, x, y, z), without clipping."""
        q = np.asarray(joints, dtype=np.float64).reshape(6) - self.reference
        if not np.isfinite(q).all():
            raise ValueError("FK joints must contain finite angles")
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        pose = self.data.oMf[self.frame_id]
        xyzw = pin.Quaternion(pose.rotation).coeffs()
        return pose.translation.copy(), xyzw[[3, 0, 1, 2]].copy()

    def solve(self, target_pos, target_wxyz, state):
        initial = np.asarray(state, dtype=np.float64).reshape(6) - self.reference
        if not np.isfinite(initial).all():
            raise ValueError("Pink IK state must contain finite joint angles")
        position = np.asarray(target_pos, dtype=np.float64).reshape(3)
        quaternion = np.asarray(target_wxyz, dtype=np.float64).reshape(4)
        if (not np.isfinite(position).all() or not np.isfinite(quaternion).all()
                or np.linalg.norm(quaternion) <= np.finfo(float).eps):
            raise ValueError("Pink IK target must be a finite position and valid quaternion")
        w, x, y, z = quaternion
        self.frame_task.set_target(pin.SE3(
            pin.Quaternion(w, x, y, z).normalized().matrix(),
            position,
        ))
        self.smoothness_task.set_target(initial)
        # MuJoCo's soft joint limits can allow a measured overshoot. Project
        # only the IK seed; FK and the smoothness target keep the actual state.
        configuration = Configuration(
            self.model, self.data,
            np.clip(initial, self.model.lowerPositionLimit, self.model.upperPositionLimit),
            copy_data=True, forward_kinematics=True,
        )
        for _ in range(self.iterations):
            velocity = solve_ik(
                configuration, self.tasks, dt=self.dt, solver="daqp", damping=0.0,
            )
            if not np.isfinite(velocity).all():
                raise RuntimeError("Pink IK returned a non-finite velocity")
            configuration.integrate_inplace(velocity, self.dt)
            configuration.update(np.clip(
                configuration.q,
                self.model.lowerPositionLimit,
                self.model.upperPositionLimit,
            ))
        return configuration.q.copy() + self.reference
