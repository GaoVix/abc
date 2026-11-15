import argparse
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation


def joint_distance(q, reference):
    return 0.5 * np.dot(q - reference, q - reference)


def joint_distance_jacobian(q, reference):
    return q - reference


def pose_error(q, model, data, joints, site, target):
    data.qpos[model.jnt_qposadr[joints]] = q
    mujoco.mj_kinematics(model, data)
    rotation = data.site_xmat[site, :].reshape(3, 3)
    phi = Rotation.from_matrix(rotation @ target[:3, :3].T).as_rotvec()
    return np.concatenate((data.site_xpos[site, :] - target[:3, 3], phi), axis=0)


def pose_jacobian(q, model, data, joints, site, target):
    phi = pose_error(q, model, data, joints, site, target)[3:]
    mujoco.mj_comPos(model, data)
    linear, angular = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, linear, angular, site)
    x, y, z = phi
    cross = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    angle = np.linalg.norm(phi)
    coefficient = 1 / 12 + angle ** 2 / 720 if angle < 1e-5 else (
        1 / angle ** 2 - 1 / (2 * angle * np.tan(angle / 2)))
    rotation_jacobian = np.eye(3) - 0.5 * cross + coefficient * cross @ cross
    dofs = model.jnt_dofadr[joints]
    return np.concatenate((linear[:, dofs], rotation_jacobian @ angular[:, dofs]), axis=0)


def solve_ik(model, data, side, target, q0, reference=None):
    """Solve a world-frame TCP pose (meters, radians), without joint limits."""
    joints = [model.joint(f"{side}_joint{i}").id for i in range(1, 7)]
    site = model.site(f"{side}_tcp_site").id
    q0 = np.asarray(q0, dtype=float).copy()
    reference = q0.copy() if reference is None else np.asarray(reference, dtype=float).copy()
    pose_args = (model, data, joints, site, target)
    result = minimize(
        joint_distance, q0, args=(reference,), jac=joint_distance_jacobian, method="SLSQP",
        constraints={"type": "eq", "fun": pose_error, "jac": pose_jacobian, "args": pose_args},
        options={"ftol": 1e-10, "maxiter": 500},
    )
    result.pose_error = pose_error(result.x, *pose_args)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("actions", help="Real joint angles: (N, 6), radians; first row initializes IK")
    parser.add_argument("poses", help="World-frame EEF poses: (N, 4, 4), meters")
    parser.add_argument("--side", choices=("left", "right"), default="left")
    parser.add_argument("--output", default="ik_joints.npy")
    args = parser.parse_args()
    model = mujoco.MjModel.from_xml_path(str(
        Path(__file__).resolve().parents[1] / "abc_sim/models/yam_dishrack_base.xml"))
    data = mujoco.MjData(model)
    poses = np.load(args.poses)
    joints = np.empty((len(poses), 6))
    errors = np.empty((len(poses), 2))
    previous = np.load(args.actions)[0, :].copy()
    for i, target in enumerate(poses):
        result = solve_ik(model, data, args.side, target, previous)
        previous = result.x.copy()
        joints[i, :] = previous
        errors[i, :] = (np.linalg.norm(result.pose_error[:3]) * 1000,
                        np.rad2deg(np.linalg.norm(result.pose_error[3:])))
    np.save(args.output, joints)
    print("saved:", args.output)
    print("max position error (mm):", errors[:, 0].max())
    print("max rotation error (deg):", errors[:, 1].max())
