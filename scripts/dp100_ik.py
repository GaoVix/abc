"""Select a DP100 grid path, then solve whole-trajectory IK."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import torch
from scipy.optimize import Bounds, NonlinearConstraint, minimize
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation
from scipy.sparse import block_diag, csr_matrix, diags, eye, kron
from torch.func import hessian, vmap

from abc_minimal.eef_pose_grid_gpu import GPUEEFPoseGrid

if __package__:
    from .scipy_ik import pose_error, pose_jacobian
else:
    from scipy_ik import pose_error, pose_jacobian


MEDIAN_BASE = Path(__file__).resolve().parents[1] / "outputs/base_pose_median_20261026/median_base_poses.json"


def set_median_bases(model, data):
    mujoco.mj_forward(model, data)
    top = data.camera("top")
    world_from_camera = np.eye(4)
    world_from_camera[:3, :3] = top.xmat.reshape(3, 3) @ np.diag([1.0, -1.0, -1.0])
    world_from_camera[:3, 3] = top.xpos
    median = json.loads(MEDIAN_BASE.read_text())["poses"]
    for side in ("left", "right"):
        transform = world_from_camera @ np.asarray(median[side]["T_camera_from_base"])
        body = model.body(f"{side}_arm")
        body.pos[:] = transform[:3, 3]
        body.quat[:] = Rotation.from_matrix(transform[:3, :3]).as_quat()[[3, 0, 1, 2]]
    mujoco.mj_forward(model, data)
    return world_from_camera


def select_path(candidates, pose_costs):
    parents = []
    cost = pose_costs[0].copy()
    for frame, (previous, current) in enumerate(zip(candidates, candidates[1:]), 1):
        transition = cdist(previous, current, "sqeuclidean") / 0.1 ** 2 + cost[:, None]
        parent = np.argmin(transition, axis=0)
        parents.append(parent)
        cost = transition[parent, np.arange(len(current))] + pose_costs[frame]
    choice = np.empty(len(candidates), dtype=np.int32)
    choice[-1] = np.argmin(cost)
    for frame in range(len(candidates) - 1, 0, -1):
        choice[frame - 1] = parents[frame - 1][choice[frame]]
    return choice


def quaternion_product(a, b):
    return torch.cat((a[:1] * b[:1] - torch.dot(a[1:], b[1:]).reshape(1),
                      a[0] * b[1:] + b[0] * a[1:] + torch.linalg.cross(a[1:], b[1:])))


class PoseConstraintHessian:
    def __init__(self, model, data, joints, site, targets):
        qpos = model.jnt_qposadr[joints]
        previous = data.qpos[qpos].copy()
        data.qpos[qpos] = 0
        mujoco.mj_kinematics(model, data)
        anchors = data.xanchor[joints].copy()
        axes = data.xaxis[joints].copy()
        self.site_position = torch.as_tensor(data.site_xpos[site].copy(), dtype=torch.float64)
        site_rotation = data.site_xmat[site].reshape(3, 3)
        site_quaternion = Rotation.from_matrix(site_rotation).as_quat()[[3, 0, 1, 2]]
        self.site_quaternion = torch.as_tensor(site_quaternion, dtype=torch.float64)
        self.anchors = torch.as_tensor(anchors, dtype=torch.float64)
        self.axes = torch.as_tensor(axes, dtype=torch.float64)
        self.skews = torch.as_tensor(np.array([
            [[0, -z, y], [z, 0, -x], [-y, x, 0]] for x, y, z in axes
        ]), dtype=torch.float64)
        self.target_positions = torch.as_tensor(targets[:, :3, 3].copy(), dtype=torch.float64)
        target_quaternions = Rotation.from_matrix(targets[:, :3, :3]).as_quat()[:, [3, 0, 1, 2]]
        target_quaternions[:, 1:] *= -1
        self.target_quaternions = torch.as_tensor(target_quaternions.copy(), dtype=torch.float64)
        self.scale = torch.tensor([1000, 1000, 1000, 180 / np.pi,
                                   180 / np.pi, 180 / np.pi], dtype=torch.float64)
        data.qpos[qpos] = previous
        mujoco.mj_kinematics(model, data)
        self.batched_hessian = vmap(hessian(self.weighted_residual, argnums=0))

    def weighted_residual(self, q, target_position, target_quaternion, multipliers):
        rotation = torch.eye(3, dtype=q.dtype)
        position = torch.zeros(3, dtype=q.dtype)
        quaternion = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=q.dtype)
        for i in range(6):
            axis = self.axes[i]
            angle = q[i]
            joint_rotation = (torch.cos(angle) * torch.eye(3, dtype=q.dtype)
                              + (1 - torch.cos(angle)) * torch.outer(axis, axis)
                              + torch.sin(angle) * self.skews[i])
            position = position + rotation @ ((torch.eye(3, dtype=q.dtype) - joint_rotation)
                                            @ self.anchors[i])
            rotation = rotation @ joint_rotation
            joint_quaternion = torch.cat((torch.cos(angle / 2).reshape(1),
                                          axis * torch.sin(angle / 2)))
            quaternion = quaternion_product(quaternion, joint_quaternion)
        position = position + rotation @ self.site_position
        quaternion = quaternion_product(quaternion_product(quaternion, self.site_quaternion),
                                        target_quaternion)
        quaternion = torch.where(quaternion[0] < 0, -quaternion, quaternion)
        vector = quaternion[1:]
        squared_norm = torch.dot(vector, vector)
        norm = torch.sqrt(squared_norm + 1e-30)
        w = quaternion[0]
        safe_w = torch.where(torch.abs(w) > 1e-12, w, torch.ones_like(w))
        factor = torch.where(squared_norm < 1e-8,
                             2 / safe_w - 2 * squared_norm / (3 * safe_w ** 3),
                             2 * torch.atan2(norm, w) / norm)
        residual = torch.cat((position - target_position, vector * factor))
        return torch.dot(residual * self.scale, multipliers)

    def __call__(self, x, multipliers):
        q = torch.as_tensor(x.reshape(-1, 6), dtype=torch.float64)
        v = torch.as_tensor(multipliers.reshape(-1, 6), dtype=torch.float64)
        blocks = list(self.batched_hessian(q, self.target_positions,
                                           self.target_quaternions, v).numpy())
        blocks[0] = csr_matrix(blocks[0])
        return block_diag(blocks, format="csr")


class Trajectory:
    def __init__(self, model, data, side, targets, reference, smoothness):
        self.model, self.data, self.targets, self.reference = model, data, targets, reference
        self.weight = smoothness / np.deg2rad(1.0) ** 2
        self.joints = [model.joint(f"{side}_joint{i}").id for i in range(1, 7)]
        self.site = model.site(f"{side}_tcp_site").id
        self.residual = np.empty((len(reference), 6))
        self.jac = np.zeros((reference.size, reference.size))
        frames = len(reference)
        diagonal = np.ones(frames)
        diagonal[:-1] += self.weight
        diagonal[1:] += self.weight
        off_diagonal = np.full(frames - 1, -self.weight)
        self.hessian = kron(diags((off_diagonal, diagonal, off_diagonal), (-1, 0, 1)),
                            eye(6), format="csr")
        self.constraint_scale = diags(np.tile([1000, 1000, 1000, 180 / np.pi,
                                                180 / np.pi, 180 / np.pi], frames), format="csr")
        self.last_x = None

    def objective(self, x):
        q = x.reshape(-1, 6)
        return 0.5 * (np.sum((q - self.reference) ** 2)
                      + self.weight * np.sum(np.diff(q, axis=0) ** 2))

    def gradient(self, x):
        q = x.reshape(-1, 6)
        difference = np.diff(q, axis=0)
        gradient = q - self.reference
        gradient[:-1] -= self.weight * difference
        gradient[1:] += self.weight * difference
        return gradient.ravel()

    def evaluate(self, x):
        if self.last_x is not None and np.array_equal(x, self.last_x):
            return
        for frame, (q, target) in enumerate(zip(x.reshape(-1, 6), self.targets)):
            args = (self.model, self.data, self.joints, self.site, target)
            self.residual[frame] = pose_error(q, *args)
            block = slice(6 * frame, 6 * frame + 6)
            self.jac[block, block] = pose_jacobian(q, *args)
        self.last_x = x.copy()

    def constraints(self, x):
        self.evaluate(x)
        return self.residual.ravel().copy()

    def constraint_jacobian(self, x):
        self.evaluate(x)
        return self.jac

    def sparse_constraint_jacobian(self, x):
        self.evaluate(x)
        blocks = [self.jac[i:i + 6, i:i + 6]
                  for i in range(0, self.reference.size, 6)]
        blocks[0] = csr_matrix(blocks[0])
        return self.constraint_scale @ block_diag(blocks, format="csr")

    def scaled_constraints(self, x):
        return self.constraint_scale @ self.constraints(x)

    def objective_hessian(self, x):
        return self.hessian


def solve_sequence(model, data, side, targets, grid, smoothness=1.0, camera_frame=False,
                   solver="slsqp", maxiter=500):
    """Solve TCP poses against the real-data median bases; targets are world-frame by default."""
    world_from_camera = set_median_bases(model, data)
    if camera_frame:
        targets = world_from_camera @ targets
    base = data.body(f"{side}_arm")
    base_from_world = np.eye(4)
    rotation = base.xmat.reshape(3, 3)
    base_from_world[:3, :3] = rotation.T
    base_from_world[:3, 3] = -rotation.T @ base.xpos

    candidates, pose_costs = [], []
    for frame, target in enumerate(targets):
        for radius in (1, 2, 3, 4):
            found = grid.query(base_from_world @ target, k=100, neighbors=radius)
            if len(found["rows"]):
                break
        if not len(found["rows"]):
            raise ValueError(f"No grid candidates for frame {frame}")
        candidates.append(found["joints"].astype(np.float64))
        pose_costs.append((found["position_error_m"] / 0.01) ** 2
                          + (found["rotation_error_rad"] / 0.1) ** 2)

    choice = select_path(candidates, pose_costs)
    seeds = np.stack([candidate[index] for candidate, index in zip(candidates, choice)])
    problem = Trajectory(model, data, side, targets, seeds, smoothness)
    bounds = np.tile(model.jnt_range[problem.joints], (len(seeds), 1))
    if solver == "trust-constr":
        constraint_hessian = PoseConstraintHessian(model, data, problem.joints,
                                                   problem.site, targets)
        constraint = NonlinearConstraint(problem.scaled_constraints, 0, 0,
                                        jac=problem.sparse_constraint_jacobian,
                                        hess=constraint_hessian)
        result = minimize(problem.objective, seeds.ravel().copy(), jac=problem.gradient,
                          hess=problem.objective_hessian, method="trust-constr",
                          bounds=Bounds(bounds[:, 0], bounds[:, 1], keep_feasible=True),
                          constraints=[constraint],
                          options={"maxiter": maxiter, "gtol": 1e-8, "xtol": 1e-10,
                                   "sparse_jacobian": True})
    else:
        result = minimize(
            problem.objective, seeds.ravel().copy(), jac=problem.gradient,
            method="SLSQP", bounds=bounds,
            constraints={"type": "eq", "fun": problem.constraints,
                         "jac": problem.constraint_jacobian},
            options={"ftol": 1e-10, "maxiter": maxiter},
        )
    q = result.x.reshape(-1, 6)
    errors = np.empty((len(seeds), 2))
    for frame, (joints, target) in enumerate(zip(q, targets)):
        residual = pose_error(joints, model, data, problem.joints, problem.site, target)
        errors[frame] = (np.linalg.norm(residual[:3]) * 1000,
                         np.rad2deg(np.linalg.norm(residual[3:])))
    within_bounds = ((q.ravel() >= bounds[:, 0] - 1e-10)
                     & (q.ravel() <= bounds[:, 1] + 1e-10)).reshape(-1, 6).all(axis=1)
    return {"q": q, "seeds": seeds, "ranks": choice + 1,
            "errors_mm_deg": errors, "optimizer_success": result.success,
            "joint_bounds_passed": within_bounds,
            "iterations": result.nit, "message": result.message}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("poses", help="World-frame TCP poses, shape (N, 4, 4), in meters")
    parser.add_argument("--grid", required=True)
    parser.add_argument("--side", choices=("left", "right"), default="left")
    parser.add_argument("--smoothness", type=float, default=1.0)
    parser.add_argument("--solver", choices=("slsqp", "trust-constr"), default="slsqp")
    parser.add_argument("--maxiter", type=int, default=500)
    parser.add_argument("--camera-frame", action="store_true",
                        help="Input poses use the calibrated top-camera/OpenCV frame")
    parser.add_argument("--model", default=str(Path(__file__).resolve().parents[1]
                                                / "abc_sim/models/yam_dishrack_base.xml"))
    parser.add_argument("--output", default="dp100_ik.npz")
    args = parser.parse_args()
    model = mujoco.MjModel.from_xml_path(args.model)
    result = solve_sequence(model, mujoco.MjData(model), args.side, np.load(args.poses),
                            GPUEEFPoseGrid(args.grid), args.smoothness, args.camera_frame,
                            args.solver, args.maxiter)
    np.savez(args.output, **result)
    print("saved:", args.output)
    print("pose passed:", np.all(result["errors_mm_deg"] <= 0.1, axis=1).sum(),
          "/", len(result["q"]))
    print("pose and bounds passed:",
          (np.all(result["errors_mm_deg"] <= 0.1, axis=1)
           & result["joint_bounds_passed"]).sum(), "/", len(result["q"]))
