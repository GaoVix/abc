"""Select a DP100 grid path, then solve the whole trajectory with SLSQP."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation

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


def select_path(candidates):
    parents = []
    cost = np.zeros(len(candidates[0]))
    for previous, current in zip(candidates, candidates[1:]):
        transition = cdist(previous, current, "sqeuclidean") + cost[:, None]
        parent = np.argmin(transition, axis=0)
        parents.append(parent)
        cost = transition[parent, np.arange(len(current))]
    choice = np.empty(len(candidates), dtype=np.int32)
    choice[-1] = np.argmin(cost)
    for frame in range(len(candidates) - 1, 0, -1):
        choice[frame - 1] = parents[frame - 1][choice[frame]]
    return choice


class Trajectory:
    def __init__(self, model, data, side, targets, reference, smoothness):
        self.model, self.data, self.targets, self.reference = model, data, targets, reference
        self.weight = smoothness / np.deg2rad(1.0) ** 2
        self.joints = [model.joint(f"{side}_joint{i}").id for i in range(1, 7)]
        self.site = model.site(f"{side}_tcp_site").id
        self.residual = np.empty((len(reference), 6))
        self.jac = np.zeros((reference.size, reference.size))
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


def solve_sequence(model, data, side, targets, grid, smoothness=1.0, camera_frame=False):
    """Solve TCP poses against the real-data median bases; targets are world-frame by default."""
    world_from_camera = set_median_bases(model, data)
    if camera_frame:
        targets = world_from_camera @ targets
    base = data.body(f"{side}_arm")
    base_from_world = np.eye(4)
    rotation = base.xmat.reshape(3, 3)
    base_from_world[:3, :3] = rotation.T
    base_from_world[:3, 3] = -rotation.T @ base.xpos

    candidates = []
    for frame, target in enumerate(targets):
        for radius in (1, 2, 3, 4):
            found = grid.query(base_from_world @ target, k=100, neighbors=radius)
            if len(found["rows"]):
                break
        if not len(found["rows"]):
            raise ValueError(f"No grid candidates for frame {frame}")
        candidates.append(found["joints"].astype(np.float64))

    choice = select_path(candidates)
    seeds = np.stack([candidate[index] for candidate, index in zip(candidates, choice)])
    problem = Trajectory(model, data, side, targets, seeds, smoothness)
    bounds = np.tile(model.jnt_range[problem.joints], (len(seeds), 1))
    result = minimize(
        problem.objective, seeds.ravel().copy(), jac=problem.gradient,
        method="SLSQP", bounds=bounds,
        constraints={"type": "eq", "fun": problem.constraints,
                     "jac": problem.constraint_jacobian},
        options={"ftol": 1e-10, "maxiter": 500},
    )
    q = result.x.reshape(-1, 6)
    errors = np.empty((len(seeds), 2))
    for frame, (joints, target) in enumerate(zip(q, targets)):
        residual = pose_error(joints, model, data, problem.joints, problem.site, target)
        errors[frame] = (np.linalg.norm(residual[:3]) * 1000,
                         np.rad2deg(np.linalg.norm(residual[3:])))
    return {"q": q, "seeds": seeds, "ranks": choice + 1,
            "errors_mm_deg": errors, "optimizer_success": result.success,
            "iterations": result.nit, "message": result.message}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("poses", help="World-frame TCP poses, shape (N, 4, 4), in meters")
    parser.add_argument("--grid", required=True)
    parser.add_argument("--side", choices=("left", "right"), default="left")
    parser.add_argument("--smoothness", type=float, default=1.0)
    parser.add_argument("--camera-frame", action="store_true",
                        help="Input poses use the calibrated top-camera/OpenCV frame")
    parser.add_argument("--model", default=str(Path(__file__).resolve().parents[1]
                                                / "abc_sim/models/yam_dishrack_base.xml"))
    parser.add_argument("--output", default="dp100_ik.npz")
    args = parser.parse_args()
    model = mujoco.MjModel.from_xml_path(args.model)
    result = solve_sequence(model, mujoco.MjData(model), args.side, np.load(args.poses),
                            GPUEEFPoseGrid(args.grid), args.smoothness, args.camera_frame)
    np.savez(args.output, **result)
    print("saved:", args.output)
    print("pose passed:", np.all(result["errors_mm_deg"] <= 0.1, axis=1).sum(),
          "/", len(result["q"]))
