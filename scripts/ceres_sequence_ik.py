"""Refine a DP joint trajectory with analytic Ceres residuals."""

import argparse
from pathlib import Path

import mujoco
import numpy as np
import pyceres
from scipy.spatial.transform import Rotation

from scripts.scipy_ik import pose_error, pose_jacobian

ONE_DEGREE_RAD = np.pi / 180
DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / 'abc_sim/models/yam_dishrack_base.xml'


class PositionCost(pyceres.CostFunction):
    def __init__(self, model, data, side, target, scale):
        super().__init__()
        self.set_num_residuals(3)
        self.set_parameter_block_sizes([6])
        joints = [model.joint(f'{side}_joint{i}').id for i in range(1, 7)]
        self.args = (model, data, joints, model.site(f'{side}_tcp_site').id, target)
        self.scale = scale

    def Evaluate(self, parameters, residuals, jacobians):
        residuals[:] = self.scale * pose_error(parameters[0], *self.args)[:3]
        if jacobians is not None and jacobians[0] is not None:
            jacobians[0][:] = (self.scale * pose_jacobian(parameters[0], *self.args)[:3]).ravel()
        return True


class RotationCost(pyceres.CostFunction):
    def __init__(self, model, data, side, target, scale):
        super().__init__()
        self.set_num_residuals(3)
        self.set_parameter_block_sizes([6])
        joints = [model.joint(f'{side}_joint{i}').id for i in range(1, 7)]
        self.args = (model, data, joints, model.site(f'{side}_tcp_site').id, target)
        self.scale = scale

    def Evaluate(self, parameters, residuals, jacobians):
        residuals[:] = self.scale * pose_error(parameters[0], *self.args)[3:]
        if jacobians is not None and jacobians[0] is not None:
            jacobians[0][:] = (self.scale * pose_jacobian(parameters[0], *self.args)[3:]).ravel()
        return True


class JointCost(pyceres.CostFunction):
    def __init__(self, coefficients, offset, scale):
        super().__init__()
        self.set_num_residuals(6)
        self.set_parameter_block_sizes([6] * len(coefficients))
        self.coefficients = np.asarray(coefficients) * scale
        self.offset = np.asarray(offset) * scale
        self.blocks = [coefficient * np.eye(6).ravel() for coefficient in self.coefficients]

    def Evaluate(self, parameters, residuals, jacobians):
        residuals[:] = sum(c * q for c, q in zip(self.coefficients, parameters)) - self.offset
        if jacobians is not None:
            for jacobian, block in zip(jacobians, self.blocks):
                if jacobian is not None:
                    jacobian[:] = block
        return True


def joint_radii(model, side, first_radius=.2):
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    joints = [model.joint(f'{side}_joint{i}').id for i in range(1, 7)]
    points = np.vstack((data.xanchor[joints], data.site(f'{side}_tcp_site').xpos))
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    remaining = np.cumsum(lengths[::-1])[::-1]
    remaining[-1] = remaining[-2]
    return first_radius * np.sqrt(remaining[0] / remaining)


def pose_errors(model, side, targets, q):
    data = mujoco.MjData(model)
    joints = [model.joint(f'{side}_joint{i}').id for i in range(1, 7)]
    site = model.site(f'{side}_tcp_site').id
    residuals = np.array([pose_error(joints_q, model, data, joints, site, target)
                          for joints_q, target in zip(q, targets)])
    return np.column_stack((1000 * np.linalg.norm(residuals[:, :3], axis=1),
                            np.rad2deg(np.linalg.norm(residuals[:, 3:], axis=1))))


def solve_sequence(model, side, targets, initial, reference, half_width,
                   position_scale_mm=.1, rotation_scale_deg=.1,
                   reference_weight=1., first_order_weight=1., first_order_window=1,
                   second_order_weight=1., maxiter=500,
                   position_weight=1., rotation_weight=4., check_gradients=False):
    q = initial.astype(np.float64)
    if half_width is not None:
        lower, upper = q - half_width, q + half_width
    blocks = list(q)
    data = mujoco.MjData(model)
    problem = pyceres.Problem()
    position_scale = np.sqrt(position_weight) * 1000 / position_scale_mm
    rotation_scale = np.sqrt(rotation_weight) / np.deg2rad(rotation_scale_deg)
    reference_scale = np.sqrt(reference_weight)
    first_order_scale = np.sqrt(first_order_weight / max(first_order_window, 1)) / ONE_DEGREE_RAD
    second_order_scale = np.sqrt(second_order_weight) / ONE_DEGREE_RAD
    for frame, (block, target) in enumerate(zip(blocks, targets)):
        problem.add_residual_block(PositionCost(model, data, side, target, position_scale), None, [block])
        problem.add_residual_block(RotationCost(model, data, side, target, rotation_scale), None, [block])
        problem.add_residual_block(JointCost([1], reference[frame], reference_scale), None, [block])
        for previous in range(max(0, frame - first_order_window), frame):
            problem.add_residual_block(JointCost([-1, 1], 0., first_order_scale), None,
                                       [blocks[previous], block])
        if frame > 1:
            problem.add_residual_block(JointCost([1, -2, 1], 0., second_order_scale),
                                       None, blocks[frame - 2:frame + 1])
        if half_width is not None:
            fixed = np.flatnonzero(lower[frame] == upper[frame]).tolist()
            if fixed:
                problem.set_manifold(block, pyceres.SubsetManifold(6, fixed))
            for joint in range(6):
                if joint not in fixed:
                    problem.set_parameter_lower_bound(block, joint, lower[frame, joint])
                    problem.set_parameter_upper_bound(block, joint, upper[frame, joint])
    options = pyceres.SolverOptions()
    options.linear_solver_type = pyceres.LinearSolverType.SPARSE_NORMAL_CHOLESKY
    options.check_gradients = check_gradients
    options.gradient_check_relative_precision = 1e-6
    options.max_num_iterations = maxiter
    options.function_tolerance = 1e-10
    options.parameter_tolerance = 1e-10
    summary = pyceres.SolverSummary()
    pyceres.solve(options, problem, summary)
    return q, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', type=Path, help='Defaults to the input model, otherwise yam_dishrack_base.xml')
    parser.add_argument('--first-joint-radius-rad', type=float, default=.2)
    parser.add_argument('--joint-radii-rad', type=float, nargs=6)
    parser.add_argument('--unlimited', action='store_true')
    parser.add_argument('--scale-bounds-by-dp-error', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--position-scale-mm', type=float, default=.1)
    parser.add_argument('--rotation-scale-deg', type=float, default=.1)
    parser.add_argument('--position-weight', type=float, default=1.)
    parser.add_argument('--rotation-weight', type=float, default=4.)
    parser.add_argument('--reference-weight', type=float, default=1.)
    parser.add_argument('--first-order-weight', type=float, default=1.)
    parser.add_argument('--first-order-window', type=int, default=1,
                        help='Number of previous frames; 0 disables first-order smoothing')
    parser.add_argument('--second-order-weight', type=float, default=1.)
    parser.add_argument('--maxiter', type=int, default=500)
    parser.add_argument('--check-gradients', action='store_true')
    args = parser.parse_args(argv)
    weights = dict(position_scale_mm=args.position_scale_mm, rotation_scale_deg=args.rotation_scale_deg,
                   position_weight=args.position_weight, rotation_weight=args.rotation_weight,
                   reference_weight=args.reference_weight, first_order_weight=args.first_order_weight,
                   first_order_window=args.first_order_window, second_order_weight=args.second_order_weight)

    with np.load(args.input) as saved:
        source = {key: saved[key] for key in saved.files}
    args.model = args.model or Path(str(source.get('model_path', DEFAULT_MODEL_PATH)))
    model = mujoco.MjModel.from_xml_path(str(args.model))
    for side, transform in zip(('left', 'right'), source['base_poses_world']):
        body = model.body(f'{side}_arm')
        body.pos[:] = transform[:3, 3]
        body.quat[:] = Rotation.from_matrix(transform[:3, :3]).as_quat()[[3, 0, 1, 2]]

    solutions, errors, bounds, usable, success = [], [], [], [], []
    for arm, side in enumerate(('left', 'right')):
        radius = None
        if not args.unlimited:
            radius = (joint_radii(model, side, args.first_joint_radius_rad) if args.joint_radii_rad is None
                      else np.asarray(args.joint_radii_rad))
        initial, reference = source['initial'][:, arm], source['reference'][:, arm]
        targets = source['targets_world'][:, arm]
        scale = np.ones(len(initial))
        if args.scale_bounds_by_dp_error and not args.unlimited:
            dp_error = pose_errors(model, side, targets, initial)
            scale = np.sqrt(np.hypot(dp_error[:, 0] / 10., np.deg2rad(dp_error[:, 1]) / .1))
        half_width = None if args.unlimited else scale[:, None] * radius
        q, summary = solve_sequence(model, side, targets, initial, reference, half_width,
                                    maxiter=args.maxiter, check_gradients=args.check_gradients, **weights)
        solutions.append(q)
        errors.append(pose_errors(model, side, targets, q))
        bounds.append(np.full_like(initial, np.inf) if half_width is None else half_width)
        usable.append(summary.IsSolutionUsable())
        success.append(summary.termination_type == pyceres.TerminationType.CONVERGENCE)

    bounds = np.stack(bounds, axis=1)
    source.update(q=np.stack(solutions, axis=1), errors=np.stack(errors, axis=1),
                  solver='ceres', model_path=str(args.model.resolve()), **weights,
                  ceres_usable=np.array(usable), solver_success=np.array(success),
                  joint_limits_enabled=not args.unlimited,
                  joint_lower_bounds_rad=source['initial'] - bounds,
                  joint_upper_bounds_rad=source['initial'] + bounds)
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / 'trajectory.npz', **source)
    print(f'Saved to {args.output / "trajectory.npz"}', flush=True)


if __name__ == '__main__':
    main()
