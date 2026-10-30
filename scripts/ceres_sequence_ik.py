"""Refine a DP joint trajectory with analytic Ceres residuals."""

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np
import pyceres
from scipy.spatial.transform import Rotation

from scripts.scipy_ik import pose_error, pose_jacobian

ONE_DEGREE_RAD = np.pi / 180


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', type=Path, default=Path(__file__).resolve().parents[1] / 'abc_sim/models/yam_dishrack_base.xml')
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
    args = parser.parse_args()

    with np.load(args.input) as saved:
        source = {key: saved[key] for key in saved.files}
    model = mujoco.MjModel.from_xml_path(str(args.model))
    for side, transform in zip(('left', 'right'), source['base_poses_world']):
        body = model.body(f'{side}_arm')
        body.pos[:] = transform[:3, 3]
        body.quat[:] = Rotation.from_matrix(transform[:3, :3]).as_quat()[[3, 0, 1, 2]]

    started = time.perf_counter()
    solutions, errors, radii, scales, bounds, reports = [], [], [], [], [], []
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
                                    args.position_scale_mm, args.rotation_scale_deg, args.reference_weight,
                                    args.first_order_weight, args.first_order_window,
                                    args.second_order_weight, args.maxiter,
                                    position_weight=args.position_weight, rotation_weight=args.rotation_weight,
                                    check_gradients=args.check_gradients)
        error = pose_errors(model, side, targets, q)
        report = dict(side=side, optimizer_usable=summary.IsSolutionUsable(), summary=summary.BriefReport(),
                      initial_cost=summary.initial_cost, final_cost=summary.final_cost,
                      joint_radii_rad=None if half_width is None else radius.tolist(),
                      bound_scale_min_max=None if half_width is None else [float(scale.min()), float(scale.max())],
                      base_radius_min_max_rad=(None if half_width is None else
                                               [float(half_width[:, 0].min()), float(half_width[:, 0].max())]),
                      position_mean_max_mm=[float(error[:, 0].mean()), float(error[:, 0].max())],
                      rotation_mean_max_deg=[float(error[:, 1].mean()), float(error[:, 1].max())],
                      pose_pass_0_1_mm_deg=int(np.all(error <= .1, axis=1).sum()), frames=len(q),
                      max_joint_step_deg=float(np.rad2deg(np.abs(np.diff(q, axis=0))).max(initial=0.)),
                      max_bound_violation_rad=(None if half_width is None else
                                               float(np.maximum(np.abs(q - initial) - half_width, 0.).max())))
        solutions.append(q)
        errors.append(error)
        radii.append(np.full(6, np.inf) if half_width is None else radius)
        scales.append(scale)
        bounds.append(np.full_like(initial, np.inf) if half_width is None else half_width)
        reports.append(report)
        print(json.dumps(report, indent=2), flush=True)

    bounds = np.stack(bounds, axis=1)
    source.update(q=np.stack(solutions, axis=1), errors=np.stack(errors, axis=1),
                  solver='ceres', optimized_sides=np.array(['left', 'right']),
                  smoothness=args.first_order_weight, first_order_window=args.first_order_window,
                  acceleration_smoothness=args.second_order_weight,
                  reference_weight=args.reference_weight,
                  position_scale_mm=args.position_scale_mm, rotation_scale_deg=args.rotation_scale_deg,
                  position_weight=args.position_weight, rotation_weight=args.rotation_weight,
                  joint_radii_rad=np.stack(radii), joint_limits_enabled=not args.unlimited,
                  joint_bound_scales=np.stack(scales, axis=1),
                  joint_base_radii_rad=bounds[..., 0],
                  scale_bounds_by_dp_error=args.scale_bounds_by_dp_error and not args.unlimited,
                  model_joint_limits_enabled=False, joint_limit_margin_deg=np.nan,
                  joint_lower_bounds_rad=source['initial'] - bounds,
                  joint_upper_bounds_rad=source['initial'] + bounds)
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / 'trajectory.npz', **source)
    report = dict(seconds=time.perf_counter() - started, input=str(args.input.resolve()),
                  check_gradients=args.check_gradients, gradient_check_relative_precision=1e-6,
                  base_mode=str(source.get('base_mode', 'input')),
                  joint_bound_rule=('none' if args.unlimited else
                                    'initial[t, joint] +/- scale[t, arm] * radius[joint]; no MJCF range intersection'),
                  scale_bounds_by_dp_error=args.scale_bounds_by_dp_error and not args.unlimited,
                  bound_scale_rule=('sqrt(hypot(position_error_m / 0.01, rotation_error_rad / 0.1))'
                                    if args.scale_bounds_by_dp_error and not args.unlimited else None),
                  radius_rule=(None if args.unlimited else 'explicit --joint-radii-rad'
                               if args.joint_radii_rad is not None else
                               'first_radius * sqrt(cumulative_link_length[0] / cumulative_link_length[j]); J6 uses J5 length'),
                  explicit_joint_radii_rad=None if args.unlimited else args.joint_radii_rad,
                  first_joint_radius_rad=None if args.unlimited else args.first_joint_radius_rad,
                  position_scale_mm=args.position_scale_mm, rotation_scale_deg=args.rotation_scale_deg,
                  position_weight=args.position_weight, rotation_weight=args.rotation_weight,
                  reference_weight=args.reference_weight, first_order_weight=args.first_order_weight,
                  first_order_window=args.first_order_window,
                  second_order_weight=args.second_order_weight, arms=reports)
    (args.output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'Saved to {args.output}; {report["seconds"]:.2f}s', flush=True)


if __name__ == '__main__':
    main()
