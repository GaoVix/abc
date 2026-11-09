"""Sequence IK with exact EEF constraints and the Ceres joint objective/bounds."""

import argparse
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import Bounds, NonlinearConstraint, minimize
from scipy.sparse import block_diag, csr_array, diags, eye, kron
from scipy.spatial.transform import Rotation

from scripts.ceres_sequence_ik import DEFAULT_MODEL_PATH, ONE_DEGREE_RAD, joint_radii, pose_errors
from scripts.scipy_ik import pose_error


def skew(vectors):
    x, y, z = np.moveaxis(vectors, -1, 0)
    zero = np.zeros_like(x)
    return np.stack((zero, -z, y, z, zero, -x, -y, x, zero), axis=-1).reshape(*vectors.shape[:-1], 3, 3)


def rotation_coefficients(squared_angle):
    coefficient, derivative = np.empty_like(squared_angle), np.empty_like(squared_angle)
    small = squared_angle < .01
    s = squared_angle[small]
    coefficient[small] = 1/12 + s*(1/720 + s*(1/30240 + s/1209600))
    derivative[small] = 1/360 + s*(1/7560 + s*(1/201600 + s/5987520))
    angle = np.sqrt(squared_angle[~small])
    half_cotangent = .5*angle/np.tan(.5*angle)
    coefficient[~small] = (1-half_cotangent)/angle**2
    derivative[~small] = (-2+half_cotangent+.25*angle**2/np.sin(.5*angle)**2)/angle**4
    return coefficient, derivative


class SequenceIK:
    def __init__(self, model, side, targets, reference, position_scale_mm=.1, rotation_scale_deg=.1,
                 position_weight=1., rotation_weight=4., reference_weight=1., first_order_weight=1.,
                 first_order_window=1, second_order_weight=1.):
        self.model, self.targets = model, targets
        self.reference = reference
        self.data = mujoco.MjData(model)
        self.joints = [model.joint(f'{side}_joint{i}').id for i in range(1, 7)]
        self.dofs = model.jnt_dofadr[self.joints]
        self.site_jacobian = np.empty((2, 3, model.nv))
        self.site = model.site(f'{side}_tcp_site').id
        self.pose_scale = np.repeat([np.sqrt(position_weight) * 1000 / position_scale_mm,
                                    np.sqrt(rotation_weight) / np.deg2rad(rotation_scale_deg)], 3)
        self.reference_weight = reference_weight
        self.first_weight = first_order_weight / max(first_order_window, 1) / ONE_DEGREE_RAD**2
        self.second_weight = second_order_weight / ONE_DEGREE_RAD**2
        n = len(reference)
        self.window = min(first_order_window, n - 1)
        hessian = reference_weight * eye(n)
        for lag in range(1, self.window + 1):
            difference = diags([-np.ones(n - lag), np.ones(n - lag)], [0, lag], shape=(n - lag, n))
            hessian += self.first_weight * (difference.T @ difference)
        if n > 2:
            difference = diags([np.ones(n - 2), -2 * np.ones(n - 2), np.ones(n - 2)],
                               [0, 1, 2], shape=(n - 2, n))
            hessian += self.second_weight * (difference.T @ difference)
        self.hessian = kron(hessian, eye(6), format='csr')
        self.last_x = None

    def joint_cost(self, x):
        q = x.reshape(-1, 6)
        reference_error = q - self.reference
        cost = .5 * self.reference_weight * np.sum(reference_error**2)
        gradient = self.reference_weight * reference_error
        for lag in range(1, self.window + 1):
            velocity = q[lag:] - q[:-lag]
            cost += .5 * self.first_weight * np.sum(velocity**2)
            gradient[lag:] += self.first_weight * velocity
            gradient[:-lag] -= self.first_weight * velocity
        acceleration = q[2:] - 2 * q[1:-1] + q[:-2]
        cost += .5 * self.second_weight * np.sum(acceleration**2)
        gradient[:-2] += self.second_weight * acceleration
        gradient[1:-1] -= 2 * self.second_weight * acceleration
        gradient[2:] += self.second_weight * acceleration
        return cost, gradient.ravel()

    def evaluate(self, x):
        if self.last_x is not None and np.array_equal(x, self.last_x):
            return
        residuals, jacobians = [], []
        for q, target in zip(x.reshape(-1, 6), self.targets):
            args = (self.model, self.data, self.joints, self.site, target)
            residuals.append(pose_error(q, *args))
            mujoco.mj_comPos(self.model, self.data)
            linear, angular = self.site_jacobian
            mujoco.mj_jacSite(self.model, self.data, linear, angular, self.site)
            jacobians.append(np.concatenate((linear[:, self.dofs], angular[:, self.dofs])))
        residuals = np.asarray(residuals)
        self.geometric_jacobians = np.asarray(jacobians)
        self.rotation_error = residuals[:, 3:]
        cross = skew(self.rotation_error)
        coefficient, _ = rotation_coefficients(np.sum(self.rotation_error**2, axis=1))
        self.rotation_log_jacobian = np.eye(3) - .5*cross + coefficient[:, None, None]*(cross @ cross)
        jacobians = self.geometric_jacobians.copy()
        jacobians[:, 3:] = self.rotation_log_jacobian @ jacobians[:, 3:]
        self.pose = residuals * self.pose_scale
        self.pose_jac = jacobians * self.pose_scale[None, :, None]
        self.last_x = x.copy()

    def constraints(self, x):
        self.evaluate(x)
        return self.pose.ravel()

    def constraint_jacobian(self, x):
        self.evaluate(x)
        return block_diag([csr_array(block) for block in self.pose_jac], format='csr')

    def constraint_hessian(self, x, multipliers):
        self.evaluate(x)
        multipliers = multipliers.reshape(-1, 6) * self.pose_scale
        linear = self.geometric_jacobians[:, :3].transpose(0, 2, 1)
        angular = self.geometric_jacobians[:, 3:].transpose(0, 2, 1)
        hessian = np.empty((len(self.targets), 6, 6))
        rows, columns = np.triu_indices(6)
        # Serial hinge chain: dJp_i/dq_j = w_min(i,j) x Jp_max(i,j).
        position = np.cross(angular[:, rows], linear[:, columns])
        values = np.einsum('nc,nic->ni', multipliers[:, :3], position)
        hessian[:, rows, columns] = values
        hessian[:, columns, rows] = values

        phi = self.rotation_error
        cross = skew(phi)
        coefficient, derivative = rotation_coefficients(np.sum(phi**2, axis=1))
        phi_jacobian = (self.rotation_log_jacobian @ self.geometric_jacobians[:, 3:]).transpose(0, 2, 1)
        cross_derivative = skew(phi_jacobian)
        coefficient_derivative = derivative[:, None] * np.einsum('nc,njc->nj', phi, phi_jacobian)
        log_jacobian_derivative = (-.5*cross_derivative
                                  + coefficient[:, None, None, None]*(cross_derivative @ cross[:, None]
                                                                     + cross[:, None] @ cross_derivative)
                                  + coefficient_derivative[:, :, None, None]*(cross @ cross)[:, None])
        weighted_derivative = np.einsum('na,njab->njb', multipliers[:, 3:], log_jacobian_derivative)
        hessian += np.einsum('njb,nib->nij', weighted_derivative, angular)
        # Only upstream joints rotate a joint's world-frame axis.
        axis_derivative = np.cross(angular[:, None], angular[:, :, None])
        axis_derivative *= np.tril(np.ones((6, 6)), -1)[None, :, :, None]
        weighted_rotation = np.einsum('na,nab->nb', multipliers[:, 3:], self.rotation_log_jacobian)
        hessian += np.einsum('nb,nijb->nij', weighted_rotation, axis_derivative)
        hessian = .5 * (hessian + hessian.transpose(0, 2, 1))
        return block_diag([csr_array(block) for block in hessian], format='csr')


def solve_sequence(model, side, targets, initial, reference, lower, upper, maxiter=500, callback=None, **weights):
    problem = SequenceIK(model, side, targets, reference, **weights)
    equality = NonlinearConstraint(problem.constraints, 0., 0., jac=problem.constraint_jacobian,
                                   hess=problem.constraint_hessian)
    return minimize(problem.joint_cost, initial.ravel(), method='trust-constr', jac=True,
                    hess=lambda x: problem.hessian, constraints=equality,
                    bounds=Bounds(lower.ravel(), upper.ravel(), keep_feasible=True) if lower is not None else None,
                    callback=callback,
                    options=dict(maxiter=maxiter, gtol=1e-8, xtol=1e-10, barrier_tol=1e-10,
                                 sparse_jacobian=True, factorization_method='AugmentedSystem'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='DP input or Ceres trajectory.npz; Ceres q becomes init/ref')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', type=Path, help='Defaults to the input model, otherwise yam_dishrack_base.xml')
    parser.add_argument('--first-joint-radius-rad', type=float, default=.2, help='Used when calculating new bounds')
    parser.add_argument('--recalculate-bounds', action='store_true', help='Recalculate bounds around the input joints using their EEF errors')
    parser.add_argument('--unlimited', action='store_true')
    weights = dict(position_scale_mm=.1, rotation_scale_deg=.1, position_weight=1., rotation_weight=4.,
                   reference_weight=1., first_order_weight=1., first_order_window=1, second_order_weight=1.)
    for name, default in weights.items():
        parser.add_argument('--' + name.replace('_', '-'), type=type(default), help=f'Input value, otherwise {default}')
    parser.add_argument('--maxiter', type=int, default=500)
    args = parser.parse_args()
    with np.load(args.input) as saved:
        source = {key: saved[key] for key in saved.files}
    saved_names = dict(first_order_weight='smoothness', second_order_weight='acceleration_smoothness')
    for name, default in weights.items():
        value = getattr(args, name)
        if value is None:
            value = source.get(name, source.get(saved_names.get(name, name), default))
        weights[name] = type(default)(value)
    args.model = args.model or Path(str(source.get('model_path', DEFAULT_MODEL_PATH)))
    from_ceres = str(source.get('solver', '')) == 'ceres'
    if from_ceres:
        source['dp_initial'] = source['initial']
        source['initial'] = source['q']
        source['reference'] = source['q']
        source['seed_errors'] = source['errors']
    limited = not args.unlimited and (not from_ceres or bool(source['joint_limits_enabled']))
    model = mujoco.MjModel.from_xml_path(str(args.model))
    for side, transform in zip(('left', 'right'), source['base_poses_world']):
        body = model.body(f'{side}_arm')
        body.pos[:] = transform[:3, 3]
        body.quat[:] = Rotation.from_matrix(transform[:3, :3]).as_quat()[[3, 0, 1, 2]]

    args.output.mkdir(parents=True, exist_ok=True)
    solutions, errors, lower_bounds, upper_bounds, success = [], [], [], [], []
    for arm, side in enumerate(('left', 'right')):
        initial, reference = source['initial'][:, arm], source['reference'][:, arm]
        targets = source['targets_world'][:, arm]
        if from_ceres and limited and not args.recalculate_bounds:
            lower, upper = source['joint_lower_bounds_rad'][:, arm], source['joint_upper_bounds_rad'][:, arm]
        elif limited:
            seed_error = pose_errors(model, side, targets, initial)
            scale = np.sqrt(np.hypot(seed_error[:, 0] / 10., np.deg2rad(seed_error[:, 1]) / .1))
            half_width = scale[:, None] * joint_radii(model, side, args.first_joint_radius_rad)
            lower, upper = initial - half_width, initial + half_width
        else:
            lower, upper = np.full_like(initial, -np.inf), np.full_like(initial, np.inf)
        result = solve_sequence(model, side, targets, initial, reference,
                                lower if limited else None, upper if limited else None,
                                maxiter=args.maxiter, **weights)
        q = result.x.reshape(-1, 6)
        solutions.append(q)
        errors.append(pose_errors(model, side, targets, q))
        lower_bounds.append(lower)
        upper_bounds.append(upper)
        success.append(result.success)

    lower_bounds, upper_bounds = np.stack(lower_bounds, axis=1), np.stack(upper_bounds, axis=1)
    source.update(q=np.stack(solutions, axis=1), errors=np.stack(errors, axis=1), solver='trust-constr',
                  model_path=str(args.model.resolve()), **weights, solver_success=np.array(success),
                  joint_limits_enabled=limited,
                  recalculate_bounds=args.recalculate_bounds,
                  joint_lower_bounds_rad=lower_bounds, joint_upper_bounds_rad=upper_bounds)
    np.savez_compressed(args.output / 'trajectory.npz', **source)
    print(f'Saved to {args.output / "trajectory.npz"}', flush=True)


if __name__ == '__main__':
    main()
