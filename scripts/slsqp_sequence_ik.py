import argparse
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from scipy_ik import pose_error, pose_jacobian


class SequenceIK:
    def __init__(self, model, side, targets, initial, reference, epsilon_caps=(20., 10.),
                 epsilon_scale=1., reference_weight=0.1, first_order_weight=1., second_order_weight=1.):
        self.model, self.data = model, mujoco.MjData(model)
        self.targets, self.initial, self.reference = targets, initial, reference
        self.joints = [model.joint(f'{side}_joint{i}').id for i in range(1, 7)]
        self.site = model.site(f'{side}_tcp_site').id
        self.reference_weight = reference_weight
        self.first_order_weight = first_order_weight
        self.second_order_weight = second_order_weight
        self.last_x = None
        self.residual = np.empty((len(initial), 6))
        self.pose_jacobian = np.empty((len(initial), 6, 6))
        initial_errors = self.errors(initial.ravel())
        self.epsilon_mm_deg = epsilon_scale * np.clip(initial_errors, .1, epsilon_caps)
        self.epsilon = self.epsilon_mm_deg / np.array([1000., 180. / np.pi])

    def objective(self, x):
        q = x.reshape(-1, 6)
        value = .5 * (self.reference_weight * np.sum((q - self.reference) ** 2)
                       + self.first_order_weight * np.sum(np.diff(q, axis=0) ** 2))
        value += .5 * self.second_order_weight * np.sum(np.diff(q, n=2, axis=0) ** 2)
        return value

    def gradient(self, x):
        q = x.reshape(-1, 6)
        gradient = self.reference_weight * (q - self.reference)
        velocity = self.first_order_weight * np.diff(q, axis=0)
        gradient[:-1, :] -= velocity
        gradient[1:, :] += velocity
        acceleration = self.second_order_weight * np.diff(q, n=2, axis=0)
        gradient[:-2, :] += acceleration
        gradient[1:-1, :] -= 2 * acceleration
        gradient[2:, :] += acceleration
        return gradient.ravel()

    def evaluate(self, x):
        if self.last_x is not None and np.array_equal(x, self.last_x):
            return
        for frame, (q, target) in enumerate(zip(x.reshape(-1, 6), self.targets)):
            args = (q, self.model, self.data, self.joints, self.site, target)
            self.residual[frame, :] = pose_error(*args)
            self.pose_jacobian[frame, :, :] = pose_jacobian(*args)
        self.last_x = x.copy()

    def errors(self, x):
        self.evaluate(x)
        return np.linalg.norm(self.residual.reshape(-1, 2, 3), axis=2) * np.array([1000., 180. / np.pi])

    def constraints(self, x):
        self.evaluate(x)
        residual = self.residual.reshape(-1, 2, 3)
        return (1 - np.sum(residual ** 2, axis=2) / self.epsilon ** 2).ravel()

    def constraint_jacobian(self, x):
        self.evaluate(x)
        residual = self.residual.reshape(-1, 2, 3)
        pose_jacobian = self.pose_jacobian.reshape(-1, 2, 3, 6)
        blocks = -2 * np.einsum('nti,ntij->ntj', residual, pose_jacobian) / self.epsilon[:, :, None] ** 2
        jacobian = np.zeros((2 * len(self.initial), x.size))
        for frame, block in enumerate(blocks):
            jacobian[2 * frame:2 * frame + 2, 6 * frame:6 * frame + 6] = block
        return jacobian

    def solve(self, maxiter=1000):
        return minimize(self.objective, self.initial.ravel().copy(), jac=self.gradient,
                        method='SLSQP', constraints=dict(type='ineq', fun=self.constraints, jac=self.constraint_jacobian),
                        options=dict(ftol=1e-8, maxiter=maxiter))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', type=Path, default=Path(__file__).resolve().parents[1] / 'abc_sim/models/yam_dishrack_base.xml')
    parser.add_argument('--position-epsilon-mm', type=float, default=20.)
    parser.add_argument('--rotation-epsilon-deg', type=float, default=10.)
    parser.add_argument('--epsilon-scale', type=float, default=1.)
    parser.add_argument('--reference-weight', type=float, default=0.1)
    parser.add_argument('--first-order-weight', type=float, default=1.)
    parser.add_argument('--second-order-weight', type=float, default=1.)
    parser.add_argument('--maxiter', type=int, default=1000)
    args = parser.parse_args()

    with np.load(args.input) as saved:
        source = {key: saved[key] for key in saved.files}
    model = mujoco.MjModel.from_xml_path(str(args.model))
    if 'base_poses_world' in source:
        for side, transform in zip(('left', 'right'), source['base_poses_world']):
            body = model.body(f'{side}_arm')
            body.pos[:] = transform[:3, 3]
            body.quat[:] = Rotation.from_matrix(transform[:3, :3]).as_quat()[[3, 0, 1, 2]]

    reference = source.get('reference', source['initial'])
    solutions, success = [], []
    epsilon_caps = (args.position_epsilon_mm, args.rotation_epsilon_deg)
    for arm, side in enumerate(('left', 'right')):
        problem = SequenceIK(model, side, source['targets_world'][:, arm, :, :], source['initial'][:, arm, :],
                             reference[:, arm, :], epsilon_caps, args.epsilon_scale, args.reference_weight,
                             args.first_order_weight, args.second_order_weight)
        result = problem.solve(args.maxiter)
        solutions.append(result.x.reshape(-1, 6))
        success.append(result.success)
        if not result.success:
            print(f'{side}: {result.message}', flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / 'trajectory.npz', q=np.stack(solutions, axis=1), optimizer_success=np.array(success))
    print(args.output / 'trajectory.npz', flush=True)


if __name__ == '__main__':
    main()
