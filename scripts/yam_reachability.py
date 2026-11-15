"""Enumerate YAM's analytic IK branches for reachability classification only."""

import numpy as np


STATUS = ('unknown_singular', 'reachable', 'geometrically_unreachable', 'no_solution_within_expanded_limits')


def ry(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]])


def rz(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def branches(target):
    position, rotation = target[:3, 3], target[:3, :3]
    planar = position - np.array([0., 0., .0631 + .0409]) - .0353 * rotation[:, 2]
    if np.linalg.norm(planar[:2]) < 1e-10:
        return [], 'yaw_singularity'
    angle = np.arctan2(planar[1], planar[0])
    solutions = []
    for q1 in (angle, angle + np.pi):
        local_r = rz(-q1) @ rotation
        axis = local_r[:, 2]
        cosine = np.hypot(axis[0], axis[2])
        if cosine < 1e-10:
            return [], 'wrist_singularity'
        for wrist_sign in (1, -1):
            q5 = np.arctan2(-axis[1], wrist_sign * cosine)
            phi = np.arctan2(-axis[2], axis[0]) + (np.pi if wrist_sign < 0 else 0.)
            roll = (ry(phi) @ rz(-q5) @ ry(np.pi / 2)).T @ local_r
            q6 = np.arctan2(roll[1, 0], roll[0, 0])
            wrist = rz(-q1) @ planar - np.array([.02, 0., 0.]) - .074 * ry(phi)[:, 0]
            l1, l2 = .264, np.hypot(.245, .06)
            beta = np.arctan2(.06, .245)
            elbow_cos = (l1 * l1 + l2 * l2 - np.dot(wrist, wrist)) / (2 * l1 * l2)
            if abs(elbow_cos) > 1 + 1e-10:
                continue
            for sign in (1, -1):
                kappa = sign * np.arccos(np.clip(elbow_cos, -1., 1.))
                q3 = kappa - beta
                q2 = np.arctan2(l2 * np.sin(kappa), -l1 + l2 * np.cos(kappa)) - np.arctan2(wrist[2], wrist[0])
                solutions.append(np.array([q1, q2, q3, q2 - q3 - phi, q5, q6]))
    return solutions, 'regular'


def classify(target, limits):
    roots, status = branches(target)
    if status != 'regular':
        return 0
    if not roots:
        return 2
    low, high = limits.T
    q = np.asarray(roots)
    q += 2 * np.pi * np.round(((low + high) / 2 - q) / (2 * np.pi))
    return 1 if np.any(np.all((q >= low - 1e-10) & (q <= high + 1e-10), axis=1)) else 3
