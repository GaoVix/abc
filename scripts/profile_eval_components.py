"""Time eval components on a short, model-free physical hold trajectory.

Instrumentation is local to this process. Production files/settings are unchanged.
GPU operations synchronize at component boundaries so their time is attributed
to the operation rather than the subsequent CPU readback.
"""
import argparse
from collections import defaultdict
from contextlib import contextmanager
import json
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import torch
import warp as wp

import abc_sim
from abc_minimal.base_compensation import BaseCompensation
from abc_sim.env import _GRIPPER_CTRL_MAX
import abc_sim.task_eval.dishrack as dishrack


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--worlds', type=int, default=100)
parser.add_argument('--chunks', type=int, default=4)
parser.add_argument('--gpu-id', type=int, default=0)
parser.add_argument('--output', type=Path, default=Path('outputs/eval_timing_profile'))
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
torch.cuda.set_device(args.gpu_id)
stats = defaultdict(lambda: dict(seconds=0., calls=0))


@contextmanager
def measured(name, gpu=False):
    start = time.perf_counter()
    try:
        yield
    finally:
        if gpu:
            torch.cuda.synchronize(args.gpu_id)
        stats[name]['seconds'] += time.perf_counter() - start
        stats[name]['calls'] += 1


def instrument(owner, attribute, name, gpu=False):
    original = getattr(owner, attribute)

    def timed(*a, **kw):
        with measured(name, gpu):
            return original(*a, **kw)

    setattr(owner, attribute, timed)


class EvaluatorMuJoCo:
    """Time MuJoCo calls only inside the CPU dishrack evaluator."""
    def __init__(self, original):
        self.original = original

    def __getattr__(self, name):
        return getattr(self.original, name)

    def MjData(self, *a, **kw):
        with measured('evaluator.data_allocation'):
            return self.original.MjData(*a, **kw)

    def mj_forward(self, *a, **kw):
        with measured('evaluator.cpu_forward'):
            return self.original.mj_forward(*a, **kw)


request = dict(plate_count=3, plate_variant='plate_0', dish_rack_variant='dish_rack_0',
               randomize_variants=False, randomize_scales=False,
               randomize_arm_bases=True, arm_base_source='training')
setup_start = time.perf_counter()
env = abc_sim.make_batched_env('dishrack', num_worlds=args.worlds,
                              camera_height=168, camera_width=224,
                              camera_gpu_id=args.gpu_id)
try:
    obs = env.reset([20260511 + i for i in range(args.worlds)], {'randomization': request})
    conversion = BaseCompensation(ik_backend='mink', reference='median')
    conversion.reset(env.randomization, obs['state'])
    virtual_hold = conversion.convert(obs['state'], state=True)
    # Compile and warm the same 17-substep graph before timed measurements.
    for _ in range(3):
        env.step(obs['state'])
    env.get_obs()
    torch.cuda.synchronize(args.gpu_id)
    setup_seconds = time.perf_counter() - setup_start
    print(f'Warmup complete: {setup_seconds:.2f}s', flush=True)

    dishrack.mujoco = EvaluatorMuJoCo(dishrack.mujoco)
    instrument(dishrack.DishrackEvaluator, '_evaluate_one', 'evaluator.score')
    instrument(dishrack.DishrackEvaluator, '_rack_half_extents', 'evaluator.rack_bounds')
    instrument(dishrack.DishrackEvaluator, '_contact_flags_for_plate', 'evaluator.plate_contacts')
    instrument(dishrack, '_body_subtree_ids', 'evaluator.body_subtrees')
    instrument(dishrack, '_geom_ids_for_bodies', 'evaluator.geom_sets')
    instrument(conversion, '_fk', 'conversion.source_fk')
    instrument(conversion, '_solve', 'conversion.arm_ik')
    for controller in conversion.controllers:
        instrument(controller, 'solve', 'conversion.controller_solve')
    instrument(conversion.controllers[0].mink, 'solve_ik', 'conversion.inner_qp')
    instrument(env._runtime._mjw, 'refit_bvh', 'render.bvh', gpu=True)
    instrument(env._runtime._mjw, 'render', 'render.rays', gpu=True)

    contacts = []
    wall_start = time.perf_counter()
    for chunk in range(args.chunks):
        for step in range(15):
            if step == 11:
                with measured('observation.physical_state'):
                    state = env.state()
                with measured('observation.render_total', gpu=True):
                    env.render_cameras()
                with measured('observation.state_conversion'):
                    conversion.convert(state, state=True)
            with measured('action.physical_state_read'):
                physical = env.state()
            with measured('action.conversion'):
                action = conversion.convert(virtual_hold, physical_state=physical)
            with measured('physics.control_preparation'):
                action = np.asarray(action, dtype=np.float32).copy()
                action[:, env.base_env._gripper_indices] *= _GRIPPER_CTRL_MAX
                ctrl = np.zeros((args.worlds, env.model.nu), dtype=np.float32)
                ctrl[:, env.base_env._ctrl_indices] = action
            with measured('physics.control_upload', gpu=True):
                env._runtime.set_ctrl_batch(ctrl)
            with measured('physics.gpu_step', gpu=True):
                env._runtime.step(nstep=env.control_decimation)
            with measured('physics.state_download'):
                env.qpos = env._runtime.d_warp.qpos.numpy()
                if int(env._runtime.d_warp.nacon.numpy()[0]) >= env._runtime.d_warp.naconmax:
                    raise RuntimeError('Contact buffer overflow')
            with measured('evaluator.total'):
                env.evaluate()
            if not np.isfinite(env.qpos).all():
                raise RuntimeError('Non-finite physical state during hold benchmark')
        contacts.append(float(env._runtime.d_warp.nacon.numpy()[0]))
        print(f'Profile chunk {chunk + 1}/{args.chunks}: elapsed={time.perf_counter()-wall_start:.2f}s', flush=True)

    elapsed = time.perf_counter() - wall_start
    result = dict(status='complete', worlds=args.worlds, chunks=args.chunks,
                  control_steps=args.chunks * 15, warmup_seconds=setup_seconds,
                  measured_wall_seconds=elapsed, stats=dict(stats),
                  hardware=dict(gpu=torch.cuda.get_device_name(args.gpu_id),
                                cpu=platform.processor(), platform=platform.platform()),
                  cameras=dict(rendered=[env.model.cam(i).name for i in range(env.model.ncam)],
                               returned=env.camera_names), contact_totals=contacts,
                  git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                  scope='Local short hold, no checkpoint or policy inference. Timers are nested; do not sum parent and child entries. Not a reproduction of remote policy trajectories. Explicit GPU synchronization changes overlap.')
    for entry in result['stats'].values():
        entry['milliseconds_per_call'] = entry['seconds'] * 1000 / entry['calls']
        entry['seconds_per_15step_chunk'] = entry['seconds'] / args.chunks
    (args.output / 'profile.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)
finally:
    env.close()
