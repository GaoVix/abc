"""Record the selected old/new case with CPU MuJoCo and FP32 eager inference.

Loads the saved object placement directly and preserves the original batched
world's diffusion-noise stream. The production rollout loop is unchanged.
Videos are rendered from saved trajectories after policy evaluation finishes.
"""
import argparse
import copy
import gc
import hashlib
import json
import time
from dataclasses import fields
from datetime import datetime, timedelta, timezone
from itertools import zip_longest
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from abc_minimal.config import ClipConfig, DiTConfig, SimEvalConfig
from abc_minimal.eval_policy import (
    _make_env, build_summary, resolve_prefix_length, resolve_prompt, resolved_physics,
    rollout_worlds, video_frame,
)
from abc_minimal.policy import DiTInferencePolicy
from abc_sim.randomization.arm_bases import set_median_arm_bases
from abc_sim.randomization.core import RandomizationState
from scripts.find_eval_failure_case import scene


def digest(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


class RecordedScene:
    """Restore one historical placement and record CPU states at each action."""
    def __init__(self, env, record, base, collision_mode, render_samples=None):
        self.env, self.record = env, record
        self.base, self.collision_mode = base, collision_mode
        self.render_samples = render_samples
        self.qpos, self.qvel, self.times, self.actions = [], [], [], []

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, seed, options=None):
        native = self.env.env
        native.forget_arm_state()
        native.reset(seed=seed, randomize=False)
        camera = native.data.camera('top')
        top_before = np.r_[camera.xpos.copy(), camera.xmat.copy()]
        state = RandomizationState.from_dict(copy.deepcopy(self.record['randomization']))
        historical_bases = state.metadata.pop('arm_base_poses', {})
        state.metadata['arm_collision_mode'] = self.collision_mode
        state.metadata['arm_base_source'] = self.base
        native._task_randomizer.apply(native.model, native.data, state)
        if self.base == 'median':
            bases = set_median_arm_bases(native.model, native.data)
            for name, pose in historical_bases.items():
                actual = bases[name]
                if not np.allclose(actual['pos'], pose['pos'], atol=1e-8, rtol=0):
                    raise ValueError('Current median base differs from the historical base position')
                a, b = np.asarray(actual['quat']), np.asarray(pose['quat'])
                if min(np.linalg.norm(a-b), np.linalg.norm(a+b)) > 1e-7:
                    raise ValueError('Current median base differs from the historical base rotation')
            state.metadata['arm_base_poses'] = bases
        native.data.qvel[:] = 0
        native.data.ctrl[:] = 0
        native.data.qacc_warmstart[:] = 0
        native.data.time = 0
        mujoco.mj_forward(native.model, native.data)
        camera = native.data.camera('top')
        if not np.allclose(top_before, np.r_[camera.xpos, camera.xmat], atol=1e-12, rtol=0):
            raise ValueError('Top camera moved while restoring the scene')
        for name, pose in state.object_states.items():
            joint = mujoco.mj_name2id(native.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint >= 0:
                adr = int(native.model.jnt_qposadr[joint])
                actual = native.data.qpos[adr:adr+7]
            else:
                body = native.model.body(name)
                actual = np.r_[body.pos, body.quat]
            if not np.allclose(actual, np.r_[pose['pos'], pose['quat']], atol=1e-12, rtol=0):
                raise ValueError(f'Historical object placement was not restored: {name}')
        native.cur_step = 0
        native._last_randomization = state
        native._task_evaluator.reset(nworld=1)
        native._task_evaluator.configure_from_randomization(native.model, state)
        if native._camera_backend == 'mujoco' and self.render_samples is not None:
            # Set framebuffer sampling before creating the renderer. The local
            # NVIDIA/EGL tests found 1-level RGB differences with 16x MSAA.
            native.renderer.close()
            native.model.vis.quality.offsamples = self.render_samples
            native.renderer = mujoco.Renderer(native.model, height=self.env.height, width=self.env.width)
        native._reset_camera_backend()
        self.env.randomization = state
        self.initial_arrays = {name: np.array(getattr(native.model, name)) for name in
                               ('body_pos', 'body_quat', 'cam_pos', 'cam_quat', 'geom_contype', 'geom_conaffinity')}
        self.initial_arrays.update(qpos=native.data.qpos.copy(), qvel=native.data.qvel.copy(),
                                   ctrl=native.data.ctrl.copy(), top_camera_world=top_before)
        self.snapshot()
        return self.env.obs()

    def snapshot(self):
        data = self.env.env.data
        self.qpos.append(data.qpos.copy())
        self.qvel.append(data.qvel.copy())
        self.times.append(float(data.time))

    def step_one(self, action):
        self.env.step_one(action)
        data = self.env.env.data
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            raise ValueError('Nonfinite CPU physics state')
        if data.time <= self.times[-1]:
            raise ValueError('CPU physics reset its integration time; possible divergence')
        for warning in ('mjWARN_BADQPOS', 'mjWARN_BADQVEL', 'mjWARN_BADQACC'):
            if data.warning[getattr(mujoco.mjtWarning, warning)].number:
                raise ValueError(f'CPU physics reported {warning}')
        self.actions.append(np.array(action))
        self.snapshot()

    def close(self):
        # rollout_worlds closes here; retain its renderer for offline replay.
        pass


class WorldNoise:
    """Use the same [policy_seed, original_world_index] RNG as batched eval."""
    def __init__(self, policy, policy_seed, world_index):
        self.policy = policy
        self.rng = np.random.default_rng([policy_seed, world_index])
        self.noise, self.predictions = [], []
        self.observations = []

    def __getattr__(self, name):
        return getattr(self.policy, name)

    def infer(self, obs, noise=None, **kwargs):
        sample = self.rng.standard_normal((self.chunk_length, self.action_dim), dtype=np.float32)
        result = self.policy.infer(obs, noise=sample, **kwargs)
        self.noise.append(sample)
        self.predictions.append(result.copy())
        self.observations.append(dict(state=digest(obs['state']), images={
            name: digest(value) for name, value in obs['images'].items()}))
        return result


def render_trajectory(env, folder, stride, camera_keys):
    path = folder/'rollout.mp4'
    dt = env.env.env._physics_dt * env.env.env._control_decimation
    native = env.env.env
    frames = list(range(0, len(env.qpos), stride))
    if frames[-1] != len(env.qpos)-1:
        frames.append(len(env.qpos)-1)
    with imageio.get_writer(str(path), fps=1/(dt*stride), macro_block_size=1) as writer:
        for index in frames:
            native.data.qpos[:] = env.qpos[index]
            native.data.qvel[:] = env.qvel[index]
            native.data.time = env.times[index]
            native.cur_step = index
            mujoco.mj_forward(native.model, native.data)
            writer.append_data(video_frame(env.env.render_cameras(), camera_keys))
    return path, 1/(dt*stride)


def run_one(side, report, args, root):
    selected = report['selected'][side]
    saved = report['selected_configs'][side]
    record = report['selected_world_records'][side]
    if saved.get('parallel_worlds', 0) <= 0:
        raise ValueError('This replay expects a candidate selected from batched evaluations')
    folder = root/side
    folder.mkdir()
    model_fields = {item.name for item in fields(DiTConfig)}
    model = DiTConfig(**{key: value for key, value in saved.get('model', {}).items() if key in model_fields})
    base = 'xml' if side == 'old' else 'median'
    original_mode = selected['arm_collision_mode']
    collision_mode = original_mode if original_mode in ('full', 'eef_only') else 'full'
    if args.collision_mode != 'historical':
        collision_mode = args.collision_mode
    config = SimEvalConfig(
        checkpoint=selected['checkpoint'], policy='dit', task='dishrack', model=model,
        clip=ClipConfig(**saved.get('clip', {})),
        num_worlds=1, parallel_worlds=0, seed=report['selected']['world_seed'],
        policy_seed=saved.get('policy_seed', 0), gpu_id=args.gpu_id,
        diffusion_steps=saved.get('diffusion_steps', 10),
        execute_chunk_dim=saved.get('execute_chunk_dim', 15),
        num_chunks=args.chunks or saved.get('num_chunks', 236),
        rtc=saved.get('rtc', True), rtc_prefix_length=saved.get('rtc_prefix_length', 4),
        rtc_inference_lead_steps=saved.get('rtc_inference_lead_steps', 4),
        prefix_length=saved.get('prefix_length') or 0,
        camera_height=saved.get('camera_height', 168), camera_width=saved.get('camera_width', 224),
        camera_backend=args.camera_backend, fast_inference=False, compensate_arm_bases=False,
        norm_stats_path=saved.get('norm_stats_path'), prompt=saved.get('prompt'),
        randomization=json.dumps(dict(arm_base_source=base, arm_collision_mode=collision_mode,
                                      randomize_arm_bases=False)),
        log_every_chunk=True, save_video=False, output_dir=str(folder))
    config.prompt = resolve_prompt(config)
    start = time.perf_counter()
    print(f'{side}: {config.checkpoint}; base={base}; collisions={collision_mode}; '
          f'CPU MuJoCo; renderer={args.camera_backend}; FP32 eager', flush=True)
    policy = DiTInferencePolicy(Path(config.checkpoint), config, f'cuda:{args.gpu_id}')
    noise_policy = WorldNoise(policy, config.policy_seed, selected['world_index'])
    prefix = resolve_prefix_length(config, policy.trained_max_prefix, config.model)
    config.prefix_length = prefix
    render_samples = args.mujoco_render_samples if args.camera_backend == 'mujoco' else None
    env = RecordedScene(_make_env(config, tuple(model.camera_keys)), record, base, collision_mode, render_samples)
    try:
        worlds = rollout_worlds(config, noise_policy, env, prefix, None, folder, config.model)
        np.savez_compressed(folder/'trajectory.npz', qpos=np.array(env.qpos), qvel=np.array(env.qvel),
                            time=np.array(env.times), actions=np.array(env.actions),
                            diffusion_noise=np.array(noise_policy.noise), predictions=np.array(noise_policy.predictions))
        np.savez(folder/'initial.npz', **env.initial_arrays)
        (folder/'observation_hashes.json').write_text(json.dumps(noise_policy.observations, indent=2))
        video, fps = render_trajectory(env, folder, args.render_stride, tuple(model.camera_keys))
        worlds[0]['video_path'] = str(video)
        worlds[0]['source_world_index'] = selected['world_index']
        build_summary(config=config, ckpt_path=Path(config.checkpoint), device=f'cuda:{args.gpu_id}',
                      worlds=worlds, out_dir=folder, physics=resolved_physics(env.env))
        return dict(success=worlds[0]['success'], max_reward=worlds[0]['max_reward'], steps=worlds[0]['steps'],
                    historical_success=selected['success'], base=base, collision_mode=collision_mode,
                    camera_backend=args.camera_backend, mujoco_render_samples=render_samples,
                    video=str(video), fps=fps, seconds=time.perf_counter()-start,
                    summary=str(folder/'summary.json'), initial_hash=digest(env.initial_arrays['qpos']))
    finally:
        env.env.close()


def comparison_video(results, root, width, top_index):
    readers = {side: imageio.get_reader(result['video']) for side, result in results.items()}
    try:
        font = ImageFont.truetype('DejaVuSans.ttf', 17)
    except OSError:
        font = ImageFont.load_default()
    last = {}
    path = root/'comparison_top.mp4'
    with imageio.get_writer(str(path), fps=results['old']['fps'], macro_block_size=1) as writer:
        for pair in zip_longest(readers['old'], readers['new']):
            canvas = Image.new('RGB', (896, 400), '#181b20')
            draw = ImageDraw.Draw(canvas)
            for column, side in enumerate(('old', 'new')):
                if pair[column] is not None:
                    last[side] = pair[column]
                frame = last[side][:, top_index[side]*width[side]:(top_index[side]+1)*width[side]]
                canvas.paste(Image.fromarray(frame).resize((448, 336)), (448*column, 64))
                outcome = 'SUCCESS' if results[side]['success'] else 'FAIL'
                label = f"{side.upper()} | {results[side]['base']} base | {outcome}"
                draw.text((10+448*column, 8), label, font=font, fill='#80dd9d' if outcome=='SUCCESS' else '#ff9d89')
                draw.text((10+448*column, 35), f"Collisions: {results[side]['collision_mode']}",
                          font=font, fill='#d2d6dd')
            writer.append_data(np.array(canvas))
    for reader in readers.values():
        reader.close()
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--chunks', type=int)
    parser.add_argument('--render-stride', type=int, default=2)
    parser.add_argument('--camera-backend', choices=('mjwarp', 'mujoco'), default='mjwarp')
    parser.add_argument('--mujoco-render-samples', type=int, default=0,
                        help='Native MuJoCo framebuffer samples; 0 disables MSAA for repeatability tests.')
    parser.add_argument('--collision-mode', choices=('historical', 'full', 'eef_only'), default='historical')
    args = parser.parse_args()
    if args.render_stride < 1 or (args.chunks is not None and args.chunks < 1):
        parser.error('render-stride and chunks must be positive')
    if args.mujoco_render_samples < 0:
        parser.error('mujoco-render-samples must be nonnegative')
    report = json.loads(args.selection.read_text())
    if report.get('status') != 'found':
        raise ValueError('Selection does not contain a matched case')
    for side in ('old', 'new'):
        if not Path(report['selected'][side]['checkpoint']).is_file():
            raise FileNotFoundError(report['selected'][side]['checkpoint'])
    old, new = (report['selected_world_records'][side] for side in ('old', 'new'))
    if scene(old) is None or scene(old) != scene(new):
        raise ValueError('Selected initial object poses, scales or variants differ')
    seed, index = report['selected']['world_seed'], report['selected']['old']['world_index']
    if index != report['selected']['new']['world_index']:
        raise ValueError('Original world indices differ; the policy noise streams would differ')
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')
    root = args.output or args.selection.parent/f'world_{index:03d}_cpu_fp32_{stamp}'
    root.mkdir(parents=True, exist_ok=True)
    if any((root/name).exists() for name in ('old', 'new', 'summary.json')):
        raise FileExistsError(f'Refusing to overwrite a previous diagnostic: {root}')
    (root/'selection.json').write_text(json.dumps(report, indent=2))
    torch.cuda.set_device(args.gpu_id)
    results = {}
    for side in ('old', 'new'):
        results[side] = run_one(side, report, args, root)
        gc.collect()
        torch.cuda.empty_cache()
        (root/'progress.json').write_text(json.dumps(results, indent=2))
    widths = {side: report['selected_configs'][side].get('camera_width', 224) for side in results}
    top = {side: report['selected_configs'][side].get('model', {}).get('camera_keys', ['top','left','right']).index('top') for side in results}
    result = dict(world_seed=seed, source_world_index=index, physics='CPU MuJoCo', fast_inference=False,
                  camera_backend=args.camera_backend,
                  smoke_test_only=report.get('smoke_test_only', False),
                  noise_stream=[report['selected_configs']['old'].get('policy_seed', 0), index],
                  historical_outcomes_reproduced=all(row['success']==row['historical_success'] for row in results.values()),
                  collision_mode=args.collision_mode, results=results,
                  comparison_video=str(comparison_video(results, root, widths, top)),
                  note='Controlled new diagnostic; actual outcomes are labeled. Historical GPU/compiled trajectories were not saved.',
                  source_sha256={path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in
                                 ('scripts/replay_eval_failure_case.py', 'scripts/find_eval_failure_case.py',
                                  'abc_minimal/policy.py','abc_minimal/eval_policy.py','abc_minimal/sim_env.py',
                                  'abc_sim/env.py','abc_sim/randomization/arm_bases.py')})
    (root/'summary.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
