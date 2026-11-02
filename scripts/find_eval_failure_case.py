"""Find a recorded XML-retargeted success paired with a median-model failure.

Standard library only. Reads completed eval summaries; never launches evaluation.
"""
import argparse
import json
from pathlib import Path

OLD = 'dit_joint_retargeted_official_timing_20261022_125331'
NEW = 'dit_median_sequence_ik_20261101_183529'
SETTINGS = dict(policy_seed=0, num_chunks=236, diffusion_steps=10,
                execute_chunk_dim=15, rtc=True, rtc_prefix_length=4,
                rtc_inference_lead_steps=4, camera_height=168, camera_width=224,
                fast_inference=True, parallel_worlds=0, camera_backend='mjwarp')


def rounded(value):
    if isinstance(value, dict):
        return {key: rounded(item) for key, item in value.items()}
    if isinstance(value, list):
        return [rounded(item) for item in value]
    return round(value, 9) if isinstance(value, float) else value


def scene(world):
    state = world.get('randomization') or {}
    objects = state.get('object_states')
    if not objects:
        return None
    meta = state.get('metadata', {})
    count = meta.get('plate_count')
    variants = meta.get('plate_variants')
    if variants is None and count is not None and meta.get('plate_variant'):
        variants = [meta['plate_variant']] * count
    return rounded(dict(objects=objects, scales=state.get('scale_states', {}),
                        count=count, plates=variants, rack=meta.get('dish_rack_variant')))


def read_run(path):
    try:
        data = json.loads(path.read_text())
        config = data.get('config', {})
        checkpoint = Path(data.get('checkpoint') or config.get('checkpoint', ''))
        model = checkpoint.parent.name
        if model not in (OLD, NEW) or not checkpoint.stem.isdigit():
            return None
        worlds = data.get('worlds', [])
        if not worlds or len(worlds) != config.get('num_worlds', len(worlds)):
            return None
        request = config.get('randomization') or {}
        request = json.loads(request) if isinstance(request, str) else request
        if config.get('compensate_arm_bases', False) or request.get('randomize_arm_bases', False):
            return None
        sources = {
            (w.get('randomization') or {}).get('metadata', {}).get(
                'arm_base_source', request.get('arm_base_source', 'xml')) for w in worlds
        }
        required = 'xml' if model == OLD else 'median'
        if sources != {required}:
            return None
        # Keep horizon, inference, batching and noise settings aligned.
        protocol = {key: config.get(key, value) for key, value in SETTINGS.items()}
        protocol['prefix_length'] = config.get('prefix_length') or 0
        return dict(path=str(path), checkpoint=str(checkpoint), step=int(checkpoint.stem),
                    model=model, config=config, request=request, protocol=protocol,
                    worlds={int(w['world_seed']): w for w in worlds},
                    success=sum(bool(w.get('success')) for w in worlds))
    except (OSError, ValueError, TypeError, KeyError):
        return None


def video(run, world):
    path = world.get('video_path')
    if path and Path(path).is_file():
        return path
    fallback = Path(run['path']).parent / f"world_{world['world_index']:03d}.mp4"
    return str(fallback) if fallback.is_file() else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/mnt/afs/L202500703/abc_storage/output'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--allow-step-mismatch', action='store_true')
    args = parser.parse_args()
    if not args.root.is_dir():
        raise SystemExit(f'Eval results directory not found: {args.root}')
    paths = set(args.root.glob('eval*/summary.json')) | set(args.root.glob('eval*/*/summary.json'))
    runs = [run for path in sorted(paths) if (run := read_run(path)) is not None]
    inventory = [dict(summary=r['path'], checkpoint=r['checkpoint'], step=r['step'],
                      base='xml' if r['model'] == OLD else 'median',
                      successes=r['success'], worlds=len(r['worlds'])) for r in runs]
    candidates = []
    different_scenes = 0
    for old in (r for r in runs if r['model'] == OLD):
        for new in (r for r in runs if r['model'] == NEW):
            if old['protocol'] != new['protocol']:
                continue
            if old['step'] != new['step'] and not args.allow_step_mismatch:
                continue
            for seed in old['worlds'].keys() & new['worlds'].keys():
                a, b = old['worlds'][seed], new['worlds'][seed]
                if not a.get('success') or b.get('success') or a.get('diverged') or b.get('diverged'):
                    continue
                initial = scene(a)
                if initial is None or initial != scene(b):
                    different_scenes += 1
                    continue
                candidates.append(dict(world_seed=seed, old=old, new=new,
                                       old_world=a, new_world=b))
    candidates.sort(key=lambda x: (x['old']['step'] != x['new']['step'], -min(x['old']['step'], x['new']['step']),
                                  x['new_world'].get('max_reward', 1), x['old_world'].get('steps', 999999),
                                  x['world_seed'], x['old']['path'], x['new']['path']))
    result = dict(status='found' if candidates else 'no_matching_pair', eligible_runs=inventory,
                  matching_pairs=len(candidates), rejected_scene_mismatches=different_scenes,
                  criteria='XML-retargeted success vs fixed-camera median-model failure; no compensation, no random bases, no physics divergence; same actual world seed, initial object poses and rollout protocol; same checkpoint step unless explicitly allowed.')
    if candidates:
        choice = candidates[0]
        result['selected'] = dict(world_seed=choice['world_seed'], initial_objects_verified_equal=True)
        for side in ('old', 'new'):
            run, world = choice[side], choice[f'{side}_world']
            result['selected'][side] = dict(summary=run['path'], checkpoint=run['checkpoint'],
                checkpoint_step=run['step'], world_index=world['world_index'], success=world['success'],
                max_reward=world.get('max_reward'), final_reward=world.get('reward'), steps=world.get('steps'),
                arm_collision_mode=run['request'].get('arm_collision_mode', 'original XML'),
                video=video(run, world))
        # Preserve original metadata and scoring curves for subsequent diagnosis.
        result['selected_world_records'] = {side: choice[f'{side}_world'] for side in ('old', 'new')}
        result['selected_configs'] = {side: choice[side]['config'] for side in ('old', 'new')}
        if any(result['selected'][side]['video'] is None for side in ('old', 'new')):
            result['video_note'] = 'A saved historical video is missing. The original eval does not save qpos/actions by default; an exact historical video cannot be reconstructed from summary.json alone. A diagnostic rerun must verify its actual outcome before labeling success/failure.'
    output = args.output or args.root/'retargeted_vs_median_case_20261103'/'selection.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ('selected_world_records', 'selected_configs')}, ensure_ascii=False, indent=2))
    print(f'Report: {output}')


if __name__ == '__main__':
    main()
