"""Render XML-base joint replays with calibrated original-data reference arms.

`--export-only` runs where the fine-tuned checkpoints/caches are available. Its
small portable bundle can be copied to a machine with the original MCAP, then
rendered with `--bundle`. No inference, training or data conversion is changed.
"""
import os

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(variable, '1')

import argparse
import copy
from dataclasses import asdict
import gzip
import json
from pathlib import Path
import subprocess
import zipfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHUNK = 'episode_298becaa-b397-4766-9c80-6fb74e182f4b:1336'
CELLS = ('old_on_old', 'new_on_old', 'old_on_new', 'new_on_new')
JOINT_COLUMNS = [*range(6), *range(7, 13)]
SEVEN_CASES = (
    ('compare_chunk_loss_20261105_125057', 'episode_298becaa-b397-4766-9c80-6fb74e182f4b', 1336),
    ('compare_chunk_loss_20261105_125057', 'episode_5d3515a9-3f0d-4275-a5bd-1daa9db9c78a', 1076),
    ('compare_real_10000_seed124_20261105_140707', 'episode_18c1a663-4a50-4de8-838f-e1e979ba9bfb', 1084),
    ('compare_real_10000_seed124_20261105_140707', 'episode_f6bc340d-405f-42c4-9a10-8b8e894ae11d', 555),
    ('compare_real_10000_seed124_20261105_140707', 'episode_7ce30833-d859-41ce-bd54-710e96f66728', 1213),
    ('compare_real_10000_seed124_20261105_140707', 'episode_6a9bc75b-ad04-443f-b7fa-9fdb67d099cb', 378),
    ('compare_real_10000_seed124_20261105_140707', 'episode_3bb77850-b887-48de-b79d-996a1381cc09', 3366),
)


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def verified_completed_bundle(job):
    bundle, info = job.output/'bundle.npz', job.output/'bundle.json'
    if not bundle.exists() and not info.exists():
        return None
    if not bundle.is_file() or not info.is_file():
        raise ValueError(f'Incomplete bundle at {job.output}; use a new output directory')
    saved = json.loads(info.read_text())
    source = json.loads((job.loss_run/'input.json').read_text())
    locations = json.loads((job.loss_run/'sampled_chunks.json').read_text())
    episode, first = job.chunk.rsplit(':', 1)
    selected = [r for r in locations if r['source_episode'] == episode and r['source_frame'] == int(first)]
    if len(selected) != 1 or saved['chunk'] != selected[0]:
        raise ValueError(f'Completed chunk source mapping differs: {job.chunk}')
    if (saved['seed'] != source['seed'] or saved['generation_steps'] != job.diffusion_steps
            or not saved.get('exact_saved_labels')):
        raise ValueError(f'Completed chunk sampling settings differ: {job.chunk}')
    for side in ('old', 'new'):
        if Path(saved['checkpoints'][side]).resolve() != Path(source[f'{side}_checkpoint']).resolve():
            raise ValueError(f'Completed chunk checkpoint differs: {job.chunk}')
    row = selected[0]
    with np.load(bundle) as arrays:
        for key in ('old_state', 'new_state', 'old_action', 'new_action', *[f'{c}_prediction' for c in CELLS]):
            if arrays[key].shape != (row['chunk_length'], 14) or not np.isfinite(arrays[key]).all():
                raise ValueError(f'Invalid completed array: {job.chunk} / {key}')
        rng = np.random.default_rng([source['seed'], 0, row['sample_id'], 0])
        expected_noise = rng.standard_normal((row['chunk_length'], 14), dtype=np.float32)[None]
        np.testing.assert_array_equal(arrays['noise'], expected_noise)
    return bundle


def seven_cases(args):
    """Preserve each selected chunk's original loss-run/RNG identity."""
    output = args.output
    if (output/'manifest.json').exists() and not args.resume:
        raise ValueError('Use a new output directory')
    jobs = []
    for number, (run, episode, first) in enumerate(SEVEN_CASES, start=1):
        name = f'{number:02d}_{episode}_frames_{first}_{first+29}'
        job = copy.copy(args)
        job.output = output/name
        job.loss_run = args.loss_root/run
        job.chunk = f'{episode}:{first}'
        job.bundle = args.bundle_root/name/'bundle.npz' if args.bundle_root else None
        required = ([job.bundle, job.bundle.with_suffix('.json')] if job.bundle else
                    [job.loss_run/file for file in ('input.json', 'sampled_chunks.json', 'screen_draws.npz')])
        for path in required:
            if not path.is_file():
                raise FileNotFoundError(path)
        job.completed_bundle = verified_completed_bundle(job) if args.resume else None
        jobs.append(job)
    if not args.export_only:
        records = {r['episode']: r for r in json.load(gzip.open(args.calibrations, 'rt'))['records']}
        for _, episode, _ in SEVEN_CASES:
            path = args.raw_root/records[episode]['source_relative']
            if not path.is_file():
                raise FileNotFoundError(f'Original MCAP required for native-resolution images: {path}')
    output.mkdir(parents=True, exist_ok=True)
    manifest = dict(status='running', selected=7, completed=0, cases=[])
    save_json(output/'manifest.json', manifest)
    prediction_session = {} if not args.bundle_root else None
    try:
        for number, job in enumerate(jobs, start=1):
            print(f'CASE {number}/7: {job.chunk}', flush=True)
            try:
                bundle = job.bundle or job.completed_bundle
                if job.completed_bundle:
                    print('Reusing verified completed bundle; no model inference.', flush=True)
                if bundle is None:
                    bundle = export_predictions(job, prediction_session)
                if not args.export_only:
                    render_bundle(job, bundle)
                manifest['cases'].append(dict(chunk=job.chunk, loss_run=str(job.loss_run),
                                              directory=job.output.name,
                                              bundle=(str(bundle) if args.bundle_root else f'{job.output.name}/bundle.npz'),
                                              reused=job.completed_bundle is not None,
                                              video=None if args.export_only else f'{job.output.name}/comparison.mp4'))
                manifest['completed'] = number
                save_json(output/'manifest.json', manifest)
            except Exception as error:
                manifest.update(status='failed', failed_case=job.chunk, error=f'{type(error).__name__}: {error}')
                save_json(output/'manifest.json', manifest)
                raise
    finally:
        if prediction_session:
            prediction_session.clear()
            import torch
            torch.cuda.empty_cache()
    manifest['status'] = 'complete'
    save_json(output/'manifest.json', manifest)
    archive = output/('seven_chunk_bundles.zip' if args.export_only else 'seven_chunk_videos.zip')
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as zipped:
        zipped.write(output/'manifest.json', arcname='manifest.json')
        for job in jobs:
            names = ('bundle.npz', 'bundle.json') if args.export_only else (
                'comparison.mp4', 'comparison_preview.png', 'report.json')
            for name in names:
                zipped.write(job.output/name, arcname=f'{job.output.name}/{name}')
    print(f'COMPLETE 7/7: {archive}', flush=True)


def export_predictions(args, session=None):
    import torch
    from abc_minimal.dataloader import decode_frame, read_state_action_rows
    from scripts.compare_chunk_val_loss import (
        CLIPTextEmbedder, ClipConfig, DiTConfig, DiTPolicy, EpisodeDataset,
        PromptConfig, batch_to_device, checkpoint_metadata, collate,
        configure_determinism, load_operator_label_maps, load_pretrained,
        noise_and_times,
    )

    if (args.output/'bundle.npz').exists():
        raise ValueError('Use a new output directory')
    source = json.loads((args.loss_run/'input.json').read_text())
    episode, first = args.chunk.rsplit(':', 1)
    rows = json.loads((args.loss_run/'sampled_chunks.json').read_text())
    selected = [(i, r) for i, r in enumerate(rows)
                if r['source_episode'] == episode and r['source_frame'] == int(first)]
    if len(selected) != 1:
        raise ValueError(f'Expected one saved chunk: {args.chunk}')
    source_index, row = selected[0]
    # One session spans the entire seven-case run. Cache by actual input paths
    # so different source loss runs can share the same loaded models safely.
    if session is None:
        session = {}
    checkpoint_cache = session.setdefault('checkpoint_metadata', {})
    metadata, checkpoint_keys = {}, {}
    for side in ('old', 'new'):
        checkpoint_keys[side] = str(Path(source[f'{side}_checkpoint']).resolve())
        if checkpoint_keys[side] not in checkpoint_cache:
            checkpoint_cache[checkpoint_keys[side]] = checkpoint_metadata(Path(checkpoint_keys[side]))
        metadata[side] = checkpoint_cache[checkpoint_keys[side]]
    configs = {s: DiTConfig(**m['train']['model']) for s, m in metadata.items()}
    if asdict(configs['old']) != asdict(configs['new']):
        raise ValueError('Model architectures differ')
    if metadata['old']['step'] != metadata['new']['step']:
        raise ValueError('Checkpoint steps differ')
    for field in ('state', 'actions'):
        for statistic in ('mean', 'std'):
            np.testing.assert_array_equal(metadata['old']['stats'][field][statistic],
                                          metadata['new']['stats'][field][statistic])
    configure_determinism(source['seed'])
    if not 0 <= args.gpu_id < torch.cuda.device_count():
        raise ValueError(f'GPU {args.gpu_id} is unavailable; visible count={torch.cuda.device_count()}')
    torch.cuda.set_device(args.gpu_id)
    device = torch.device(f'cuda:{args.gpu_id}')
    text_configs = [ClipConfig(**metadata[s]['train'].get('clip', {})) for s in ('old', 'new')]
    if asdict(text_configs[0]) != asdict(text_configs[1]):
        raise ValueError('Text encoder configurations differ')
    embedder_cache = session.setdefault('text_encoders', {})
    embedder_key = (args.gpu_id, repr(asdict(text_configs[0])))
    if embedder_key not in embedder_cache:
        print('Loading text encoder once for this run...', flush=True)
        embedder_cache[embedder_key] = CLIPTextEmbedder(text_configs[0], device=device)
    embedder = embedder_cache[embedder_key]
    dataset_cache = session.setdefault('datasets', {})
    batches, arrays, clocks = {}, {}, {}
    for side in ('old', 'new'):
        meta, config = metadata[side], configs[side]
        dataset_root = Path(source['cache_roots'][side])/row['split']
        dataset_key = (checkpoint_keys[side], str(dataset_root.resolve()))
        if dataset_key not in dataset_cache:
            prompt = PromptConfig(**meta['train'].get('prompt', {}))
            labels = load_operator_label_maps(prompt.operator_label_map_path) if prompt.use_operator_id_as_prompt else {}
            print(f'Opening {side} dataset once for this run...', flush=True)
            dataset_cache[dataset_key] = EpisodeDataset(dataset_root, meta['stats'],
                train=False, mask_state_ratio=0., model_config=config,
                default_task_name='load_the_plates_into_the_dish_rack',
                prompt_config=prompt, operator_label_maps=labels)
        dataset = dataset_cache[dataset_key]
        index = row[f'{side}_dataset_index']
        ep = int(np.searchsorted(dataset.cum, index, side='right'))
        local = int(index - (dataset.cum[ep-1] if ep else 0))
        folder = Path(row[f'{side}_episode_path'])
        if dataset.episodes[ep][0].resolve() != folder.resolve() or local != row[f'{side}_frame']:
            raise ValueError('Saved source-frame mapping changed')
        sample = dataset[index]
        batches[side] = batch_to_device(collate([sample], config.camera_keys), device, embedder, config.camera_keys)
        values = read_state_action_rows(folder, local, local+row['chunk_length'], config)
        arrays[f'{side}_state'], arrays[f'{side}_action'] = values[:, :14], values[:, 14:]
        saved = json.loads((folder/'episode_metadata.json').read_text())
        if 't0_ns' not in saved or 'tick_ns' not in saved:
            raise ValueError('Missing official frame clock metadata')
        # Official export stores the clock origin before its first tick:
        # frame zero is t0 + tick_ns, including after segment slicing.
        clocks[side] = [int(saved['t0_ns']+(local+i+1)*saved['tick_ns']) for i in range(row['chunk_length'])]
        if 'top_camera_timestamps_ns' in saved:
            arrays[f'{side}_top_image_timestamps_ns'] = np.asarray(
                saved['top_camera_timestamps_ns'][local:local+row['chunk_length']], dtype=np.int64)
        cameras = saved.get('cameras') or config.camera_keys
        length = (folder/'states_actions.bin').stat().st_size//(28*8)
        native = decode_frame(folder, local, length, cameras, config.camera_keys)
        for camera in config.camera_keys:
            arrays[f'{side}_condition_image_{camera}'] = batches[side]['images'][camera].cpu().float().numpy()
            arrays[f'{side}_decoded_image_{camera}'] = native[camera].cpu().numpy()
    if clocks['old'] != clocks['new']:
        raise ValueError('Old/new physical frame timestamps differ')
    if all(f'{s}_top_image_timestamps_ns' in arrays for s in ('old', 'new')):
        np.testing.assert_array_equal(arrays['old_top_image_timestamps_ns'], arrays['new_top_image_timestamps_ns'])
    arrays['ticks_ns'] = np.asarray(clocks['old'], dtype=np.int64)
    arrays['noise'], original_time = noise_and_times([row['sample_id']], (row['chunk_length'], 14),
                                                      1, source['seed'], 0, device)
    noise = arrays.pop('noise')[0]
    arrays['noise'] = noise.cpu().numpy()
    steps = args.diffusion_steps
    print(f'Generating four actual prediction chunks, {steps} flow steps, shared initial noise...', flush=True)
    model_cache = session.setdefault('models', {})
    for model_side in ('old', 'new'):
        dino_bf16 = metadata[model_side]['train'].get('dino_bf16', True)
        model_key = (checkpoint_keys[model_side], args.gpu_id, dino_bf16)
        if model_key not in model_cache:
            print(f'Loading {model_side} model once; keeping it resident on {device}...', flush=True)
            model = DiTPolicy(configs[model_side])
            load_pretrained(model, Path(checkpoint_keys[model_side]))
            model.img_backbone.set_bfloat16(dino_bf16)
            model_cache[model_key] = model.to(device).eval()
        else:
            print(f'Reusing resident {model_side} model...', flush=True)
        model = model_cache[model_key]
        with torch.inference_mode():
            for data_side in ('old', 'new'):
                predicted = model.sample_actions(batches[data_side], num_steps=steps, noise=noise.clone())
                stats = metadata[model_side]['stats']['actions']
                values = predicted.cpu().float().numpy()[0]*(stats['std']+1e-6)+stats['mean']
                if values.shape != (row['chunk_length'], 14) or not np.isfinite(values).all():
                    raise ValueError('Invalid generated action chunk')
                arrays[f'{model_side}_on_{data_side}_prediction'] = values
                print(f'Generated {model_side}_on_{data_side}', flush=True)
    with np.load(args.loss_run/'screen_draws.npz') as saved:
        losses = {cell: float(saved[cell][source_index, 0].mean()) for cell in CELLS}
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output/'bundle.npz'
    if target.exists():
        raise ValueError('Use a new output directory')
    np.savez_compressed(target, **arrays)
    save_json(args.output/'bundle.json', dict(
        chunk=row, checkpoints={s: source[f'{s}_checkpoint'] for s in ('old', 'new')},
        checkpoint_steps={s: metadata[s]['step'] for s in ('old', 'new')},
        losses=losses, loss_draw_time=float(original_time[0, 0, 0, 0]),
        generation='Native sample_actions, pure noise t=1 -> t=0; no prefix; no joint clipping',
        generation_steps=steps, noise='Original screen draw-0 epsilon shared by all four combinations',
        seed=source['seed'], loss_run=str(args.loss_run), exact_saved_labels=True,
    ))
    archive = args.output/'preview_bundle.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as zipped:
        for path in (target, args.output/'bundle.json'):
            zipped.write(path, arcname=path.name)
    print(f'Portable prediction bundle: {target} ({target.stat().st_size:,} bytes)', flush=True)
    print(f'Copy back this archive only: {archive} ({archive.stat().st_size:,} bytes)', flush=True)
    return target


def render_bundle(args, bundle):
    # Imports are delayed so exporting predictions never initializes OpenGL.
    import imageio.v2 as imageio
    import mujoco
    from mcap.reader import make_reader
    from mcap_protobuf.decoder import DecoderFactory
    from scipy.ndimage import binary_erosion
    from scipy.spatial.transform import Rotation
    from abc_minimal.export_mcap import STATE_TOPICS, ACTION_TOPICS, floor_indices
    from scripts.verify_camera_fk_ik import Robot, pose, video_frames

    metadata = json.loads(bundle.with_suffix('.json').read_text())
    with np.load(bundle) as saved:
        arrays = {k: saved[k].copy() for k in saved.files}
    row = metadata['chunk']
    count, first = row['chunk_length'], row['source_frame']
    rendered_cells = ('old_on_old', 'new_on_new')
    for name in ('old_state', 'old_action', 'new_state', 'new_action',
                 *[f'{c}_prediction' for c in rendered_cells]):
        if arrays[name].shape != (count, 14) or not np.isfinite(arrays[name]).all():
            raise ValueError(f'Missing/nonfinite data: {name}')
    record = next(r for r in json.load(gzip.open(args.calibrations, 'rt'))['records']
                  if r['episode'] == row['source_episode'])
    calibration = record['calibration']
    intrinsics = calibration['recorded_intrinsics']
    path = args.raw_root/record['source_relative']
    top_topic = calibration['camera_topic']
    topics = [top_topic, '/left-wrist-camera', '/right-wrist-camera']
    if top_topic != '/top-camera':
        raise ValueError('This native-pixel replay currently requires calibrated monocular top video')
    scalars = {t: [] for t, _ in STATE_TOPICS+ACTION_TOPICS}
    camera_clocks = {t: [] for t in topics}
    with path.open('rb') as stream:
        reader = make_reader(stream, decoder_factories=[DecoderFactory()])
        for _, channel, message, value in reader.iter_decoded_messages(topics=[*scalars, *topics]):
            if channel.topic in scalars:
                scalars[channel.topic].append((message.log_time, value.position))
            else:
                camera_clocks[channel.topic].append(message.log_time)
    all_clocks = [*[sorted(c) for c in camera_clocks.values()],
                  *[[v[0] for v in sorted(c, key=lambda row: row[0])] for c in scalars.values()]]
    tick_ns = 33333333
    expected_ticks = max(c[0] for c in all_clocks)+(np.arange(first, first+count)+1)*tick_ns
    np.testing.assert_array_equal(expected_ticks, arrays['ticks_ns'])
    parts = []
    for topic, dim in STATE_TOPICS+ACTION_TOPICS:
        values = sorted(scalars[topic], key=lambda v: v[0])
        ts = np.array([v[0] for v in values], dtype=np.int64)
        parts.append(np.asarray([v[1] for v in values]).reshape(-1, dim)[floor_indices(ts, expected_ticks)])
    original_rows = np.concatenate(parts, axis=1)
    images = []
    for camera_index, topic in enumerate(topics):
        frames = iter(video_frames(path, topic))
        current_time, current = next(frames)
        following = next(frames, None)
        selected = []
        selected_times = []
        for tick in expected_ticks:
            while following is not None and following[0] <= tick:
                current_time, current = following
                following = next(frames, None)
            if current_time > tick:
                raise ValueError('No causal image')
            selected.append(current.to_ndarray(format='rgb24'))
            selected_times.append(current_time)
        if camera_index == 0:
            for side in ('old', 'new'):
                if f'{side}_top_image_timestamps_ns' in arrays:
                    np.testing.assert_array_equal(selected_times, arrays[f'{side}_top_image_timestamps_ns'])
        images.append(selected)
    sizes = {(im.shape[1], im.shape[0]) for cam in images for im in cam}
    width, height = intrinsics['width'], intrinsics['height']
    if sizes != {(width, height)}:
        raise ValueError(f'Native cameras have different sizes {sizes}; refusing to resize original images')
    real = Robot(ROOT/'abc_sim/models/yam_bimanual_empty.xml', calibration['base_poses'])
    virtual = Robot(ROOT/'abc_sim/models/yam_dishrack_base.xml')
    # Robot's calibration helper hides scenery for overlays. Restore the whole
    # original XML scene here, matching the original saved-joint replay view.
    original_model = mujoco.MjModel.from_xml_path(str(ROOT/'abc_sim/models/yam_dishrack_base.xml'))
    virtual.model.geom_group[:] = original_model.geom_group
    del original_model
    virtual.options = mujoco.MjvOption()
    virtual.options.geomgroup[:] = 0
    virtual.options.geomgroup[:3] = 1
    virtual.options.sitegroup[:] = 0
    virtual.model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    mujoco.mj_resetDataKeyframe(virtual.model, virtual.data, virtual.model.key('home').id)
    mujoco.mj_forward(real.model, real.data)
    mujoco.mj_forward(virtual.model, virtual.data)
    real_camera_from_world = np.asarray(calibration['left_base_to_camera'])
    world_from_camera = pose(virtual.data.camera('top'))
    world_from_camera[:3, :3] @= np.diag([1., -1., -1.])
    camera_from_world = np.linalg.inv(world_from_camera)
    # The colored reference uses the original real joints and each episode's
    # calibrated real bases, expressed in the unchanged simulation world.
    # It must never reuse the processed-data/XML bases.
    sim_world_from_real_world = world_from_camera@real_camera_from_world
    reference_bases = {side: sim_world_from_real_world@np.asarray(calibration['base_poses'][side])
                       for side in ('left', 'right')}
    reference_robot = Robot(ROOT/'abc_sim/models/yam_bimanual_empty.xml', reference_bases)
    reference_robot.model.camera('top').fovy[:] = virtual.model.camera('top').fovy
    reference_robot.model.vis.quality.offsamples = 2
    for geom in range(reference_robot.model.ngeom):
        if 'camera' in reference_robot.model.body(int(reference_robot.model.geom_bodyid[geom])).name:
            reference_robot.model.geom_group[geom] = 0
    mujoco.mj_forward(reference_robot.model, reference_robot.data)
    reference_geoms = [[i for i in range(reference_robot.model.ngeom)
                        if reference_robot.model.body(int(reference_robot.model.geom_bodyid[i])).name.startswith(side+'_')]
                       for side in ('left', 'right')]
    virtual.model.vis.quality.offsamples = 2
    renderer = mujoco.Renderer(virtual.model, height, width)
    reference_renderer = mujoco.Renderer(reference_robot.model, height, width)
    reference_renderer.enable_segmentation_rendering()
    f = height/(2*np.tan(np.deg2rad(float(virtual.model.camera('top').fovy[0]))/2))
    colors = [(40, 210, 255), (255, 165, 45)]
    font = ImageFont.truetype('DejaVuSans.ttf', 18)
    small = ImageFont.truetype('DejaVuSans.ttf', 16)
    header, footer, top = 42, 100, 62
    panel_height = header+height+footer
    camera_row_height = header+height

    def pixel(point):
        if point[2] <= 0:
            return None
        x, y = point[:2]/point[2]
        uv = f*x+(width-1)/2, f*y+(height-1)/2
        return uv if np.isfinite(uv).all() and 0 <= uv[0] < width and 0 <= uv[1] < height else None

    def poses(values, robot=virtual):
        robot.set_state(values)
        transform = camera_from_world if robot is virtual else real_camera_from_world
        return np.array([transform@robot.tcp(arm) for arm in range(2)])

    def reference_masks(values, expected):
        reference_robot.set_state(values)
        actual = np.array([camera_from_world@reference_robot.tcp(arm) for arm in range(2)])
        np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-12)
        reference_renderer.update_scene(reference_robot.data, camera='top', scene_option=reference_robot.options)
        for eye in reference_renderer.scene.camera:
            eye.pos[:] = world_from_camera[:3, 3]
            eye.forward[:] = world_from_camera[:3, 2]
            eye.up[:] = -world_from_camera[:3, 1]
        ids = reference_renderer.render()[..., 0]
        return [np.isin(ids, geoms) for geoms in reference_geoms]

    def pose_errors(actual, target):
        return [[float(np.linalg.norm(a[:3, 3]-b[:3, 3])*1000),
                 float(np.rad2deg(Rotation.from_matrix(a[:3, :3]@b[:3, :3].T).magnitude()))]
                for a, b in zip(actual, target)]

    def make_panel(rgb, title, lines, border=None):
        result = Image.new('RGB', (width, panel_height if lines else camera_row_height), '#151b24')
        if rgb is not None:
            result.paste(Image.fromarray(rgb), (0, header))
        draw = ImageDraw.Draw(result)
        draw.text((12, 10), title, font=font, fill='white')
        for i, line in enumerate(lines):
            draw.text((12, header+height+7+23*i), line, font=small, fill='white')
        if border:
            draw.rectangle((1, header+1, width-2, header+height-2), outline=border, width=3)
        return result

    prediction_inputs = {f'{cell}_prediction': cell.split('_on_')[1] for cell in rendered_cells}
    report_frames = []
    layout = [
        ('old_state', 'OLD data: state', 'physical_state'),
        ('old_action', 'OLD data: action label', 'physical_action'),
        ('old_on_old_prediction', 'OLD model / OLD input: generated', 'physical_action'),
        ('new_state', 'NEW data: state', 'physical_state'),
        ('new_action', 'NEW data: action label', 'physical_action'),
        ('new_on_new_prediction', 'NEW model / NEW input: generated', 'physical_action'),
    ]
    destination = args.output
    destination.mkdir(parents=True, exist_ok=True)
    if any((destination/name).exists() for name in ('comparison.mp4', 'comparison_rgb_lossless.mp4')):
        raise ValueError('Use a new output directory')
    options = dict(codec='libx264rgb', pixelformat='rgb24', quality=None,
                   macro_block_size=1, output_params=['-crf', '0', '-preset', 'fast', '-threads', '1'])
    print('Rendering XML joint replays + colored original joints at calibrated real bases...', flush=True)
    try:
        with imageio.get_writer(str(destination/'comparison_rgb_lossless.mp4'), fps=30, **options) as video:
            for index in range(count):
                reference = {'physical_state': poses(original_rows[index, :14], real),
                             'physical_action': poses(original_rows[index, 14:], real)}
                reference.update({f'{s}_action': poses(arrays[f'{s}_action'][index]) for s in ('old', 'new')})
                masks = {'physical_state': reference_masks(original_rows[index, :14], reference['physical_state']),
                         'physical_action': reference_masks(original_rows[index, 14:], reference['physical_action'])}
                canvas = Image.new('RGB', (3*width, camera_row_height+2*panel_height+top), '#0c1119')
                draw = ImageDraw.Draw(canvas)
                draw.text((12, 6), f"{row['source_episode']} | source {first+index} | 30 Hz | same-size native pixels", font=font, fill='white')
                guide = (f'White/gray + white cross: XML arm/TCP | Cyan/orange + circle: original real arm/TCP | loss: old={metadata["losses"]["old_on_old"]:.6f}, '
                         f'new={metadata["losses"]["new_on_new"]:.6f}, t={metadata["loss_draw_time"]:.6f} | '
                         'red: >0.1 mm/deg; amber: >30 deg step')
                draw.text((12, 33), guide, font=small, fill='#a9c3d7')
                for camera, label in enumerate(('Real data: TOP', 'Real data: LEFT wrist', 'Real data: RIGHT wrist')):
                    p = make_panel(images[camera][index], label, [])
                    canvas.paste(p, (camera*width, top))
                frame_report = {'source_frame': first+index}
                for slot, (key, title, target_name) in enumerate(layout, start=3):
                    actual = poses(arrays[key][index])
                    target = reference[target_name]
                    errors = pose_errors(actual, target)
                    renderer.update_scene(virtual.data, camera='top', scene_option=virtual.options)
                    rgb = renderer.render()
                    for mask, color in zip(masks[target_name], colors):
                        rgb[mask] = np.rint(.76*rgb[mask]+.24*np.asarray(color)).astype(np.uint8)
                        rgb[mask & ~binary_erosion(mask)] = color
                    jump = None if index == 0 else float(np.rad2deg(np.abs(
                        arrays[key][index, JOINT_COLUMNS]-arrays[key][index-1, JOINT_COLUMNS])).max())
                    is_prediction = key in prediction_inputs
                    if is_prediction:
                        data_side = prediction_inputs[key]
                        truth = arrays[f'{data_side}_action'][index]
                        joint_error = float(np.rad2deg(np.abs(arrays[key][index, JOINT_COLUMNS]-truth[JOINT_COLUMNS])).mean())
                        grip_error = float(np.abs(arrays[key][index, [6, 13]]-truth[[6, 13]]).max())
                        label_errors = pose_errors(actual, reference[f'{data_side}_action'])
                        last_line = f'vs label: joint MAE {joint_error:.2f} deg | grip max abs {grip_error:.4f}'
                    else:
                        label_errors = None
                        grip_truth = original_rows[index, :14] if key.endswith('state') else original_rows[index, 14:]
                        grip_error = float(np.abs(arrays[key][index, [6, 13]]-grip_truth[[6, 13]]).max())
                        last_line = f'Grip max abs difference vs physical stream: {grip_error:.4f}'
                    border = '#ff4040' if np.any(np.asarray(errors)>.1) else ('#ffb347' if jump is not None and jump>30 else None)
                    lines = [f'L TCP {errors[0][0]:.4f} mm / {errors[0][1]:.4f} deg | target: {target_name}',
                             f'R TCP {errors[1][0]:.4f} mm / {errors[1][1]:.4f} deg',
                             f'Max raw joint step: {"N/A (chunk start)" if jump is None else f"{jump:.2f} deg"}', last_line]
                    p = make_panel(rgb, title, lines, border)
                    pen = ImageDraw.Draw(p)
                    for arm, (sim_tcp, reference_tcp) in enumerate(zip(actual, target)):
                        # The reference circle shares the overlaid arm's color;
                        # the white cross belongs to the XML replay arm.
                        for transform, cross in ((reference_tcp, False), (sim_tcp, True)):
                            uv = pixel(transform[:3, 3])
                            if uv is None:
                                continue
                            xx, yy = uv[0], uv[1]+header
                            if cross:
                                pen.line((xx-7, yy, xx+7, yy), fill='white', width=2)
                                pen.line((xx, yy-7, xx, yy+7), fill='white', width=2)
                            else:
                                pen.ellipse((xx-5, yy-5, xx+5, yy+5), outline=colors[arm], width=2)
                    canvas.paste(p, ((slot%3)*width, top+camera_row_height+(slot//3-1)*panel_height))
                    frame_report[key] = dict(tcp_error_mm_deg=errors, max_joint_step_deg=jump,
                                             max_gripper_difference=grip_error, border=border,
                                             prediction_vs_label_tcp_error_mm_deg=label_errors,
                                             tcp_marker_pixels=dict(
                                                 real_reference_circle=[pixel(t[:3, 3]) for t in target],
                                                 sim_xml_white_cross=[pixel(t[:3, 3]) for t in actual]))
                video.append_data(np.asarray(canvas))
                if index == 0:
                    canvas.save(destination/'comparison_preview.png')
                report_frames.append(frame_report)
                print(f'Rendered {index+1}/{count}', flush=True)
    finally:
        renderer.close()
        reference_renderer.close()
    # Preserve exact RGB frames in the master. Use a standard, explicitly
    # tagged YUV/BT.709 MP4 for players that misinterpret RGB H.264 4:4:4.
    subprocess.run([
        'ffmpeg', '-nostdin', '-v', 'error', '-y', '-i', str(destination/'comparison_rgb_lossless.mp4'),
        '-vf', 'scale=in_range=pc:out_range=tv:out_color_matrix=bt709,format=yuv420p,'
               'setparams=range=limited:color_primaries=bt709:color_trc=bt709:colorspace=bt709',
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '12', '-profile:v', 'high',
        '-pix_fmt', 'yuv420p', '-color_range', 'tv', '-colorspace', 'bt709',
        '-color_primaries', 'bt709', '-color_trc', 'bt709', '-movflags', '+faststart',
        '-threads', '2', str(destination/'comparison.mp4'),
    ], check=True)
    save_json(destination/'report.json', dict(
        status='complete', bundle=str(bundle), metadata=metadata,
        panels=9, panel_image_size=[width, height], fps=30, frames=count,
        original_pixels_unchanged_in_master=True, complete_predictions=True,
        lossless_master='comparison_rgb_lossless.mp4', playback='comparison.mp4',
        playback_encoding='H.264 yuv420p, explicit BT.709 limited range; high-quality playback copy',
        rendered_prediction_cells=list(rendered_cells), cross_predictions_displayed=False,
        normal_borders=False, camera_annotation_notes=False,
        render_model=str(ROOT/'abc_sim/models/yam_dishrack_base.xml'), render_base_mode='xml',
        playback_mode='direct_qpos_fk_replay', physics_steps=0,
        sim_camera='Original fixed XML top camera',
        real_image_overlay=False,
        reference_arm_overlay=True, reference_base_mode='per_sequence_real_calibration_6d',
        tcp_markers=dict(real_reference='cyan/orange circle, matching the overlaid arm',
                         sim_xml='white cross, matching the XML replay arm'),
        reference_joint_source='Original MCAP state/action on the unchanged official 30 Hz clock',
        reference_base_poses_world={side: transform.tolist() for side, transform in reference_bases.items()},
        camera_from_reference_base={side: (camera_from_world@transform).tolist()
                                    for side, transform in reference_bases.items()},
        tcp_error_target='Original real FK; prediction-versus-label TCP errors also saved separately',
        object_reconstruction=False,
        camera_from_base={side: (camera_from_world@pose(virtual.data.body(f'{side}_arm'))).tolist()
                          for side in ('left', 'right')},
        image_source='First row: original MCAP. Lower rows: original simulation XML scene.',
        metrics=report_frames,
    ))
    print(destination/'comparison.mp4', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--loss-run', type=Path)
    parser.add_argument('--seven', action='store_true', help='Process the seven agreed cases in their original two loss runs')
    parser.add_argument('--resume', action='store_true', help='With --seven --export-only, verify and reuse completed bundles')
    parser.add_argument('--loss-root', type=Path, default=Path('/mnt/afs/L202500703/abc_storage/output'))
    parser.add_argument('--bundle-root', type=Path, help='Extracted seven_chunk_bundles.zip, for local rendering without checkpoints')
    parser.add_argument('--chunk', default=DEFAULT_CHUNK)
    parser.add_argument('--bundle', type=Path)
    parser.add_argument('--export-only', action='store_true')
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--diffusion-steps', type=int, default=10)
    parser.add_argument('--raw-root', type=Path)
    parser.add_argument('--calibrations', type=Path, default=ROOT/'scripts/dishrack_joint_retarget_calibrations.json.gz')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.resume and not (args.seven and args.export_only):
        parser.error('--resume requires --seven --export-only')
    if args.seven:
        if args.loss_run or args.bundle:
            parser.error('--seven uses --loss-root or --bundle-root, not --loss-run/--bundle')
        if args.bundle_root and args.export_only:
            parser.error('--bundle-root is for rendering, not --export-only')
        if not args.export_only and args.raw_root is None:
            parser.error('--raw-root is required for rendering; --export-only exports small prediction bundles')
        if args.diffusion_steps < 1:
            parser.error('--diffusion-steps must be positive')
        seven_cases(args)
        return
    if args.bundle_root:
        parser.error('--bundle-root requires --seven')
    if not (args.loss_run or args.bundle) or (args.loss_run and args.bundle):
        parser.error('Provide exactly one of --loss-run or --bundle')
    if args.bundle and args.export_only:
        parser.error('--export-only requires --loss-run')
    if args.diffusion_steps < 1:
        parser.error('--diffusion-steps must be positive')
    if not args.export_only and args.raw_root is None:
        parser.error('--raw-root is required for original-resolution rendering')
    bundle = args.bundle or export_predictions(args)
    if not args.export_only:
        render_bundle(args, bundle)


if __name__ == '__main__':
    main()
