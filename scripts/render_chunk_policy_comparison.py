"""Export real model predictions, then render full paired chunk overlays.

`--export-only` runs where the fine-tuned checkpoints/caches are available. Its
small portable bundle can be copied to a machine with the original MCAP, then
rendered with `--bundle`. No inference, training or data conversion is changed.
"""
import os

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(variable, '1')

import argparse
from dataclasses import asdict
import gzip
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHUNK = 'episode_298becaa-b397-4766-9c80-6fb74e182f4b:1336'
CELLS = ('old_on_old', 'new_on_old', 'old_on_new', 'new_on_new')
JOINT_COLUMNS = [*range(6), *range(7, 13)]


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def export_predictions(args):
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
    metadata = {s: checkpoint_metadata(Path(source[f'{s}_checkpoint'])) for s in ('old', 'new')}
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
    embedder = CLIPTextEmbedder(text_configs[0], device=device)
    batches, arrays, clocks = {}, {}, {}
    for side in ('old', 'new'):
        meta, config = metadata[side], configs[side]
        prompt = PromptConfig(**meta['train'].get('prompt', {}))
        labels = load_operator_label_maps(prompt.operator_label_map_path) if prompt.use_operator_id_as_prompt else {}
        dataset = EpisodeDataset(Path(source['cache_roots'][side])/row['split'], meta['stats'],
                                 train=False, mask_state_ratio=0., model_config=config,
                                 default_task_name='load_the_plates_into_the_dish_rack',
                                 prompt_config=prompt, operator_label_maps=labels)
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
    for model_side in ('old', 'new'):
        model = DiTPolicy(configs[model_side])
        load_pretrained(model, Path(source[f'{model_side}_checkpoint']))
        model.img_backbone.set_bfloat16(metadata[model_side]['train'].get('dino_bf16', True))
        model.to(device).eval()
        with torch.inference_mode():
            for data_side in ('old', 'new'):
                predicted = model.sample_actions(batches[data_side], num_steps=steps, noise=noise.clone())
                stats = metadata[model_side]['stats']['actions']
                values = predicted.cpu().float().numpy()[0]*(stats['std']+1e-6)+stats['mean']
                if values.shape != (row['chunk_length'], 14) or not np.isfinite(values).all():
                    raise ValueError('Invalid generated action chunk')
                arrays[f'{model_side}_on_{data_side}_prediction'] = values
                print(f'Generated {model_side}_on_{data_side}', flush=True)
        del model
        torch.cuda.empty_cache()
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


def inverse_brown(x, y, distortion):
    k1, k2, p1, p2, k3 = distortion
    r2 = x*x+y*y
    radial = 1+r2*(k1+r2*(k2+r2*k3))
    a, b = x*radial, y*radial
    return a+2*p1*a*b+p2*(r2+2*a*a), b+2*p2*a*b+p1*(r2+2*b*b)


def render_bundle(args, bundle):
    # Imports are delayed so exporting predictions never initializes OpenGL.
    import imageio.v2 as imageio
    import mujoco
    from mcap.reader import make_reader
    from mcap_protobuf.decoder import DecoderFactory
    from scipy.ndimage import binary_erosion, map_coordinates
    from scipy.spatial.transform import Rotation
    from abc_minimal.export_mcap import STATE_TOPICS, ACTION_TOPICS, floor_indices
    from scripts.verify_camera_fk_ik import Robot, pose, video_frames

    metadata = json.loads(bundle.with_suffix('.json').read_text())
    with np.load(bundle) as saved:
        arrays = {k: saved[k].copy() for k in saved.files}
    row = metadata['chunk']
    count, first = row['chunk_length'], row['source_frame']
    for name in ('old_state', 'old_action', 'new_state', 'new_action',
                 *[f'{c}_prediction' for c in CELLS]):
        if arrays[name].shape != (count, 14) or not np.isfinite(arrays[name]).all():
            raise ValueError(f'Missing/nonfinite data: {name}')
    record = next(r for r in json.load(gzip.open(args.calibrations, 'rt'))['records']
                  if r['episode'] == row['source_episode'])
    calibration = record['calibration']
    intrinsics = calibration['recorded_intrinsics']
    if intrinsics['distortion_model'] != 'inverse_brown_conrady':
        raise ValueError('Unsupported recorded camera distortion model')
    path = args.raw_root/record['source_relative']
    top_topic = calibration['camera_topic']
    topics = [top_topic, '/left-wrist-camera', '/right-wrist-camera']
    if top_topic != '/top-camera':
        raise ValueError('This native-pixel overlay currently requires calibrated monocular top video')
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
    mujoco.mj_forward(real.model, real.data)
    mujoco.mj_forward(virtual.model, virtual.data)
    real_camera_from_world = np.asarray(calibration['left_base_to_camera'])
    world_from_camera = pose(virtual.data.camera('top'))
    world_from_camera[:3, :3] @= np.diag([1., -1., -1.])
    camera_from_world = np.linalg.inv(world_from_camera)
    for geom in range(virtual.model.ngeom):
        body = virtual.model.body(int(virtual.model.geom_bodyid[geom])).name
        if 'camera' in body:
            virtual.model.geom_group[geom] = 0
    virtual.model.vis.quality.offsamples = 2
    renderer = mujoco.Renderer(virtual.model, height, width)
    renderer.enable_segmentation_rendering()
    k = np.asarray(intrinsics['K']).reshape(3, 3)
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    x, y = inverse_brown((u-k[0, 2])/k[0, 0], (v-k[1, 2])/k[1, 1], intrinsics['D'])
    f = height/(2*np.tan(np.deg2rad(float(virtual.model.camera('top').fovy[0]))/2))
    coordinates = np.array([f*y+(height-1)/2, f*x+(width-1)/2])
    arm_geoms = [[i for i in range(virtual.model.ngeom)
                  if virtual.model.body(int(virtual.model.geom_bodyid[i])).name.startswith(s+'_')]
                 for s in ('left', 'right')]
    colors = [(40, 210, 255), (255, 165, 45)]
    font = ImageFont.truetype('DejaVuSans.ttf', 18)
    small = ImageFont.truetype('DejaVuSans.ttf', 16)
    header, footer, top = 42, 100, 62
    panel_height = header+height+footer

    def pixel(point):
        if point[2] <= 0:
            return None
        a, b = xx, yy = point[:2]/point[2]
        # Points outside the source image need numerical error reporting,
        # but no drawn marker or unstable distortion inversion.
        if max(abs(xx), abs(yy)) > 1.5:
            return None
        for _ in range(20):
            aa, bb = inverse_brown(a, b, intrinsics['D'])
            a, b = a+xx-aa, b+yy-bb
        uv = a*k[0, 0]+k[0, 2], b*k[1, 1]+k[1, 2]
        return uv if np.isfinite(uv).all() and 0 <= uv[0] < width and 0 <= uv[1] < height else None

    def poses(values, robot=virtual):
        robot.set_state(values)
        transform = camera_from_world if robot is virtual else real_camera_from_world
        return np.array([transform@robot.tcp(arm) for arm in range(2)])

    def make_panel(rgb, title, lines, border=None):
        result = Image.new('RGB', (width, panel_height), '#151b24')
        if rgb is not None:
            result.paste(Image.fromarray(rgb), (0, header))
        draw = ImageDraw.Draw(result)
        draw.text((12, 10), title, font=font, fill='white')
        for i, line in enumerate(lines):
            draw.text((12, header+height+7+23*i), line, font=small, fill='white')
        if border:
            draw.rectangle((1, header+1, width-2, header+height-2), outline=border, width=3)
        return result

    prediction_inputs = {f'{cell}_prediction': cell.split('_on_')[1] for cell in CELLS}
    report_frames = []
    layout = [
        ('old_state', 'OLD data: state', 'physical_state'),
        ('old_action', 'OLD data: action label', 'physical_action'),
        ('old_on_old_prediction', 'OLD model / OLD input: generated', 'old_action'),
        ('new_state', 'NEW data: state', 'physical_state'),
        ('new_action', 'NEW data: action label', 'physical_action'),
        ('new_on_new_prediction', 'NEW model / NEW input: generated', 'new_action'),
        ('old_on_new_prediction', 'OLD model / NEW input: generated', 'new_action'),
        ('new_on_old_prediction', 'NEW model / OLD input: generated', 'old_action'),
    ]
    destination = args.output
    destination.mkdir(parents=True, exist_ok=True)
    if (destination/'comparison.mp4').exists():
        raise ValueError('Use a new output directory')
    options = dict(codec='libx264rgb', pixelformat='rgb24', quality=None,
                   macro_block_size=1, output_params=['-crf', '0', '-preset', 'fast', '-threads', '1'])
    print('Rendering all 12 panels; original image pixels unchanged...', flush=True)
    try:
        with imageio.get_writer(str(destination/'comparison.mp4'), fps=30, **options) as video:
            for index in range(count):
                reference = {'physical_state': poses(original_rows[index, :14], real),
                             'physical_action': poses(original_rows[index, 14:], real)}
                reference.update({f'{s}_action': poses(arrays[f'{s}_action'][index]) for s in ('old', 'new')})
                canvas = Image.new('RGB', (3*width, 4*panel_height+top), '#0c1119')
                draw = ImageDraw.Draw(canvas)
                draw.text((12, 6), f"{row['source_episode']} | source {first+index} | 30 Hz | same-size native pixels", font=font, fill='white')
                draw.text((12, 33), 'cyan=L; orange=R; white cross=target TCP | generation uses only first image/state; no joint clipping', font=small, fill='#a9c3d7')
                for camera, label in enumerate(('Shared real TOP', 'Shared real LEFT wrist', 'Shared real RIGHT wrist')):
                    p = make_panel(images[camera][index], label,
                        ['Original source image, shown once; no resize/undistortion.',
                         'Causal 30 Hz clock; no time offset or interpolation.',
                         f'Actual model condition: source frame {first} only.'])
                    canvas.paste(p, (camera*width, top))
                frame_report = {'source_frame': first+index}
                for slot, (key, title, target_name) in enumerate(layout, start=3):
                    actual = poses(arrays[key][index])
                    target = reference[target_name]
                    errors = [[float(np.linalg.norm(a[:3, 3]-b[:3, 3])*1000),
                               float(np.rad2deg(Rotation.from_matrix(a[:3, :3]@b[:3, :3].T).magnitude()))]
                              for a, b in zip(actual, target)]
                    renderer.update_scene(virtual.data, camera='top', scene_option=virtual.options)
                    for eye in renderer.scene.camera:
                        eye.pos[:] = world_from_camera[:3, 3]
                        eye.forward[:] = world_from_camera[:3, 2]
                        eye.up[:] = -world_from_camera[:3, 1]
                    segmentation = renderer.render()
                    ids = map_coordinates(segmentation[..., 0], coordinates, order=0, mode='constant', cval=-1)
                    rgb = images[0][index].copy()
                    for geoms, color in zip(arm_geoms, colors):
                        mask = np.isin(ids, geoms)
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
                        last_line = f'Joint MAE {joint_error:.2f} deg | grip max abs {grip_error:.4f}'
                    else:
                        grip_truth = original_rows[index, :14] if key.endswith('state') else original_rows[index, 14:]
                        grip_error = float(np.abs(arrays[key][index, [6, 13]]-grip_truth[[6, 13]]).max())
                        last_line = f'Grip max abs difference vs physical stream: {grip_error:.4f}'
                    border = '#ff4040' if np.any(np.asarray(errors)>.1) else ('#ffb347' if jump is not None and jump>30 else '#48c989')
                    lines = [f'L TCP {errors[0][0]:.4f} mm / {errors[0][1]:.4f} deg | target: {target_name}',
                             f'R TCP {errors[1][0]:.4f} mm / {errors[1][1]:.4f} deg',
                             f'Max raw joint step: {"N/A (chunk start)" if jump is None else f"{jump:.2f} deg"}', last_line]
                    p = make_panel(rgb, title, lines, border)
                    pen = ImageDraw.Draw(p)
                    for arm, (a, b) in enumerate(zip(actual, target)):
                        for transform, cross in ((b, True), (a, False)):
                            uv = pixel(transform[:3, 3])
                            if uv is None:
                                continue
                            xx, yy = uv[0], uv[1]+header
                            if cross:
                                pen.line((xx-7, yy, xx+7, yy), fill='white', width=2)
                                pen.line((xx, yy-7, xx, yy+7), fill='white', width=2)
                            else:
                                pen.ellipse((xx-5, yy-5, xx+5, yy+5), outline=colors[arm], width=2)
                    canvas.paste(p, ((slot%3)*width, top+(slot//3)*panel_height))
                    frame_report[key] = dict(tcp_error_mm_deg=errors, max_joint_step_deg=jump,
                                             max_gripper_difference=grip_error)
                p = make_panel(None, 'Loss / generation / display guide', [
                    'Red: TCP error >0.1 mm OR >0.1 deg.',
                    'Amber: raw joint step >30 deg (if TCP passes).',
                    'Prediction TCP errors compare against the matching action label.',
                    'Label TCP errors compare against original physical FK.'])
                pen = ImageDraw.Draw(p)
                texts = ['Saved single-draw velocity losses (not generated error):',
                         *[f'{cell}: {metadata["losses"][cell]:.6f}' for cell in CELLS],
                         f'Loss draw time t={metadata["loss_draw_time"]:.6f}',
                         '', f'Full flow generation: t=1 -> 0, {metadata["generation_steps"]} steps.',
                         'All four predictions use the same initial epsilon.',
                         'Input: ONLY first real image + first saved state.',
                         'Later real images/states are sequence context.',
                         'Grippers: saved/predicted opening, coupled fingers.',
                         'No IK, dynamics, joint clipping or interpolation in replay.',
                         'Original scene may occlude the overlaid virtual robot.']
                for line, text in enumerate(texts):
                    pen.text((12, header+14+28*line), text, font=small, fill='white')
                canvas.paste(p, (2*width, top+3*panel_height))
                video.append_data(np.asarray(canvas))
                if index == 0:
                    canvas.save(destination/'comparison_preview.png')
                report_frames.append(frame_report)
                print(f'Rendered {index+1}/{count}', flush=True)
    finally:
        renderer.close()
    save_json(destination/'report.json', dict(
        status='complete', bundle=str(bundle), metadata=metadata,
        panels=12, panel_image_size=[width, height], fps=30, frames=count,
        original_pixels_unchanged=True, complete_predictions=True,
        image_source='Original MCAP; shared raw cameras once; top repeated as overlay background only',
        metrics=report_frames,
    ))
    print(destination/'comparison.mp4', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--loss-run', type=Path)
    parser.add_argument('--chunk', default=DEFAULT_CHUNK)
    parser.add_argument('--bundle', type=Path)
    parser.add_argument('--export-only', action='store_true')
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--diffusion-steps', type=int, default=10)
    parser.add_argument('--raw-root', type=Path)
    parser.add_argument('--calibrations', type=Path, default=ROOT/'scripts/dishrack_joint_retarget_calibrations.json.gz')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
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
