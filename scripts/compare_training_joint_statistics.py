"""Read actual training exports and compare official-timing Mink with three-stage IK.

No IK, video processing, clipping, normalization, angle wrapping, or dataset writes.
Requires NumPy and the existing compare_ik_smoothness.py in this scripts directory.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import numpy as np

try:
    from compare_ik_smoothness import discover_mink, discover_three_stage, read_rows
except ModuleNotFoundError:
    from scripts.compare_ik_smoothness import discover_mink, discover_three_stage, read_rows


STREAMS = ('state', 'action')
ARMS = ('left', 'right')
COLUMNS = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12,
           14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25, 26]
GROUPS = ('mink_as_trained', 'mink_matched', 'three_stage_retained')
PERCENTILES = (1, 5, 25, 50, 75, 90, 95, 99, 99.9)
TOLERANCE_RAD = 1e-6


def save(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def xml_limits(path):
    tree = ET.parse(path)
    compiler = tree.find('compiler')
    scale = np.pi / 180 if compiler is None or compiler.get('angle', 'degree') == 'degree' else 1.
    values = []
    for side in ARMS:
        row = []
        for number in range(1, 7):
            joint = tree.find(f'.//worldbody//joint[@name="{side}_joint{number}"]')
            if joint is None or 'range' not in joint.attrib:
                raise ValueError(f'Expected explicit XML arm-joint range: {side}_joint{number}')
            bounds = [float(x) * scale for x in joint.attrib['range'].split()]
            if len(bounds) != 2 or bounds[0] >= bounds[1]:
                raise ValueError(f'Invalid joint range: {bounds}')
            row.append(bounds)
        values.append(row)
    ranges = np.array(values)
    return {'original_xml': (ranges[..., 0], ranges[..., 1]),
            'mink_xml_inward_0p1rad': (ranges[..., 0] + .1, ranges[..., 1] - .1),
            'xml_outward_15deg': (ranges[..., 0] - np.deg2rad(15),
                                 ranges[..., 1] + np.deg2rad(15))}


def describe(values):
    """Exact statistics across time, preserving stream/arm/joint axes."""
    if not len(values) or not np.isfinite(values).all():
        raise ValueError('Empty/non-finite included joint samples')
    result = {}
    for si, stream in enumerate(STREAMS):
        for ai, arm in enumerate(ARMS):
            a = values[:, si, ai]
            absolute = np.abs(a)
            signed_q = np.percentile(a, PERCENTILES, axis=0)
            absolute_q = np.percentile(absolute, PERCENTILES, axis=0)
            result[f'{stream}/{arm}'] = dict(
                samples_per_joint=len(a), mean=a.mean(axis=0).tolist(),
                std=a.std(axis=0).tolist(), rms=np.sqrt(np.mean(a*a, axis=0)).tolist(),
                mean_absolute=absolute.mean(axis=0).tolist(),
                minimum=a.min(axis=0).tolist(), maximum=a.max(axis=0).tolist(),
                signed_percentiles={str(p): signed_q[i].tolist() for i, p in enumerate(PERCENTILES)},
                absolute_percentiles={str(p): absolute_q[i].tolist() for i, p in enumerate(PERCENTILES)},
                component_rms=float(np.sqrt(np.mean(a*a))),
                component_mean_absolute=float(absolute.mean()),
                component_maximum_absolute=float(absolute.max()),
                component_absolute_percentiles={str(p): float(v) for p, v in
                                               zip(PERCENTILES, np.percentile(absolute, PERCENTILES))})
    return result


def chunk_violations(bad, lengths, chunk_length=30):
    offset, total, affected = 0, 0, 0
    for length in lengths:
        if length >= chunk_length:
            prefix = np.r_[0, np.cumsum(bad[offset:offset+length], dtype=np.int64)]
            affected += int(((prefix[chunk_length:] - prefix[:-chunk_length]) > 0).sum())
            total += length - chunk_length + 1
        offset += length
    assert offset == len(bad)
    return dict(chunks=total, violating_chunks=affected,
                violating_chunk_percent=100 * affected / total if total else None)


def limit_statistics(q_deg, lengths, bounds):
    q = np.deg2rad(q_deg)
    result = {}
    for name, (lower, upper) in bounds.items():
        excess = np.maximum(0., np.maximum(lower - q, q - upper))
        bad = excess > TOLERANCE_RAD
        frame_bad = bad.any(axis=(1, 2, 3))
        result[name] = dict(
            frames=len(q), violating_frames=int(frame_bad.sum()),
            violating_frame_percent=float(100 * frame_bad.mean()),
            violating_joint_values_by_stream_arm_joint=bad.sum(axis=0).tolist(),
            violating_arm_frames_by_stream_arm=bad.any(axis=3).sum(axis=0).tolist(),
            violating_frames_by_stream=bad.any(axis=(2, 3)).sum(axis=0).tolist(),
            maximum_excess_deg=float(np.rad2deg(excess.max())),
            maximum_excess_by_stream_arm_joint_deg=np.rad2deg(excess.max(axis=0)).tolist(),
            **chunk_violations(frame_bad, lengths))
    return result


def collect_metric(mink, new, episodes, order, fps):
    parts = {name: [] for name in GROUPS}
    lengths = {name: [] for name in GROUPS}
    for number, episode in enumerate(episodes, 1):
        split, folder, metadata = mink[episode]
        old = read_rows(folder, metadata)
        if not np.isfinite(old).all():
            raise ValueError(f'Non-finite original Mink training rows: {folder}')
        old_q = old[:, COLUMNS].reshape(-1, 2, 2, 6)
        old_delta = np.diff(old_q, n=order, axis=0) if order else old_q
        parts['mink_as_trained'].append(np.rad2deg(old_delta) * fps**order)
        lengths['mink_as_trained'].append(len(old))
        for begin, end, new_split, destination, meta in new[episode]:
            if new_split != split or end > len(old):
                raise ValueError(f'Split/frame mismatch: {destination}')
            if (meta.get('tick_ns') != metadata.get('tick_ns')
                    or 't0_ns' not in metadata
                    or meta.get('t0_ns') != metadata['t0_ns'] + begin * metadata['tick_ns']):
                raise ValueError(f'Timing mismatch: {destination}')
            if fps == 30 and not np.isclose(metadata['tick_ns'], 1e9 / 30, atol=1):
                raise ValueError(f'Not the expected 30 Hz clock: {folder}')
            current = read_rows(destination, meta)
            if len(current) != end - begin or not np.isfinite(current).all():
                raise ValueError(f'Invalid new training segment: {destination}')
            np.testing.assert_array_equal(current[:, [6, 13, 20, 27]], old[begin:end, [6, 13, 20, 27]])
            new_q = current[:, COLUMNS].reshape(-1, 2, 2, 6)
            for key, q in (('mink_matched', old_q[begin:end]),
                           ('three_stage_retained', new_q)):
                delta = np.diff(q, n=order, axis=0) if order else q
                parts[key].append(np.rad2deg(delta) * fps**order)
                lengths[key].append(len(q))
        if number % 100 == 0 or number == len(episodes):
            print(f'order={order}: {number}/{len(episodes)} sequences', flush=True)
    arrays = {name: np.concatenate(parts.pop(name), axis=0) for name in GROUPS}
    assert arrays['mink_matched'].shape == arrays['three_stage_retained'].shape
    assert lengths['mink_matched'] == lengths['three_stage_retained']
    return arrays, lengths


def flatten_csv(path, result):
    fields = ['metric', 'view', 'stream', 'arm', 'joint', 'samples', 'mean', 'std', 'rms',
              'mean_absolute', 'minimum', 'maximum', 'abs_p50', 'abs_p95', 'abs_p99', 'abs_p99_9']
    with path.open('w', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for metric, groups in result['statistics'].items():
            for view, rows in groups.items():
                for label, stats in rows.items():
                    stream, arm = label.split('/')
                    for j in range(6):
                        row = dict(metric=metric, view=view, stream=stream, arm=arm, joint=j+1,
                                   samples=stats['samples_per_joint'])
                        row.update({k: stats[k][j] for k in ('mean', 'std', 'rms', 'mean_absolute', 'minimum', 'maximum')})
                        row.update({f'abs_p{str(p).replace(".", "_")}': stats['absolute_percentiles'][str(p)][j]
                                    for p in (50, 95, 99, 99.9)})
                        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mink-root', type=Path, required=True)
    parser.add_argument('--three-stage-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--split', choices=('train', 'val', 'both'), default='train')
    parser.add_argument('--xml', type=Path, default=Path(__file__).resolve().parents[1]/'abc_sim/models/yam_dishrack_base.xml')
    parser.add_argument('--expected-sequences', type=int, default=1261,
                        help='Expected total real episode set before selecting train/val')
    parser.add_argument('--fps', type=float, default=30.)
    args = parser.parse_args()
    if args.fps <= 0 or not np.isfinite(args.fps):
        raise ValueError('fps must be finite and positive')
    for name in ('mink_root', 'three_stage_root', 'output_dir', 'xml'):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    for root in (args.mink_root, args.three_stage_root):
        if args.output_dir.is_relative_to(root):
            raise ValueError('Write statistics outside the input dataset')
    if (args.output_dir/'summary.json').exists():
        raise FileExistsError('Choose a new output directory to preserve the existing report')
    old = discover_mink(args.mink_root)
    new = discover_three_stage(args.three_stage_root)
    if set(old) != set(new) or len(old) != args.expected_sequences:
        raise ValueError(f'Episode set mismatch: Mink={len(old)}, three-stage={len(new)}; '
                         f'only_old={sorted(set(old)-set(new))[:3]}, only_new={sorted(set(new)-set(old))[:3]}')
    episodes = sorted(e for e in old if args.split == 'both' or old[e][0] == f'{args.split}_real')
    if not episodes:
        raise ValueError('No selected episodes')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bounds = xml_limits(args.xml)
    started = time.monotonic()
    result = dict(status='running', split=args.split, selected_sequences=len(episodes),
                  total_matching_sequences=len(old), mink_root=str(args.mink_root),
                  three_stage_root=str(args.three_stage_root), fps=args.fps,
                  xml=str(args.xml), xml_sha256=hashlib.sha256(args.xml.read_bytes()).hexdigest(),
                  axis_order=['stream:state,action', 'arm:left,right', 'joint:1..6'],
                  grippers_included=False, angle_wrapping=False,
                  derivative_definition='v=(q[t+1]-q[t])*fps; a=(q[t+2]-2*q[t+1]+q[t])*fps^2; '
                                        'no difference across retained-segment boundaries',
                  views={'mink_as_trained': 'Original Mink data on its full selected split.',
                         'mink_matched': 'Mink on the exact source_frame_range intervals exported for three-stage training.',
                         'three_stage_retained': 'Actual filtered three-stage training segments; identical frames to mink_matched.'},
                  units={'joint_deg':'deg', 'velocity_deg_s':'deg/s', 'acceleration_deg_s2':'deg/s^2'},
                  comparison_tolerance_rad=TOLERANCE_RAD,
                  statistics={}, joint_limits={}, jump_edges={}, angle_change_same_frames={})
    save(args.output_dir/'progress.json', dict(status='running', metric='joint_deg'))
    for order, metric in enumerate(('joint_deg', 'velocity_deg_s', 'acceleration_deg_s2')):
        arrays, lengths = collect_metric(old, new, episodes, order, args.fps)
        result['statistics'][metric] = {}
        for group in GROUPS:
            values = arrays[group]
            result['statistics'][metric][group] = describe(values)
            if order == 0:
                result['joint_limits'][group] = limit_statistics(values, lengths[group], bounds)
            elif order == 1:
                # Count each arm edge once, using the largest of its six raw joint changes.
                step = np.abs(values).max(axis=3) / args.fps
                result['jump_edges'][group] = {str(deg): (step > deg).sum(axis=0).tolist()
                                              for deg in (30, 60, 90)}
        if order == 0:
            delta = arrays['three_stage_retained'] - arrays['mink_matched']
            result['angle_change_same_frames'] = describe(delta)
            del delta
        # The input binaries already retain full samples; avoid copying gigabytes
        # of derivative arrays merely to produce this numerical summary.
        del arrays
        save(args.output_dir/'progress.json', dict(status='running', finished_metric=metric,
                                                  elapsed_seconds=time.monotonic()-started))
    result.update(status='complete', seconds=time.monotonic()-started)
    save(args.output_dir/'summary.json', result)
    flatten_csv(args.output_dir/'per_joint.csv', result)
    lines = [f'split={args.split}, real sequences={len(episodes)}',
             'Matched intervals: official-timing Mink -> final three-stage IK',
             'stream/arm       velocity RMS deg/s       acceleration RMS deg/s^2     acceleration MAX deg/s^2']
    for stream in STREAMS:
        for arm in ARMS:
            label = f'{stream}/{arm}'
            values = []
            for metric, key in [('velocity_deg_s', 'component_rms'),
                                ('acceleration_deg_s2', 'component_rms'),
                                ('acceleration_deg_s2', 'component_maximum_absolute')]:
                a, b = (result['statistics'][metric][g][label][key]
                        for g in ('mink_matched', 'three_stage_retained'))
                values.append(f'{a:.4f} -> {b:.4f}')
            lines.append(f'{label:14s}   ' + '   '.join(values))
    lines.append('Joint-limit comparison on matched frames/chunks:')
    for limit in bounds:
        rows = [result['joint_limits'][g][limit] for g in ('mink_matched', 'three_stage_retained')]
        lines.append(f'{limit}: frame {rows[0]["violating_frames"]}/{rows[0]["frames"]} -> '
                     f'{rows[1]["violating_frames"]}/{rows[1]["frames"]}; chunk '
                     f'{rows[0]["violating_chunks"]}/{rows[0]["chunks"]} -> '
                     f'{rows[1]["violating_chunks"]}/{rows[1]["chunks"]}')
    text = '\n'.join(lines) + '\n'
    (args.output_dir/'comparison.txt').write_text(text)
    save(args.output_dir/'progress.json', dict(status='complete', seconds=result['seconds']))
    print(text, flush=True)
    print(f'SUMMARY={args.output_dir / "summary.json"}\nCSV={args.output_dir / "per_joint.csv"}', flush=True)


if __name__ == '__main__':
    main()
