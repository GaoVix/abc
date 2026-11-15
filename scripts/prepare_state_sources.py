"""Read state rows on exactly the same official clock as the saved action inputs."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import hashlib
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np
from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory

from abc_minimal.export_mcap import ACTION_TOPICS, CAMERAS, STATE_TOPICS, TICK_NS, floor_indices


def read_mcap(path):
    scalar = {topic: [] for topic, _ in STATE_TOPICS + ACTION_TOPICS}
    camera_topics = [topic for _, topic in CAMERAS] + ['/top-camera', '/top-left-camera', '/top-right-camera']
    cameras, decoders = {}, {}
    factory = DecoderFactory()
    with path.open('rb') as stream:
        for schema, channel, message in make_reader(stream).iter_messages(
                topics=[*scalar, *camera_topics], log_time_order=False):
            if channel.topic in scalar:
                if channel.id not in decoders:
                    decoders[channel.id] = factory.decoder_for(channel.message_encoding, schema)
                scalar[channel.topic].append((message.log_time, list(decoders[channel.id](message.data).position)))
            else:
                cameras.setdefault(channel.topic, []).append(message.log_time)
    for rows in (*scalar.values(), *cameras.values()):
        rows.sort(key=lambda row: row[0] if isinstance(row, tuple) else row)
    if '/top-left-camera' in cameras and '/top-right-camera' in cameras:
        top = '/top-left-camera' if hashlib.sha1(path.parent.name.encode()).digest()[0] % 2 == 0 else '/top-right-camera'
    else:
        top = '/top-camera'
    clocks = [cameras[top], *[cameras[topic] for _, topic in CAMERAS if topic in cameras]]
    clocks += [[row[0] for row in rows] for rows in scalar.values()]
    ticks = np.arange(max(t[0] for t in clocks) + TICK_NS,
                      min(t[-1] for t in clocks) + 1, TICK_NS, dtype=np.int64)
    parts = []
    for topic, _ in STATE_TOPICS + ACTION_TOPICS:
        rows = scalar[topic]
        timestamps = np.array([row[0] for row in rows], dtype=np.int64)
        parts.append(np.asarray([row[1] for row in rows])[floor_indices(timestamps, ticks)])
    return np.concatenate(parts, axis=-1)


def prepare(record):
    output = Path(record['output'])
    source = Path(record['raw_source'])
    info = source.stat()
    reference_info = Path(record['source']).stat()
    signature = hashlib.sha256(json.dumps([str(source), info.st_size, info.st_mtime_ns,
                                           record['source'], reference_info.st_size,
                                           reference_info.st_mtime_ns]).encode()).hexdigest()
    report_path = output.with_suffix('.json')
    if output.exists() and report_path.exists():
        previous = json.loads(report_path.read_text())
        if previous['signature'] != signature:
            raise ValueError(f"State source changed: {record['key']}")
        return previous
    started = time.monotonic()
    rows = (read_mcap(source) if record['kind'] == 'real'
            else np.fromfile(source, dtype=np.float64).reshape(-1, 28))
    with np.load(record['source']) as saved:
        np.testing.assert_array_equal(rows[:, 14:], saved['actions'])
    states = rows[:, :14]
    if not np.isfinite(states).all():
        raise ValueError(f"Non-finite state: {record['key']}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.partial.npz')
    np.savez_compressed(temporary, states=states, frame_indices=np.arange(len(states)),
                        input_stream='states', raw_source=str(source), source_signature=signature)
    temporary.replace(output)
    report = dict(episode=record['key'], kind=record['kind'], status='complete', frames=len(states),
                  signature=signature, raw_source=str(source), source_actions_match_saved_action_clock=True,
                  seconds=time.monotonic() - started)
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--raw-root', type=Path, required=True)
    parser.add_argument('--sim-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--episodes', nargs='*')
    args = parser.parse_args()
    with gzip.open(args.manifest, 'rt') as stream:
        real = {r['episode']: r for r in json.load(stream)['records'] if r['calibration_quality'] == 'accepted'}
    sim = {p.name: p for split in ('train_sim', 'val_sim') for p in (args.sim_root / split).iterdir()}
    jobs = json.loads((args.reference / 'episodes.json').read_text())
    if args.episodes:
        jobs = [record for record in jobs if any(key in record['key'] for key in args.episodes)]
    for record in jobs:
        kind, episode = record['kind'], record['key'].removeprefix(record['kind'] + '_')
        source = args.raw_root / real[episode]['source_relative'] if kind == 'real' else sim[episode] / 'states_actions.bin'
        record.update(raw_source=str(source), output=str(args.output / kind / 'sources' / (record['key'] + '.npz')))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'jobs.json').write_text(json.dumps(jobs, indent=2) + '\n')
    started, reports = time.monotonic(), []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        for future in as_completed([pool.submit(prepare, record) for record in jobs]):
            result = future.result()
            reports.append(result)
            progress = dict(status='running', selected=len(jobs), completed=len(reports),
                            frames=sum(r['frames'] for r in reports), seconds=time.monotonic() - started)
            (args.output / 'progress.json').write_text(json.dumps(progress, indent=2) + '\n')
            if len(reports) % 50 == 0:
                print(json.dumps(progress), flush=True)
    progress['status'] = 'complete'
    (args.output / 'progress.json').write_text(json.dumps(progress, indent=2) + '\n')
    (args.output / 'summary.json').write_text(json.dumps(dict(**progress, episodes=reports), indent=2) + '\n')
    print(json.dumps(progress), flush=True)


if __name__ == '__main__':
    main()
