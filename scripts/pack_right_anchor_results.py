"""Pack final IK rows and masks for export against the server's original RGB."""

import argparse
import json
from pathlib import Path
import tarfile
import tempfile

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    settings_path = args.work_dir / 'settings.json'
    settings = json.loads(settings_path.read_text())
    expected = len(settings['inputs'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise ValueError(f'Archive already exists: {args.output}')
    temporary = args.output.with_suffix(args.output.suffix + '.tmp')
    with tempfile.TemporaryDirectory() as folder, tarfile.open(temporary, 'w:gz') as archive:
        archive.add(settings_path, arcname='right_anchor_ik_results/settings.json')
        for stream in ('states', 'actions'):
            root = args.work_dir / stream / 'equality_margin15'
            summary = json.loads((root / 'summary.json').read_text())
            if summary['status'] != 'complete' or summary['failed'] or summary['completed'] != expected:
                raise ValueError(f'Incomplete IK results: {root}')
            paths = sorted((root / 'real').glob('*/trajectory.npz'))
            if len(paths) != expected:
                raise ValueError(f'Wrong result count: {root}')
            for path in paths:
                with np.load(path) as saved:
                    keys = (f'optimized_{stream}', stream, 'frame_indices', 'errors', 'valid_mask',
                            'optimized_mask', 'camera_from_base', 'solver', 'base_mode', 'retarget_sides')
                    values = {key: saved[key] for key in keys}
                    np.testing.assert_array_equal(values[f'optimized_{stream}'][:, 7:14], values[stream][:, 7:14])
                compact = Path(folder) / 'trajectory.npz'
                np.savez_compressed(compact, **values)
                archive.add(compact, arcname=str(Path('right_anchor_ik_results') / stream /
                                                'equality_margin15/real' / path.parent.name / 'trajectory.npz'))
    temporary.replace(args.output)
    print(f'Packed {expected} real sequences, states and actions: {args.output} ({args.output.stat().st_size / 2**20:.1f} MiB)')


if __name__ == '__main__':
    main()
