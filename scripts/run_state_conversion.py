"""Run the saved action pipeline on state rows, ending with XML+15 degree IK."""

import argparse
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from zoneinfo import ZoneInfo


PILOT = ('real_episode_a8a591d9-e7fe-41b9-8874-d8f4270e80d7',
         'sim_episode_019e037e-af67-7837-a178-c70331b678ae')


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def solver_config(root, settings, stage, source, output):
    template = settings['action_ceres'] if stage == 'ceres' else settings['action_equality']
    config = json.loads((Path(template) / 'config.json').read_text())
    modules = ('optimize_dp_sequences.py', 'ceres_sequence_ik.py', 'scipy_ik.py') if stage == 'ceres' else (
        'refine_ceres_sequences.py', 'trust_constr_sequence_ik.py', 'ceres_sequence_ik.py', 'scipy_ik.py')
    snapshot = root / 'source_snapshot'
    model = snapshot / 'abc_sim/models/yam_dishrack_base.xml'
    config.update(source=str(source), output=str(output), input_stream='states',
                  model=str(model), model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
                  source_hashes={str(snapshot / 'scripts' / name): hashlib.sha256(
                      (snapshot / 'scripts' / name).read_bytes()).hexdigest() for name in modules},
                  created=time.time())
    config['input_states'] = config.pop('input_actions', config.pop('action_clip', ''))
    config['input_states'] = config['input_states'].replace('actions', 'states')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'active').mkdir(exist_ok=True)
    if not (output / 'config.json').exists():
        save(output / 'config.json', config)
    else:
        previous = json.loads((output / 'config.json').read_text())
        for key in ('source', 'input_stream', 'model_sha256', 'source_hashes', 'weights', 'maxiter'):
            assert config[key] == previous[key], key
    return config


def copy_pilot(root, stage, output):
    for kind in ('real', 'sim'):
        for folder in (root / 'pilot' / stage / kind).glob('*'):
            target = output / kind / folder.name
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(folder, target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--pilot', action='store_true')
    args = parser.parse_args()
    root = args.root.absolute()
    settings = json.loads((root / 'settings.json').read_text())
    snapshot = root / 'source_snapshot'
    for name, expected in json.loads((root / 'source_hashes.json').read_text()).items():
        assert hashlib.sha256((snapshot / name).read_bytes()).hexdigest() == expected, name
    expected_count = len(PILOT) if args.pilot else 4514
    source, dp = root / 'source', root / ('pilot/dp' if args.pilot else 'dp')
    ceres = root / 'pilot/ceres' if args.pilot else Path(settings['ceres_output'])
    equality = root / 'pilot/equality_margin15' if args.pilot else Path(settings['equality_output'])
    status_path = root / ('pilot_pipeline.json' if args.pilot else 'pipeline.json')
    status = dict(status='running', input_stream='states', selected=expected_count, stage='sources',
                  pid=os.getpid(), started=time.time(), stages={},
                  started_local=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat())
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               PYTHONUNBUFFERED='1', PYTHONFAULTHANDLER='1', PYTHONPATH=str(snapshot))
    python = str(Path(settings['repo']) / '.venv/bin/python')
    prefix = [python, '-m']
    source_command = prefix + ['scripts.prepare_state_sources', '--reference', settings['action_dp'],
        '--manifest', str(snapshot / 'scripts/dishrack_joint_retarget_calibrations.json.gz'),
        '--raw-root', settings['raw_root'], '--sim-root', settings['sim_root'],
        '--output', str(source), '--workers', '2' if args.pilot else '6']
    if args.pilot:
        source_command += ['--episodes', *PILOT]
    stages = [
        ('sources', source, source_command),
        ('dp', dp, prefix + ['scripts.batch_ceres_ik', '--source', str(source), '--output', str(dp),
                            '--prepare-only', '--stream', 'states', '--grid', settings['grid'],
                            '--workers', '2' if args.pilot else '6']),
        ('ceres', ceres, prefix + ['scripts.optimize_dp_sequences', '--source', str(dp),
                                  '--output', str(ceres), '--workers', '2' if args.pilot else '6']),
        ('equality_margin15', equality, prefix + ['scripts.refine_ceres_sequences',
                                                '--output', str(equality), '--workers', '2' if args.pilot else '12']),
    ]
    with (root / ('pilot.lock' if args.pilot else 'pipeline.lock')).open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            for stage, output, command in stages:
                if not args.pilot and stage != 'sources':
                    copy_pilot(root, stage, output)
                if stage in ('ceres', 'equality_margin15'):
                    solver_config(root, settings, 'ceres' if stage == 'ceres' else 'equality',
                                  dp if stage == 'ceres' else ceres, output)
                if stage == 'equality_margin15':
                    previous = json.loads((ceres / 'summary.json').read_text())
                    jobs = [dict(kind=r['kind'], key=r['episode'], frames=r['frames'],
                                 input=str(ceres / r['kind'] / r['episode'] / 'trajectory.npz'),
                                 input_sha256=hashlib.sha256((ceres / r['kind'] / r['episode'] / 'trajectory.npz').read_bytes()).hexdigest())
                            for r in previous['episodes']]
                    save(output / 'jobs.json', jobs)
                log = Path('/tmp') / f"abc_states_{'pilot_' if args.pilot else ''}{stage}_20261101.log"
                status.update(stage=stage, updated=time.time(), log=str(log))
                save(status_path, status)
                started = time.monotonic()
                with log.open('a') as stream:
                    process = subprocess.Popen(command, cwd=snapshot, env=env, stdout=stream,
                                               stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
                    status.update(child_pid=process.pid, command=command)
                    save(status_path, status)
                    print(json.dumps(dict(stage=stage, pid=process.pid, log=str(log))), flush=True)
                    code = process.wait()
                if code:
                    raise RuntimeError(f'{stage} exited with code {code}; see {log}')
                summary = json.loads((output / 'summary.json').read_text())
                assert summary['completed'] == expected_count and summary.get('failed', 0) == 0
                assert len({r['episode'] for r in summary['episodes']}) == expected_count
                status['stages'][stage] = dict(output=str(output), seconds=time.monotonic() - started,
                                               completed=summary['completed'], frames=summary['frames'])
                save(status_path, status)
            status.update(status='complete', finished=time.time(), updated=time.time())
        except Exception as error:
            status.update(status='failed', error=str(error), updated=time.time())
            save(status_path, status)
            raise
        save(status_path, status)
        print(json.dumps(status, indent=2), flush=True)


if __name__ == '__main__':
    main()
