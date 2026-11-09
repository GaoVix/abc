"""Replace this node's five random-seed PD-only runs with all-limit ablations.

Leave existing policy seeds 1..5 running, including on other nodes. Copy the
five local random-seed commands/environments before stopping them, map them
to policy seeds 1..5, and keep their GPU settings. Save a plan before SIGTERM.
Only the two explicitly named XML-reference checkpoints/experiment prefixes
are eligible. Run with --apply to execute; otherwise only print the plan.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time
from zoneinfo import ZoneInfo


KEEP = (1, 2, 3, 4, 5)
STOP = (186109139, 2107183402, 636244850, 1004037165, 1548170658)
MODELS = {
    'old_mink': ('dit_joint_retargeted_official_timing_20261022_125331',
                 'eval_old_mink_official_timing_xml_direct_strict_100k_unclamped_target_pseed'),
    'new_xml': ('dit_xml_default_real_ik_raw_sim_20261103_112255',
                'eval_xml_default_real_ik_raw_sim_direct_strict_100k_highres_unclamped_target_pseed'),
}


def option(argv, name, default=None):
    for index, word in enumerate(argv):
        if word == name:
            return argv[index + 1] if index + 1 < len(argv) else default
        if word.startswith(name + '='):
            return word.split('=', 1)[1]
    return default


def replace_option(argv, name, value):
    result = []
    index = 0
    while index < len(argv):
        word = argv[index]
        if word == name:
            index += 2
            continue
        if not word.startswith(name + '='):
            result.append(word)
        index += 1
    return result + [name, str(value)]


def identity(proc):
    fields = (proc / 'stat').read_text().rpartition(')')[2].split()
    return fields[19], fields[0]


@dataclass
class Job:
    pid: int
    proc: Path
    start: str
    argv: list[str]

    def alive(self):
        try:
            start, state = identity(self.proc)
            return start == self.start and state != 'Z'
        except (OSError, IndexError):
            return False


def collect_jobs(root, model=None):
    jobs = defaultdict(list)
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            start, state = identity(proc)
            argv = [s.decode() for s in (proc / 'cmdline').read_bytes().split(b'\0') if s]
            if state == 'Z' or not any(Path(word).name == 'eval_policy.py' for word in argv):
                continue
            if '--disable-arm-control-limits' not in argv or '--no-disable-arm-control-limits' in argv:
                continue
            if '--disable-all-arm-limits' in argv:
                continue
            checkpoint = Path(option(argv, '--checkpoint', ''))
            output = Path(option(argv, '--output-dir', ''))
            seed = int(option(argv, '--policy-seed', '-1'))
            for label, (model_dir, prefix) in MODELS.items():
                if (checkpoint == root / model_dir / '100000.pt'
                        and output.parent == root
                        and output.name.startswith(f'{prefix}{seed}_')):
                    jobs[label, seed].append(Job(int(proc.name), proc, start, argv))
        except (OSError, UnicodeError, ValueError, IndexError):
            continue
    if model is None:
        labels = [label for label in MODELS if any(jobs[label, seed] for seed in STOP)]
        if len(labels) != 1:
            raise RuntimeError('Expected one model with five local random-seed runs; '
                               f'found {labels}. Nothing was stopped; select --model if needed.')
        model = labels[0]
    missing = {f'{model}/pseed{seed}': len(jobs[model, seed])
               for seed in STOP if len(jobs[model, seed]) != 1}
    if missing:
        raise RuntimeError('Expected exactly five live random-seed PD-only runs on this node; '
                           'nothing was stopped. Missing/duplicate counts: ' + json.dumps(missing))
    return {(model, seed): jobs[model, seed][0] for seed in STOP}


def prepare(jobs, batch):
    launches = []
    for label in sorted({key[0] for key in jobs}):
        baseline = None
        for seed, stopped_seed in zip(KEEP, STOP):
            source = jobs[label, stopped_seed]
            if '--deterministic-inference' not in source.argv or '--no-fast-inference' not in source.argv:
                raise ValueError(f'{label}/pseed{stopped_seed} is not the requested deterministic run')
            if option(source.argv, '--parallel-worlds', '0') != '0':
                raise ValueError('Expected sequential CPU MuJoCo physics')
            gpu = option(source.argv, '--gpu-id')
            if gpu is None:
                raise ValueError(f'No GPU ID on stopped run {source.pid}')
            normalized = source.argv
            for name in ('--policy-seed', '--output-dir', '--gpu-id'):
                normalized = replace_option(normalized, name, '<varies>')
            if baseline is None:
                baseline = normalized
            elif normalized != baseline:
                raise ValueError('The five local source commands differ beyond policy seed/output/GPU; '
                                 'nothing was stopped. Inspect their settings before pairing.')
            output = batch / f'{label}_no_arm_limits_pseed{seed}'
            argv = [word for word in source.argv
                    if word not in ('--disable-arm-control-limits', '--no-disable-all-arm-limits')]
            argv = replace_option(argv, '--policy-seed', seed)
            argv = replace_option(argv, '--output-dir', output) + ['--disable-all-arm-limits']
            launches.append(dict(model=label, policy_seed=seed, source_policy_seed=stopped_seed,
                                 source_pid=source.pid, replaced_pid=source.pid, source_argv=source.argv,
                                 gpu=gpu, output=str(output), argv=argv))
    return launches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=Path('/mnt/afs/L202500703/abc_storage/output'))
    parser.add_argument('--model', choices=tuple(MODELS), default=None,
                        help='Default: identify the one model with five local random-seed runs.')
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    from abc_minimal.config import SimEvalConfig
    if 'disable_all_arm_limits' not in SimEvalConfig.__dataclass_fields__:
        raise RuntimeError('Sync the eval --disable-all-arm-limits implementation first')
    root = args.output_root.resolve()
    jobs = collect_jobs(root, args.model)
    placeholder = root / 'eval_arm_limits_switch_NEW'
    launches = prepare(jobs, placeholder)
    plan = dict(keep_policy_seeds_untouched=list(KEEP),
                stop=[dict(model=label, policy_seed=seed, pid=job.pid)
                      for (label, seed), job in jobs.items()], launch=launches, status='planned')
    print('Policy seeds 1..5 are left untouched; only this node\'s five random-seed jobs are replaced.', flush=True)
    for item in plan['stop']:
        print(f"PLAN STOP {item['model']} pseed={item['policy_seed']} PID={item['pid']}", flush=True)
    for item in launches:
        print(f"PLAN START {item['model']} pseed={item['policy_seed']} GPU={item['gpu']} copy_PID={item['source_pid']}", flush=True)
    if not args.apply:
        return

    # Capture everything and validate identities before stopping any process.
    runtime = []
    for item in launches:
        source = jobs[item['model'], item['source_policy_seed']]
        env = dict(entry.decode().split('=', 1) for entry in (source.proc / 'environ').read_bytes().split(b'\0') if entry)
        runtime.append((env, os.readlink(source.proc / 'cwd')))
    if not all(job.alive() for job in jobs.values()):
        raise RuntimeError('A source process exited while planning; nothing was stopped. Run again.')
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')
    batch = Path(tempfile.mkdtemp(prefix='eval_arm_limits_switch_' + stamp + '_', dir=root))
    plan['launch'] = prepare(jobs, batch)
    plan_path = batch / 'switch_plan.json'

    def save():
        plan_path.write_text(json.dumps(plan, indent=2) + '\n')

    save()
    try:
        for (label, seed), job in jobs.items():
            if job.alive():
                os.kill(job.pid, signal.SIGTERM)
                print(f'STOP {label} pseed={seed} PID={job.pid}', flush=True)
        deadline = time.monotonic() + 30
        while any(job.alive() for job in jobs.values()):
            if time.monotonic() >= deadline:
                raise RuntimeError('Some stopped processes have not exited; no new jobs launched')
            time.sleep(.2)
        plan['status'] = 'stopped_random_seeds'
        save()
        for item, (env, cwd) in zip(plan['launch'], runtime):
            output = Path(item['output'])
            output.mkdir()
            with (output / 'eval.log').open('x') as log:
                proc = subprocess.Popen(item['argv'], cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            item['pid'] = proc.pid
            (output / 'eval.pid').write_text(str(proc.pid) + '\n')
            save()
            print(f"START {item['model']} pseed={item['policy_seed']} GPU={item['gpu']} PID={proc.pid} LOG={output / 'eval.log'}", flush=True)
        plan['status'] = 'launched'
        save()
    except BaseException as exc:
        plan['status'], plan['error'] = 'interrupted_or_failed', str(exc)
        save()
        raise
    print(f'BATCH={batch}\nPLAN={plan_path}', flush=True)


if __name__ == '__main__':
    main()
