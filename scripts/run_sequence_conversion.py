"""Run the median experiment's real-data IK stages at the XML default bases.

Numerical solvers and grid lookup are frozen from the completed median run.
Only real target bases change. Sim keeps original XML+10 degree clipped joints.
"""

import argparse
from datetime import datetime
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from types import SimpleNamespace
import xml.etree.ElementTree as ET
from zoneinfo import ZoneInfo


REPO = Path(__file__).resolve().parents[1]
PREVIOUS = REPO / 'outputs/all_motion_states_margin15_median_20261101'
NUMERICAL_FILES = (
    'scripts/__init__.py', 'scripts/ceres_sequence_ik.py', 'scripts/scipy_ik.py',
    'scripts/trust_constr_sequence_ik.py', 'scripts/optimize_dp_sequences.py',
    'scripts/refine_ceres_sequences.py', 'scripts/yam_reachability.py',
    'abc_minimal/__init__.py', 'abc_minimal/grid_dp.py', 'abc_minimal/eef_pose_grid_gpu.py',
    'abc_minimal/export_mcap.py', 'abc_sim/models/yam_dishrack_base.xml',
    'abc_sim/models/yam_bimanual_empty.xml', 'scripts/dishrack_joint_retarget_calibrations.json.gz',
    'outputs/base_pose_median_20261026/median_base_poses.json',
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def inventory(root):
    return {kind: {p.stem for p in (root / kind / 'sources').glob('*.npz')}
            for kind in ('real', 'sim')}


def setup(args):
    root = args.output
    settings_path = root / 'settings.json'
    if settings_path.exists():
        settings = json.loads(settings_path.read_text())
        for key, value in dict(base_mode=args.base_mode, grid=str(args.grid),
                               action_sources=str(args.action_sources), state_sources=str(args.state_sources)).items():
            if settings[key] != value:
                raise ValueError(f'Existing experiment differs: {key}')
        return settings
    actions, states = inventory(args.action_sources), inventory(args.state_sources)
    if actions != states or any(len(actions[kind]) != count for kind, count in (('real',1261),('sim',3253))):
        raise ValueError('Expected the same 1261 real + 3253 sim sources for both streams')
    frozen = PREVIOUS / 'source_snapshot'
    snapshot = root / 'source_snapshot'
    previous_hashes = json.loads((PREVIOUS / 'source_hashes.json').read_text())
    for name in NUMERICAL_FILES:
        source = frozen / name
        if digest(source) != previous_hashes[name]:
            raise ValueError(f'Previous frozen source changed: {name}')
        target = snapshot / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    # New orchestration and base selection; all numerical modules above are unchanged.
    for name in ('scripts/batch_ceres_ik.py', 'scripts/export_sequence_ik.py', 'scripts/run_sequence_conversion.py',
                 'scripts/prepare_sim_identity.py'):
        target = snapshot / name
        shutil.copy2(REPO / name, target)
    # T7's filesystem does not support symlinks; copy only the referenced meshes/textures.
    for name in ('yam_dishrack_base.xml','yam_bimanual_empty.xml'):
        tree = ET.parse(snapshot/'abc_sim/models'/name)
        compiler = tree.find('compiler').attrib
        for node in tree.iter():
            if 'file' in node.attrib:
                folder = compiler.get('texturedir' if node.tag == 'texture' else 'meshdir','')
                relative = Path('abc_sim/models')/folder/node.attrib['file']
                target = snapshot/relative
                target.parent.mkdir(parents=True,exist_ok=True)
                if not target.exists():
                    shutil.copy2(REPO/relative,target)
    hashes = {str(p.relative_to(snapshot)): digest(p) for p in snapshot.rglob('*') if p.is_file()}
    save(root / 'source_hashes.json', hashes)
    templates = {}
    for stage, source in (
        ('ceres', REPO / 'outputs/all_motion_ceres_top100_median_20261030/config.json'),
        ('equality_margin15', REPO / 'outputs/all_motion_equality_xml_margin15_median_20261101/config.json'),
    ):
        templates[stage] = json.loads(source.read_text())
    save(root / 'solver_templates.json', templates)
    sizes = []
    for stream, source in (('actions',args.action_sources),('states',args.state_sources)):
        sizes.extend((stream,kind,p.name,p.stat().st_size,p.stat().st_mtime_ns)
                     for kind in ('real','sim') for p in sorted((source / kind / 'sources').glob('*.npz')))
    save(root / 'source_inventory.json', sizes)
    estimate = dict(hours=[5,9], required_free_gib=22, method='Real-only share of previous state/action runs; pending XML benchmarks',
                    workers=dict(dp=6,ceres=6,equality_margin15=12), threads_per_worker=1,
                    training_started=False)
    save(root / 'estimate.json', estimate)
    settings = dict(repo=str(REPO), python=str(REPO / '.venv/bin/python'), base_mode=args.base_mode,
                    action_sources=str(args.action_sources), state_sources=str(args.state_sources), grid=str(args.grid),
                    grid_metadata_sha256=digest(args.grid / 'metadata.json'),
                    selected=4514, real=1261, sim=3253, frames_per_stream=5706635,
                    sim_handling='identity_xml_clip10',
                    real_frames_per_stream=sum(r['frames'] for r in json.loads(
                        (REPO/'outputs/all_motion_ceres_top100_median_20261030/summary.json').read_text())['episodes'] if r['kind']=='real'),
                    timing='official causal 30Hz; no interpolation or timeoffset',
                    source_clip_margin_deg=10, distance_limit=.5, candidate_k=100,
                    first_joint_radius_rad=.2, joint_limit_margin_deg=15,
                    check_gradients=False, created=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat())
    save(settings_path, settings)
    return settings


def solver_config(root, settings, stream, stage):
    snapshot = root / 'source_snapshot'
    output = root / stream / stage
    source = root / stream / ('dp' if stage == 'ceres' else 'ceres')
    config = json.loads((root / 'solver_templates.json').read_text())[stage]
    modules = ('optimize_dp_sequences.py','ceres_sequence_ik.py','scipy_ik.py') if stage == 'ceres' else (
        'refine_ceres_sequences.py','trust_constr_sequence_ik.py','ceres_sequence_ik.py','scipy_ik.py')
    model = snapshot / 'abc_sim/models/yam_dishrack_base.xml'
    config.update(source=str(source),output=str(output),input_stream=stream,
                  base_mode=settings['base_mode'],model=str(model),model_sha256=digest(model),
                  source_hashes={str(snapshot/'scripts'/name):digest(snapshot/'scripts'/name) for name in modules})
    config['initialization'] = 'Saved distance<0.5, top100 DP; target bases: ' + settings['base_mode']
    if stream == 'states':
        config['input_states'] = 'Source states clipped to XML bounds expanded 10 degrees before FK.'
        config.pop('input_actions',None)
        config.pop('action_clip',None)
    output.mkdir(parents=True,exist_ok=True)
    (output/'active').mkdir(exist_ok=True)
    path = output/'config.json'
    if path.exists() and json.loads(path.read_text()) != config:
        raise ValueError(f'Solver settings changed: {path}')
    save(path,config)
    if stage == 'equality_margin15':
        previous = json.loads((source/'summary.json').read_text())
        jobs = []
        for report in previous['episodes']:
            path = source/report['kind']/report['episode']/'trajectory.npz'
            jobs.append(dict(kind=report['kind'],key=report['episode'],frames=report['frames'],
                             input=str(path),input_sha256=digest(path)))
        save(output/'jobs.json',jobs)


def count_chunks(root, settings, keys):
    import numpy as np
    # Use exactly the training export's masks, raw joint differences and chunk rules.
    snapshot = root/'source_snapshot'
    spec = importlib.util.spec_from_file_location('frozen_sequence_export',snapshot/'scripts/export_sequence_ik.py')
    export = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(export)
    options = SimpleNamespace(position_tolerance_mm=.1, rotation_tolerance_deg=.1,
                              jump_threshold_deg=30., chunk_length=30)
    totals = {kind:dict(frames=0,retained_frames=0,original_chunks=0,usable_chunks=0) for kind in ('all','real','sim')}
    reports = []
    expected = None
    for kind,key in keys:
        solution = []
        for stream in ('states','actions'):
            path = root/stream/'equality_margin15'/kind/key/'trajectory.npz'
            solution.append(export.load_solution(path,stream,settings['base_mode']))
        for item in solution:
            if expected is None:
                expected = item['camera_from_base']
            np.testing.assert_allclose(item['camera_from_base'],expected,rtol=0,atol=1e-12)
        _, intervals, counts = export.select_frames(*solution,options)
        for group in ('all',kind):
            for field in totals[group]:
                totals[group][field] += counts[field]
        reports.append(dict(episode=key,kind=kind,segments=len(intervals),**counts))
    for values in totals.values():
        values['usable_chunk_percent'] = 100*values['usable_chunks']/max(1,values['original_chunks'])
    save(root/'training_chunks.json',dict(status='complete',completed_sequences=len(keys),
        base_mode=settings['base_mode'],camera_from_base=expected.tolist(),totals=totals,episodes=reports))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--base-mode',choices=('xml',),default='xml')
    parser.add_argument('--grid',type=Path,default=Path('/media/ps/T7/Datasets/ABC-130k/eef_pose_grid_margin10_gpu_20261029'))
    parser.add_argument('--action-sources',type=Path,default=REPO/'outputs/median_base_grid_fallback_bounded_ik_20261026')
    parser.add_argument('--state-sources',type=Path,default=PREVIOUS/'source')
    parser.add_argument('--episodes',nargs='+',help='Pilot IDs; completed results are reused when later running all episodes')
    parser.add_argument('--setup-only',action='store_true')
    parser.add_argument('--dp-workers',type=int,default=6)
    parser.add_argument('--ceres-workers',type=int,default=6)
    parser.add_argument('--equality-workers',type=int,default=12)
    args = parser.parse_args()
    for name in ('output','grid','action_sources','state_sources'):
        setattr(args,name,getattr(args,name).expanduser().resolve())
    root = args.output
    root.mkdir(parents=True,exist_ok=True)
    with (root/'pipeline.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        settings = setup(args)
        snapshot = root/'source_snapshot'
        for name,expected in json.loads((root/'source_hashes.json').read_text()).items():
            if digest(snapshot/name) != expected:
                raise ValueError(f'Frozen source changed: {name}')
        if digest(args.grid/'metadata.json') != settings['grid_metadata_sha256']:
            raise ValueError('Grid/score metadata changed')
        if args.setup_only:
            print(json.dumps(dict(settings=settings,estimate=json.loads((root/'estimate.json').read_text())),indent=2))
            return
        written = sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
        if settings.get('sim_handling') != 'identity_xml_clip10':
            raise ValueError('This run must explicitly record the user-approved sim identity handling')
        if not args.episodes and shutil.disk_usage(root).free + written < 22*2**30:
            raise RuntimeError('Need 22 GiB free for real IK plus sim identity; no full run started')
        keys = [(kind,key) for kind,values in inventory(args.action_sources).items() for key in sorted(values)
                if not args.episodes or any(part in key for part in args.episodes)]
        if not keys:
            raise ValueError('No matching source episodes')
        real_count=sum(kind=='real' for kind,key in keys)
        sim_count=len(keys)-real_count
        status = dict(status='running',pid=os.getpid(),base_mode=args.base_mode,selected=len(keys),
                      scope='pilot' if args.episodes else 'full',started=time.time(),stages={})
        workers_path = root/'workers.json'
        if not workers_path.exists():
            save(workers_path,dict(dp=args.dp_workers,ceres=args.ceres_workers,
                                   equality_margin15=args.equality_workers))
        env = dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
                   PYTHONUNBUFFERED='1',PYTHONFAULTHANDLER='1',PYTHONPATH=str(snapshot))
        prefix = [settings['python'],'-m']
        try:
            for stream,source in (('actions',args.action_sources),('states',args.state_sources)):
                dp = root/stream/'dp'
                commands = [
                    ('dp',prefix+['scripts.batch_ceres_ik','--source',str(source),'--grid',str(args.grid),
                        '--output',str(dp),'--prepare-only','--stream',stream,'--base-mode',args.base_mode,'--kind','real']
                        + (['--episodes',*args.episodes] if args.episodes else [])),
                    ('ceres',prefix+['scripts.optimize_dp_sequences','--source',str(dp),
                        '--output',str(root/stream/'ceres')]),
                    ('equality_margin15',prefix+['scripts.refine_ceres_sequences',
                        '--output',str(root/stream/'equality_margin15')]),
                ]
                for stage,command in commands:
                    workers = int(json.loads(workers_path.read_text())[stage])
                    if workers < 1:
                        raise ValueError('Worker count must be positive')
                    command += ['--workers',str(workers)]
                    if stage != 'dp':
                        solver_config(root,settings,stream,stage)
                    output = root/stream/stage
                    log = root/'logs'/f'{stream}_{stage}.log'
                    log.parent.mkdir(exist_ok=True)
                    status.update(stream=stream,stage=stage,log=str(log),updated=time.time())
                    save(root/'pipeline.json',status)
                    before = time.monotonic()
                    with log.open('a') as handle:
                        process = subprocess.Popen(command,cwd=snapshot,env=env,stdin=subprocess.DEVNULL,
                                                   stdout=handle,stderr=subprocess.STDOUT)
                        status.update(child_pid=process.pid,command=command)
                        save(root/'pipeline.json',status)
                        print(json.dumps(dict(stream=stream,stage=stage,pid=process.pid,log=str(log))),flush=True)
                        code = process.wait()
                    if code:
                        raise RuntimeError(f'{stream}/{stage} exited {code}; see {log}')
                    report = json.loads((output/'summary.json').read_text())
                    if report['completed'] != real_count or report.get('failed',0):
                        raise RuntimeError(f'Incomplete/failed stage: {stream}/{stage}')
                    if not args.episodes and report['frames'] != settings['real_frames_per_stream']:
                        raise RuntimeError(f'Frame count changed: {stream}/{stage}')
                    if len({r['episode'] for r in report['episodes']}) != real_count:
                        raise RuntimeError('Duplicate sequence results')
                    status['stages'][f'{stream}/{stage}'] = dict(output=str(output),seconds=time.monotonic()-before,
                        completed=report['completed'],frames=report['frames'],missing=report.get('missing_candidate_arm_targets',0))
                    save(root/'pipeline.json',status)
            command=prefix+['scripts.prepare_sim_identity','--root',str(root),'--workers','4']
            if args.episodes:
                command+=['--episodes',*args.episodes]
            log=root/'logs/sim_identity.log'
            status.update(stream='both',stage='sim_identity',log=str(log),updated=time.time())
            before=time.monotonic()
            with log.open('a') as handle:
                process=subprocess.Popen(command,cwd=snapshot,env=env,stdin=subprocess.DEVNULL,
                    stdout=handle,stderr=subprocess.STDOUT)
                status.update(child_pid=process.pid,command=command)
                save(root/'pipeline.json',status)
                code=process.wait()
            if code:
                raise RuntimeError(f'Sim identity export failed: {log}')
            report=json.loads((root/'sim_identity/summary.json').read_text())
            if report['completed'] != sim_count or report['failed']:
                raise RuntimeError('Incomplete sim identity results')
            status['stages']['sim_identity']=dict(completed=sim_count,frames=report['frames'],seconds=time.monotonic()-before)
            count_chunks(root,settings,keys)
            status.update(status='complete',finished=time.time(),updated=time.time())
        except Exception as error:
            status.update(status='failed',error=str(error),updated=time.time())
            save(root/'pipeline.json',status)
            raise
        save(root/'pipeline.json',status)
        print(json.dumps(status,indent=2),flush=True)


if __name__ == '__main__':
    main()
