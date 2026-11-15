"""Keep original sim joints at XML bases, applying only source clipping.

This is an identity conversion, not an IK result. The final training exporter
uses the same state/action, pose-validity and jump-boundary rules as for real.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
from pathlib import Path
import time
import traceback

import mujoco
import numpy as np

from scripts.scipy_ik import pose_error


SIDES = ('left','right')
COLUMNS = np.array([[0,1,2,3,4,5],[7,8,9,10,11,12]])
ROOT = MODEL = DATA = LIMITS = CAMERA = BASES = MODEL_HASH = None


def save_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def pose(entity):
    result=np.eye(4)
    result[:3,:3],result[:3,3]=entity.xmat.reshape(3,3),entity.xpos
    return result


def initialize(root):
    global ROOT,MODEL,DATA,LIMITS,CAMERA,BASES,MODEL_HASH
    ROOT=Path(root)
    path=ROOT/'source_snapshot/abc_sim/models/yam_dishrack_base.xml'
    MODEL_HASH=hashlib.sha256(path.read_bytes()).hexdigest()
    MODEL=mujoco.MjModel.from_xml_path(str(path))
    DATA=mujoco.MjData(MODEL)
    mujoco.mj_forward(MODEL,DATA)
    CAMERA=pose(DATA.camera('top'))
    CAMERA[:3,:3] @= np.diag([1.,-1.,-1.])
    BASES=np.array([np.linalg.inv(CAMERA)@pose(DATA.body(f'{side}_arm')) for side in SIDES])
    joints=[[MODEL.joint(f'{side}_joint{i}').id for i in range(1,7)] for side in SIDES]
    LIMITS=MODEL.jnt_range[joints]+np.deg2rad(10)*np.array([-1.,1.])


def prepare(record):
    key,action_path,state_path=record
    started=time.monotonic()
    try:
        signature=hashlib.sha256(json.dumps([MODEL_HASH,10.,[(str(p),p.stat().st_size,p.stat().st_mtime_ns)
            for p in (action_path,state_path)]],sort_keys=True).encode()).hexdigest()
        frames=None
        reports=[]
        for stream,path in (('actions',action_path),('states',state_path)):
            output=ROOT/stream/'equality_margin15/sim'/key
            previous=output/'result.json'
            if previous.exists():
                report=json.loads(previous.read_text())
                if report.get('solver')=='identity' and report.get('signature')==signature and (output/'trajectory.npz').exists():
                    frames=report['frames'] if frames is None else frames
                    assert frames==report['frames']
                    reports.append(report)
                    continue
                raise ValueError('Existing final sim result has different method/source; archive it first')
            with np.load(path,allow_pickle=False) as saved:
                original=saved[stream].astype(np.float64)
                indices=saved['frame_indices'] if 'frame_indices' in saved else np.arange(len(original))
                sample_errors=[]
                if stream=='actions' and 'targets_camera' in saved:
                    targets=saved['targets_camera']
                    for frame in sorted({0,len(original)//2,len(original)-1}):
                        for arm,side in enumerate(SIDES):
                            joints=[MODEL.joint(f'{side}_joint{i}').id for i in range(1,7)]
                            error=pose_error(original[frame,COLUMNS[arm]],MODEL,DATA,joints,
                                MODEL.site(f'{side}_tcp_site').id,CAMERA@targets[frame,arm])
                            sample_errors.append([1000*np.linalg.norm(error[:3]),np.rad2deg(np.linalg.norm(error[3:]))])
                    if np.max(sample_errors) > 1e-6:
                        raise ValueError('Source sim FK does not match the default XML model/camera')
            if not np.isfinite(original).all():
                raise ValueError('Non-finite original sim source')
            np.testing.assert_array_equal(indices,np.arange(len(original)))
            frames=len(original) if frames is None else frames
            assert frames==len(original)
            clipped=original.copy()
            clipped[:,COLUMNS]=np.clip(original[:,COLUMNS],LIMITS[...,0],LIMITS[...,1])
            q=clipped[:,COLUMNS]
            changed=(clipped[:,COLUMNS]!=original[:,COLUMNS]).sum(axis=0)
            valid=np.ones((frames,2),dtype=bool)
            output.mkdir(parents=True,exist_ok=True)
            temporary=output/'trajectory.partial.npz'
            np.savez_compressed(temporary,**{stream:clipped,f'optimized_{stream}':clipped},q=q,
                frame_indices=indices,errors=np.zeros((frames,2,2)),valid_mask=valid,optimized_mask=valid,
                camera_from_base=BASES,world_from_camera=CAMERA,base_poses_world=CAMERA@BASES,
                base_mode='xml',solver='identity',identity_source_verified=True,source_clip_margin_deg=10.,
                input_stream=stream,episode=key,source_path=str(path),source_signature=signature,
                model_path=str(ROOT/'source_snapshot/abc_sim/models/yam_dishrack_base.xml'),
                source_model_sha256=MODEL_HASH,target_model_sha256=MODEL_HASH,
                clipped_joint_counts=changed,timing_mode='official_causal_no_interpolation_no_timeoffset',
                pose_error_definition='Zero by identical source/target XML kinematics and bases after clipping; no IK performed')
            temporary.replace(output/'trajectory.npz')
            steps=np.rad2deg(abs(np.diff(q,axis=0)))
            report=dict(episode=key,kind='sim',status='complete',solver='identity',signature=signature,
                frames=frames,arm_targets=2*frames,valid_arm_targets=2*frames,optimized_arm_targets=2*frames,
                missing_candidate_arm_targets=0,passed=2*frames,error_sum_mm_deg=[0.,0.],error_max_mm_deg=[0.,0.],
                max_joint_step_deg=float(steps.max(initial=0)),jump_arm_frame_pairs=int((steps.max(axis=-1)>30).sum()),
                frame_iterations=0,clipped_joint_values=int(changed.sum()),seconds=time.monotonic()-started,
                sampled_original_action_fk_max_mm_deg=np.max(sample_errors,axis=0).tolist() if sample_errors else None,
                identity_reason='Same XML bases, camera and kinematics; preserve original clipped sim joints')
            save_json(previous,report)
            reports.append(report)
        return dict(episode=key,kind='sim',status='complete',frames=frames,streams=reports,seconds=time.monotonic()-started)
    except Exception as error:
        return dict(episode=key,kind='sim',status='failed',error=str(error),traceback=traceback.format_exc())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--episodes',nargs='+')
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    root=args.root.resolve()
    settings=json.loads((root/'settings.json').read_text())
    sources=[{p.stem:p for p in (Path(settings[name])/'sim/sources').glob('*.npz')}
             for name in ('action_sources','state_sources')]
    assert sources[0].keys()==sources[1].keys()
    keys=sorted(key for key in sources[0] if not args.episodes or any(part in key for part in args.episodes))
    output=root/'sim_identity'
    output.mkdir(exist_ok=True)
    started=time.monotonic()
    reports=[]
    progress=dict(status='running',selected=len(keys),completed=0,failed=0,frames=0,seconds=0.,solver='identity')
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn'),
                             initializer=initialize,initargs=(str(root),)) as pool:
        for future in as_completed([pool.submit(prepare,(key,sources[0][key],sources[1][key])) for key in keys]):
            reports.append(future.result())
            progress=dict(status='running',selected=len(keys),completed=len(reports),
                failed=sum(r['status']!='complete' for r in reports),frames=sum(r.get('frames',0) for r in reports),
                seconds=time.monotonic()-started,solver='identity')
            save_json(output/'progress.json',progress)
    progress['status']='complete' if not progress['failed'] else 'failed'
    save_json(output/'progress.json',progress)
    save_json(output/'summary.json',dict(**progress,episodes=reports))
    print(json.dumps(progress),flush=True)
    if progress['failed']:
        raise SystemExit(1)


if __name__=='__main__':
    main()
