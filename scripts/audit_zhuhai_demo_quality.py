#!/usr/bin/env python3
"""Read-only audit of completed demonstrations; active partials are excluded."""
import argparse
from collections import Counter
import io
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def audit(source_root, output):
    import h5py
    import numpy as np
    from PIL import Image, ImageDraw
    from scripts.collect_isaac_zhuhai import _success_rows, sha256, write_once
    output.mkdir(parents=True, exist_ok=True)
    report = dict(schema='zhuhai_demo_quality_readonly_v1', checked_unix=time.time(),
        source_root=str(source_root.resolve()), completed_sources_only=True,
        full_HDF_hash_scope='first/middle/last successful episode of each task',
        per_frame_rule_scope='first/middle/last successful episode of each task',
        all_success_control_arrays_checked=True, new_recording=False, tasks={})
    boards=[]
    for task in ('zhuhai_task1', 'zhuhai_task2'):
        manifest_path=source_root/task/'source/manifest.json'
        raw=manifest_path.read_bytes(); manifest=json.loads(raw)
        # Full strict retained-evidence/statistics checks, without hashing active files.
        rows=_success_rows(manifest,verify_files=False)
        selected={0,len(rows)//2,len(rows)-1}
        episodes=[]; phase_orders=Counter(); failures=Counter()
        for attempt in manifest['attempts']:
            if attempt['status'] not in ('success','running'):
                failures[attempt.get('failure_reason') or attempt.get('failure') or attempt['status']]+=1
        for index,row in enumerate(rows):
            receipt_path=Path(row['expert_receipt_path'])
            if sha256(receipt_path)!=row['expert_receipt_sha256']:
                raise ValueError('Expert receipt digest changed')
            receipt=json.loads(receipt_path.read_text())
            phases=[r for r in receipt['trace'] if r.get('samples',0)>0]
            phase_orders[tuple(r['phase'] for r in phases)]+=1
            hold_seconds=sum(r.get('duration_seconds',r['samples']*.1) for r in phases if 'hold' in r['phase'])
            lid_seconds=sum(r.get('duration_seconds',r['samples']*.1) for r in phases if r['phase'].startswith('lid_'))
            with h5py.File(row['data_path'],'r') as f:
                state=f['measured_state'][:]; command=f['target_command'][:]; clock=f['physics_time'][:]
                n=len(state)
                if state.shape!=(n,16) or command.shape!=(n,16) or n!=row['observation_count']:
                    raise ValueError('Native16 shape/observation count mismatch')
                if not np.isfinite(state).all() or not np.isfinite(command).all() or not np.allclose(np.diff(clock),.1,atol=1e-7,rtol=0):
                    raise ValueError('Nonfinite control or nonuniform physics time')
                actions=np.concatenate([f['action/'+name][:] for name in ('left_arm_joint_states','left_ee_joint_states','right_arm_joint_states','right_ee_joint_states')],axis=1)
                states=np.concatenate([f['state/'+name][:] for name in ('left_arm_joint_states','left_ee_joint_states','right_arm_joint_states','right_ee_joint_states')],axis=1)
                if not np.array_equal(actions,command[1:]) or not np.array_equal(states,state[:-1]):
                    raise ValueError('HDF state/action interval alignment mismatch')
                arms=[*range(7),*range(8,15)]
                delta=np.diff(command,axis=0); motion=np.linalg.norm(delta[:,arms],axis=1)
                one_arm=np.logical_xor(np.linalg.norm(delta[:,:7],axis=1)>1e-5,np.linalg.norm(delta[:,8:15],axis=1)>1e-5)
                cameras={name:len(f['vision/'+name+'/colors']) for name in ('cam_head','cam_left_wrist','cam_right_wrist')}
                if any(count!=n-1 for count in cameras.values()):raise ValueError('RGB/action lengths mismatch')
                evidence_check=None
                if index in selected:
                    if sha256(row['data_path'])!=row['data_sha256']:raise ValueError('Sampled HDF digest changed')
                    unsafe=0; closed_after_open={}; score_decreases=0; prior=0
                    for raw_evidence in f['rule_evidence']:
                        evidence=json.loads(raw_evidence); safety=evidence['physics_safety']
                        unsafe+=not safety['passed']
                        score=evidence['score']; score_decreases+=score['base_score']<prior;prior=score['base_score']
                        for lid, lid_score in score.get('lids',{}).items():
                            if lid_score['opened_awarded'] and not lid_score['open_now']:
                                closed_after_open[lid]=closed_after_open.get(lid,0)+1
                    evidence_check=dict(unsafe_observations=unsafe,lid_closed_after_open_award_observations=closed_after_open,
                        score_decreases=score_decreases,observations=n)
                if index==0:
                    # Existing head observations, evenly spaced including terminal; no simulator/video.
                    board=Image.new('RGB',(320*4,260*2),'white');draw=ImageDraw.Draw(board)
                    for j,frame_index in enumerate(np.linspace(0,n-1,8,dtype=int)):
                        payload=(f['terminal_observation/vision/cam_head/colors'][:] if frame_index==n-1 else f['vision/cam_head/colors'][frame_index])
                        frame=Image.open(io.BytesIO(payload.tobytes())).convert('RGB')
                        x=(j%4)*320;y=(j//4)*260
                        board.paste(frame,(x,y));draw.text((x+5,y+242),f'{task} seed {row["seed"]} t={clock[frame_index]:.1f}s',fill='black')
                    board_path=output/(task+'_existing_head_contact_sheet.png');board.save(board_path);boards.append(str(board_path))
                episodes.append(dict(seed=row['seed'],frames=n-1,physics_seconds=float(clock[-1]-clock[0]),
                    wall_seconds=row['finished_unix']-row['started_unix'],physical_hold_seconds=hold_seconds,lid_phase_seconds=lid_seconds,
                    static_arm_target_fraction=float(np.mean(motion<1e-5)),single_arm_moving_fraction=float(np.mean(one_arm)),
                    target_joint_velocity_abs_max_rad_s=float(np.max(np.abs(delta[:,arms]))/.1),
                    measured_minus_target_arm_rms_rad=float(np.sqrt(np.mean((state[:,arms]-command[:,arms])**2))),
                    camera_lengths=cameras,selected_full_hash_verified=index in selected,per_frame_rule_check=evidence_check,
                    final_lid_degrees={k:float(abs(v)*180/np.pi) for k,v in row['final_rule_evidence'].get('lid_angles_rad',{}).items()}))
        def stats(key):
            values=np.array([r[key] for r in episodes]);return dict(min=float(values.min()),median=float(np.median(values)),max=float(values.max()),mean=float(values.mean()))
        report['tasks'][task]=dict(manifest_snapshot_sha256=__import__('hashlib').sha256(raw).hexdigest(),
            manifest_status=manifest['status'],successful_episodes=len(rows),failed_attempts=sum(failures.values()),running_attempts=sum(r['status']=='running' for r in manifest['attempts']),
            failure_reasons=dict(failures),all_admitted_final_full_success=True,all_admitted_initial_zero_score=True,
            all_retained_physics_safety_pass=True,physical_contact_success_scope='declared rules and checked forbidden contact categories; not every contact forbidden',
            physical_seconds=stats('physics_seconds'),wall_seconds=stats('wall_seconds'),static_arm_target_fraction=stats('static_arm_target_fraction'),
            single_arm_moving_fraction=stats('single_arm_moving_fraction'),hold_seconds=stats('physical_hold_seconds'),lid_seconds=stats('lid_phase_seconds'),
            max_target_joint_velocity_rad_s=max(r['target_joint_velocity_abs_max_rad_s'] for r in episodes),
            distinct_phase_orders=len(phase_orders),phase_orders=[dict(phases=list(order),episodes=count) for order,count in phase_orders.most_common()],episodes=episodes)
    report['contact_sheets']=boards
    write_once(output/'quality_audit.json',report)
    print(json.dumps({task:{k:v for k,v in data.items() if k not in ('episodes','phase_orders')} for task,data in report['tasks'].items()},ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();audit(args.source_root,args.out)
