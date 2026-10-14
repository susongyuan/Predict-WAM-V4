#!/usr/bin/env python3
"""Causally reconstruct policy clocks and contrast physical held-state evidence."""
from pathlib import Path
import argparse
import hashlib
import io
import json
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from harness.wam_event_clock import EventClock
from harness.zhuhai_precision_v2_training import CLOCK_CONFIG
from scripts.progress1_log import log


def sha(data):return hashlib.sha256(data).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary',required=True);parser.add_argument('--seeds',type=int,nargs='+',required=True)
    parser.add_argument('--out',required=True);args=parser.parse_args()
    out=Path(args.out)
    if out.exists():raise ValueError('preserve previous clock diagnostic')
    snapshot=Path(args.summary).read_bytes();summary=json.loads(snapshot);rows=[]
    log('只读重构既有失败/成功的因果闭合时钟并对照物理抓持；不向新策略输入评分GT，不做因果归因。','root')
    for row in summary['episodes']:
        if row['seed'] not in args.seeds:continue
        directory=Path(row['raw_result_path']).parent
        rulebytes=(directory/'rule_evidence.jsonl').read_bytes()
        assert sha(rulebytes)==row['evidence_files']['rule_evidence.jsonl']['sha256']
        rules=[json.loads(line) for line in rulebytes.splitlines()]
        clock=EventClock(**CLOCK_CONFIG);events=[];left_long_empty=0;right_long_empty=0
        for index,rule in enumerate(rules):
            name=f'onboard/observation_{index:06d}.npz';data=(directory/name).read_bytes()
            assert sha(data)==row['evidence_files'][name]['sha256']
            with np.load(io.BytesIO(data),allow_pickle=False) as z:
                feature=clock.update(z['measured_joint_gripper_state'],z['commanded_joint_gripper_state'])
            contact=any(o['touching_gripper'] for o in rule['objects'].values())
            actual_held={key:bool(o['touching_gripper'] and not o['released']
                and min(p[2] for p in o['world_corners_m'])>rule['table_surface_world_m']+.01)
                for key,o in rule['objects'].items()}
            long_empty=[feature[0]*10>=5 and not contact,feature[3]*10>=5 and not contact]
            left_long_empty+=int(long_empty[0]);right_long_empty+=int(long_empty[1])
            if any(long_empty) and (not events or index-events[-1]['observation_step']>=50):
                events.append(dict(observation_step=index,physics_seconds=rule['physics_time_seconds'],
                    policy_clock_features=feature.tolist(),closed_command_seconds=[float(feature[0]*10),float(feature[3]*10)],
                    any_task_object_gripper_contact=contact,actual_lifted_held=actual_held,
                    physical_hold_seconds={k:o['hold_seconds'] for k,o in rule['score']['items'].items()}))
        rows.append(dict(seed=row['seed'],left_long_closed_clock_without_any_object_contact_observations=left_long_empty,
            right_long_closed_clock_without_any_object_contact_observations=right_long_empty,
            examples=events,source_result_sha256=row['raw_result_sha256'],rule_sha256=sha(rulebytes)))
        print(row['seed'],left_long_empty,right_long_empty,flush=True)
    assert {r['seed'] for r in rows}==set(args.seeds)
    value=dict(schema='causal-command-clock-versus-physical-hold-v36',episodes=rows,
        summary_sha256=sha(snapshot),clock_config=CLOCK_CONFIG,clock_sha256=sha((ROOT/'harness/wam_event_clock.py').read_bytes()),
        close_feature_threshold=.5,diagnostic_strict_closed_threshold=.2,privileged_state_used_only_offline=True,
        interpretation='A commanded-closure duration is not a grasp or task-phase confirmation. This contrasts inputs with actual predicates; it does not establish how the network uses the feature or attribute failures causally.')
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as f:json.dump(value,f,indent=2,ensure_ascii=False)


if __name__=='__main__':main()
