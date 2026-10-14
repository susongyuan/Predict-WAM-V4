#!/usr/bin/env python3
"""Verify real USD task-2 authoring without SimulationApp, rendering, or GPU.

This does not verify live PhysX settling, robot expert success, or RGB occlusion.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pxr import Usd,UsdGeom,UsdSemantics
from harness.isaac_zhuhai_task2 import build,RuntimeJudge
from harness.isaac_zhuhai_camera import IsaacZhuhaiCameraRig,projection_report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--seeds',type=int,default=3)
    parser.add_argument('--out',type=Path,default=ROOT/'outputs/validation/zhuhai_task2_20261008/isaac_cpu_recheck.json')
    args=parser.parse_args();runs=[]
    for randomized in [False,True]:
        for seed in range(101001,101001+args.seeds):
            stage=Usd.Stage.CreateInMemory();manifest=build(stage,seed=seed,randomized=randomized)
            rig=IsaacZhuhaiCameraRig(stage,seed=seed,randomized=randomized);camera=rig.build()
            projection=projection_report(manifest['geometry_world_m'],rig.head_world_matrix)
            judge=RuntimeJudge(manifest,robot_prim_paths=['/World/Robot'],gripper_prim_paths=['/World/Robot/fingers'])
            bodies={key:{'world_matrix':spec['initial_world_matrix'],'linear_velocity_m_s':[0,0,0],
                'angular_velocity_rad_s':[0,0,0]} for key,spec in manifest['objects'].items()}
            initial=judge.observe(0,bodies,[])
            checks={'four_objects':set(manifest['objects'])=={'bowl','cup','plate','spoon'},
                'score_zero':initial['score']['base_score']==0,
                'full_geometry_projected':projection['all_geometry_in_frame'],
                'three_cameras':sum(prim.IsA(UsdGeom.Camera) for prim in stage.Traverse())==3,
                'original_convex_decomposition':{key:spec['collision_hull_count'] for key,spec in manifest['objects'].items()}
                    =={'bowl':24,'cup':30,'plate':22,'spoon':11},
                'semantic_labels':all(stage.GetPrimAtPath(spec['visual_path']).HasAPI(UsdSemantics.LabelsAPI,'class')
                    for spec in manifest['objects'].values())}
            runs.append(dict(seed=seed,randomized=randomized,checks=checks,passed=all(checks.values()),
                manifest=manifest,camera=camera,initial_rule_evidence=initial,head_projection=projection))
    files=['harness/isaac_zhuhai_task2.py','harness/isaac_zhuhai_camera.py','harness/zhuhai_task2_rules.py']
    report=dict(scope='USD CPU authoring only; live PhysX, RGB visibility and robot expert pending',
        all_passed=all(run['passed'] for run in runs),runs=runs,gpu_used=False,record_video=False,
        source_hashes={filename:hashlib.sha256((ROOT/filename).read_bytes()).hexdigest() for filename in files})
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(all_passed=report['all_passed'],run_count=len(runs),out=str(args.out))))
    return 0 if report['all_passed'] else 2

if __name__=='__main__':raise SystemExit(main())
