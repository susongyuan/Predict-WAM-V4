#!/usr/bin/env python3
"""Actual PhysX CPU tableware smoke: no rendering, robot planning, or GPU use."""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'third_party/RoboTwin')]
os.chdir(ROOT/'third_party/RoboTwin')
import numpy as np
import sapien
from envs.zhuhai_task2 import zhuhai_task2
from envs.zhuhai_camera import declared_projection_report,head_camera_transform


def run(seed,randomized):
    np.random.seed(seed)
    env=zhuhai_task2()
    env.scene=sapien.Scene([sapien.physx.PhysxCpuSystem()])
    env.scene.set_timestep(.004)
    env.scene.default_physical_material=env.scene.create_physical_material(.6,.48,0)
    env.table_z_bias=float(np.random.uniform(-.008,0)) if randomized else 0.
    env.zhuhai_cpu_only=True
    env.zhuhai_track,env.zhuhai_mapping='life','body_text'
    env.zhuhai_randomized=randomized
    env.zhuhai_seed=seed;env.zhuhai_source_seed=seed%100
    env.prohibited_area=[]
    env.robot=SimpleNamespace(gripper_name=[])
    env.random_light=False
    env.judge=None
    env._box('table',[0,0,.72+env.table_z_bias],[.65,.45,.02],(.5,.5,.5))
    env.load_actors()
    for _ in range(2500):env.scene.step()
    env._start_observer()
    for _ in range(250):env.scene.step()
    evidence=env.rule_state()
    geometry={key:state.corners for key,state in env.last_physical_states.items()}
    for entity in env.bin_parts:
        body=entity.find_component_by_type(sapien.physx.PhysxRigidStaticComponent)
        half=np.asarray(body.collision_shapes[0].half_size)
        geometry[entity.name+str(entity.per_scene_id)]=[entity.get_pose().p+half*np.array(sign) for sign in itertools.product([-1,1],repeat=3)]
    coverage=declared_projection_report(geometry,head_camera_transform(env.table_z_bias))
    table_contacts={item['object'] for item in evidence['contacts'] if item['other_body']=='table'}
    speed={key:float(np.linalg.norm(value.linear_velocity)) for key,value in env.last_physical_states.items()}
    checks={'objects_complete':set(evidence['objects'])=={'bowl','cup','plate','spoon'},
        'score_zero':evidence['score']['base_score']==0,'no_initial_success':not evidence['score']['full_task_success'],
        'all_on_table_contact':table_contacts==set(env.objects),'all_stable':max(speed.values())<.015,
        'cup_upright':env.last_physical_states['cup'].upright_cosine>.9848,
        'camera_geometry_in_frame':coverage['all_geometry_in_frame'],
        'all_collision_shapes_present':all(len(actor.actor.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent).collision_shapes)>0 for actor in env.objects.values())}
    result={'seed':seed,'randomized':randomized,'checks':checks,'passed':all(checks.values()),
        'evidence':evidence,'head_camera_coverage':coverage,'maximum_linear_speeds_m_s':speed,
        'gpu_used':False,'rendering_used':False,'videos_recorded':False,
        'expert_planning_verified':False,'rgb_occlusion_verified':False}
    env.scene._scene.clear()
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--seeds',type=int,default=3)
    parser.add_argument('--out',type=Path,default=ROOT/'outputs/validation/zhuhai_task2_20261008/cpu_scene_report.json')
    args=parser.parse_args()
    results=[run(seed,randomized) for randomized in [False,True] for seed in range(args.seeds)]
    files=[ROOT/'third_party/RoboTwin/envs/zhuhai_task2.py',ROOT/'third_party/RoboTwin/envs/zhuhai_camera.py',ROOT/'harness/zhuhai_task2_rules.py']
    report={'scope':'actual collision+contact+settling CPU physics; robot expert and actual RGB visibility pending GPU coordinator',
        'all_passed':all(item['passed'] for item in results),'runs':results,
        'source_hashes':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'all_passed':report['all_passed'],'runs':len(results),'failed':[{k:item[k] for k in ['seed','randomized','checks']} for item in results if not item['passed']],'report':str(args.out)},ensure_ascii=False))
    return 0 if report['all_passed'] else 2

if __name__=='__main__':raise SystemExit(main())
