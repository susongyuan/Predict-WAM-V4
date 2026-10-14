"""Real frozen WAM worker condition switching on recorded policy inputs only.
This IPC integration probe is not a new physical trajectory or task score.
"""
import argparse
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from harness.robotwin_policy_worker import PolicyClient, envelope, atomic_json, VIEWS

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--gpu-lock-owner',required=True);a=p.parse_args()
    out=Path(a.out).resolve();out.mkdir(exist_ok=False)
    run=ROOT/'outputs/runs/put_object_cabinet/wam_30k_20261005'
    client=None
    try:
        client=PolicyClient([str(ROOT/'.venvs/wam-fixed/bin/python'),'-B',str(ROOT/'harness/robotwin_policy_worker.py'),
            '--training-run-dir',str(run),'--checkpoint',str(run/'checkpoint_step_030000.pt'),
            '--gpu-lock-owner',a.gpu_lock_owner,'--vtt-adapter',str(ROOT/'outputs/training/cabinet_vtt_adapter_1000_20261006/adapter_step_001000.pt'),
            '--vtt-bank',str(ROOT/'outputs/validation/cabinet_vtt_bank_20261006')],out/'worker.log')
        client.reset('retrieval_probe',10003)
        onboard=ROOT/'outputs/runs/put_object_cabinet/middle_vtt_smoke_20261006/episode_000_seed10003/onboard'
        def read(i):
            with np.load(onboard/f'observation_{i:06d}.npz',allow_pickle=False) as f:
                return {'rgb':{v:f[v].copy() for v in VIEWS},'commanded_joint_gripper_state':f['commanded_joint_gripper_state'].copy()}
        first=client.action('retrieval_probe',0,read(0))
        clear=client.clear_actions('retrieval_probe',1,'explicit_retrieval_interface_probe')
        selected=client.request(envelope('next_demo_condition','retrieval_probe',1),'condition_selected')
        second=client.action('retrieval_probe',1,read(1))
        if (clear.get('history_preserved') is not True or second['history_steps']!=2
                or not selected['condition_source'].startswith('TRAIN-demo:')
                or second['task_condition']['source']!=selected['condition_source']
                or second['task_condition']['sha256']==first['task_condition']['sha256']
                or not second['replanned']):raise ValueError('condition switch/history/queue invariants failed')
        atomic_json(out/'result.json',{'status':'completed','interface_verified':True,'physical_task_scored':False,
            'first':first,'clear':clear,'selected':selected,'second':second,'ready':client.ready})
    except BaseException as e:
        atomic_json(out/'result.json',{'status':'failed','error':f'{type(e).__name__}: {e}'})
        raise
    finally:
        if client:client.close()
if __name__=='__main__':main()
