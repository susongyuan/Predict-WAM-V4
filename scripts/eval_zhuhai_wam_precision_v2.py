#!/usr/bin/env python3
"""Precision-WAM online clock and loader; reuse frozen native16 physics/scoring/IPC."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts import eval_zhuhai_wam_hjepa as base
from harness import zhuhai_precision_v2_training as precision
from scripts.progress1_log import log


def verify_owner(expected):
    """The new policy must never share its simulator with a live trainer."""
    import os
    precision.verify_owner(expected)
    trainer_names={'train_zhuhai_wam.py','train_zhuhai_wam_hjepa.py','train_zhuhai_wam_precision.py','train_zhuhai_wam_precision_v2.py'}
    trainer_modules={'scripts.'+Path(name).stem for name in trainer_names}
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit() or int(entry.name)==os.getpid():continue
        try:argv=(entry/'cmdline').read_bytes().decode(errors='replace').split('\0')
        except (FileNotFoundError,ProcessLookupError,PermissionError):continue
        if any(Path(arg).name in trainer_names or arg in trainer_modules for arg in argv if arg):
            raise base.InfrastructureError('training process still present; cannot overlap Precision evaluation')


def clocked_policy(context,encode,sample,settings):
    from harness.zhuhai_action_smoothing import SmoothedChunkPolicy
    from harness.wam_event_clock import EventClock
    class ClockedPolicy(SmoothedChunkPolicy):
        def __init__(self):
            self.clock=EventClock(**precision.CLOCK_CONFIG);self.event_state=None
            super().__init__(context,encode,lambda feats,inputs,seed,step:sample(feats,inputs,seed,step,self.event_state),settings)
        def reset(self,episode,seed):
            super().reset(episode,seed);self.clock.reset();self.event_state=None
        def observe(self,episode,step,observation):
            if episode!=self.episode or step!=self.next_step:raise ValueError('out-of-order clock observation')
            base.validate_observation(observation)
            self.event_state=self.clock.update(observation['measured_joint_gripper_state'],observation['commanded_joint_gripper_state'])
            return super().observe(episode,step,observation)
    return ClockedPolicy()


def worker(args):
    from harness.robotwin_multitask_policy_worker import die_with_parent
    import socket,hashlib
    die_with_parent();channel=base.Channel(socket.socket(fileno=args.ipc_fd))
    try:
        verify_owner(args.gpu_lock_owner);contract,receipt=base.read_contract(args.checkpoint)
        if base.identity(args.checkpoint)['sha256']!=receipt['sha256']:raise base.InfrastructureError('Checkpoint SHA256 differs')
        import cv2,numpy as np,torch
        from importlib.metadata import version
        from harness.robotwin_training_data import preprocess_rgb
        from harness.wam_behavior import load_upstream
        from harness.wam_v3 import V3Context
        torch.set_num_threads(2);torch.set_num_interop_threads(2);cv2.setNumThreads(2)
        wrapper,condition,actual=precision.load_policy(args.checkpoint,args.task,'cuda')
        if actual!=contract:raise base.InfrastructureError('Loaded checkpoint differs from manifest')
        cfg=contract['config']
        if version('scipy')!=contract['action_postprocessing']['worker_scipy_version']:raise base.InfrastructureError('SciPy protocol differs')
        _,runner=load_upstream();encoder_path=ROOT/cfg['model']['vision_encoder']['checkpoint_path']
        if base.identity(encoder_path/'model.safetensors')['sha256']!=contract['cache']['encoder_sha256']:raise base.InfrastructureError('DINO SHA differs')
        encoder,hidden,registers,patch=runner.ModelFactory.create_vision_encoder(str(encoder_path),dtype=torch.bfloat16,device='cuda')
        if (hidden,registers,patch)!=(1024,4,16):raise base.InfrastructureError('DINO geometry differs')
        wrapper.base.vision_encoder=encoder.eval().requires_grad_(False);wrapper.eval()
        def encode(rgb):
            verify_owner(args.gpu_lock_owner)
            with torch.inference_mode():
                pixels=np.stack([preprocess_rgb(rgb[v]) for v in base.VIEWS])
                feats,pool=wrapper.encode(torch.from_numpy(pixels)[None])
                return feats,pool[0].float().cpu().numpy()
        def sample(feats,inputs,seed,step,event_state):
            state,pool=inputs;seed_value=int(hashlib.sha256(f'{seed}:{step}'.encode()).hexdigest()[:15],16)
            generator=torch.Generator(device='cuda').manual_seed(seed_value)
            with torch.inference_mode():
                actions,_=wrapper.sample_actions(feats,torch.from_numpy(state)[None],torch.from_numpy(pool)[None],
                    torch.from_numpy(condition)[None],num_steps=10,generator=generator,
                    event_state=torch.from_numpy(event_state.copy())[None])
                return actions[0].float().cpu().numpy()
        v3=cfg['model']['v3']
        policy=clocked_policy(V3Context(v3['proprio_lags'],v3['memory_lags'],base.MODEL_VIEWS,state_dim=16),encode,sample,contract['action_postprocessing'])
        policy.runtime=dict(format=precision.FORMAT,task=args.task,**precision.protocol_binding(),checkpoint_sha256=receipt['sha256'],
            checkpoint_step=receipt['step'],strict_load=True,projections_reset=False,
            task_condition_source='new_successful_TRAIN_endpoints_only',task_condition_sha256=hashlib.sha256(condition.tobytes()).hexdigest(),
            condition_source_split='train',numerical_normalization=contract['numerical_normalization'],time=contract['time'],
            action_postprocessing=contract['action_postprocessing'],action_postprocessing_sha256=contract['action_postprocessing_sha256'],
            raw_action_protocol_alias='limited_native_driver_action; original outputs archived as model_raw_action',
            dino_frozen=True,system_mode='precision_wam_v2',hjepa_auxiliary_heads_used_online=False,future_inputs_used_online=False,
            precision_refiner_used_online=True,event_clock_used_online=True,event_clock_is_verified_grasp=False,
            precision=contract['precision'],hjepa_variant=contract['hjepa']['variant'],vlm=False,high_level_planner=False,ask=False,memory_writes=False)
        base.serve_native16(channel,policy);return 0
    except BaseException as exc:
        try:channel.send(base.envelope('infrastructure_error','startup',0,error=f'{type(exc).__name__}: {exc}'),1)
        except BaseException:pass
        raise
    finally:channel.sock.close()


def main():
    base.zt=precision;base.worker=worker;base.log=log;base.__file__=__file__;base.verify_owner=verify_owner
    return base.main()

if __name__=='__main__':raise SystemExit(main())
