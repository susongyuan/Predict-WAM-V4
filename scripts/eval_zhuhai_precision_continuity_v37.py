#!/usr/bin/env python3
"""Independent evaluation overlay: immutable Precision-V2 model, Panda braking.

Reuses the frozen physical evaluator, scoring, IPC, observations and clock.
The checkpoint's TRAIN contract is verified unchanged before applying an
explicit evaluation-only postprocessing overlay. No checkpoint is rewritten.
"""
from copy import deepcopy
from pathlib import Path
import os
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from scripts import eval_zhuhai_wam_precision_v2 as original
from harness import zhuhai_chunk_continuity_v37 as braking
from scripts.progress1_log import log
base = original.base
precision = original.precision
FROZEN_READ_CONTRACT = base.read_contract
FROZEN_WRITE_JSON = base.atomic_json
FROZEN_EPISODE = base.episode


def requested_video(seed):
    return os.environ.get('P2_CONTINUITY_THIRD_VIEW_SEED')=='932106' and seed==932106


def write_result(path,row):
    if Path(path).name=='result.json':
        selected=requested_video(row.get('seed'))
        row=dict(row,record_video=selected,third_view=selected,
            observer_policy_input=False,recording_scope='separate requested observer; three onboard policy sensors unchanged')
    return FROZEN_WRITE_JSON(path,row)


def episode(args,contract,receipt):
    result=FROZEN_EPISODE(args,contract,receipt)
    if requested_video(args.episode_seed):
        import json
        directory=Path(args.out)
        raw=json.loads((directory/'result.json').read_text())
        raw['observer_video_receipt']=json.loads((directory/'third_person_video_receipt.json').read_text())
        write_result(directory/'result.json',raw)
    return result


def overlay(contract):
    result = deepcopy(contract)
    result['action_postprocessing'] = braking.protocol()
    result['action_postprocessing_sha256'] = base.digest(result['action_postprocessing'])
    result['evaluation_overlay'] = dict(version=braking.VERSION,
        training_contract_sha256=base.digest(contract), checkpoint_modified=False,
        changes=['generation-time RTC soft-overlap VJP guidance with synchronous delay0', 'C1 arm reference anchored at last dispatched position/velocity', '250Hz target dispatch with original official per-axis limits and braking'],
        unchanged=['model weights', 'scene/scoring', 'seeds', 'original32-step spline before new C1 correction', 'exec8', 'event clock', 'gripper samples'])
    return result


def read_contract(path):
    contract, receipt = FROZEN_READ_CONTRACT(path)
    return overlay(contract), receipt


class BrakingClient(base.Native16PolicyClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        runtime = self.ready.get('runtime', {})
        settings = braking.protocol()
        if (runtime.get('action_postprocessing') != settings
                or runtime.get('action_postprocessing_sha256') != base.digest(settings)
                or runtime.get('evaluation_version') != braking.VERSION
                or runtime.get('evaluation_source_sha256') != base.identity(__file__)['sha256']):
            self.close()
            raise base.InfrastructureError('Independent braking worker identity mismatch')

    def action(self, episode, step, observation):
        reply = super().action(episode, step, observation)
        braking.set_pending(reply)
        return reply


def worker(args):
    from harness.robotwin_multitask_policy_worker import die_with_parent
    import socket, hashlib
    die_with_parent()
    channel = base.Channel(socket.socket(fileno=args.ipc_fd))
    try:
        original.verify_owner(args.gpu_lock_owner)
        frozen, receipt = FROZEN_READ_CONTRACT(args.checkpoint)
        contract = overlay(frozen)
        if base.identity(args.checkpoint)['sha256'] != receipt['sha256']:
            raise base.InfrastructureError('Checkpoint SHA mismatch')
        import cv2, numpy as np, torch
        from importlib.metadata import version
        from harness.robotwin_training_data import preprocess_rgb
        from harness.wam_behavior import load_upstream
        from harness.wam_v3 import V3Context
        torch.set_num_threads(2); torch.set_num_interop_threads(2); cv2.setNumThreads(2)
        wrapper, condition, actual = precision.load_policy(args.checkpoint, args.task, 'cuda')
        if actual != frozen: raise base.InfrastructureError('Frozen model contract differs')
        cfg = frozen['config']
        if version('scipy') != frozen['action_postprocessing']['worker_scipy_version']:
            raise base.InfrastructureError('Original spline SciPy protocol differs')
        _, runner = load_upstream()
        encoder_path = ROOT/cfg['model']['vision_encoder']['checkpoint_path']
        if base.identity(encoder_path/'model.safetensors')['sha256'] != frozen['cache']['encoder_sha256']:
            raise base.InfrastructureError('DINO SHA differs')
        encoder, hidden, registers, patch = runner.ModelFactory.create_vision_encoder(
            str(encoder_path), dtype=torch.bfloat16, device='cuda')
        if (hidden, registers, patch) != (1024, 4, 16): raise base.InfrastructureError('DINO geometry differs')
        wrapper.base.vision_encoder = encoder.eval().requires_grad_(False); wrapper.eval()
        def encode(rgb):
            original.verify_owner(args.gpu_lock_owner)
            with torch.inference_mode():
                pixels = np.stack([preprocess_rgb(rgb[v]) for v in base.VIEWS])
                feats, pool = wrapper.encode(torch.from_numpy(pixels)[None])
                return feats, pool[0].float().cpu().numpy()
        from harness.zhuhai_overlap_guidance_v37 import OverlapSampler
        sample = OverlapSampler(wrapper, condition, lambda: original.verify_owner(args.gpu_lock_owner))
        v3 = cfg['model']['v3']
        policy = braking.install_braking(original.clocked_policy(
            V3Context(v3['proprio_lags'], v3['memory_lags'], base.MODEL_VIEWS, state_dim=16),
            encode, sample, contract['action_postprocessing']), contract['action_postprocessing'])
        policy.overlap_sampler = sample
        policy.runtime = dict(format=precision.FORMAT, task=args.task, **precision.protocol_binding(),
            checkpoint_sha256=receipt['sha256'], checkpoint_step=receipt['step'], strict_load=True,
            projections_reset=False, training_contract_sha256=base.digest(frozen),
            numerical_normalization=frozen['numerical_normalization'], time=frozen['time'],
            action_postprocessing=contract['action_postprocessing'],
            action_postprocessing_sha256=contract['action_postprocessing_sha256'],
            evaluation_version=braking.VERSION, evaluation_source_sha256=base.identity(__file__)['sha256'],
            evaluation_overlay=contract['evaluation_overlay'], dino_frozen=True,
            task_condition_source='new_successful_TRAIN_endpoints_only',
            task_condition_sha256=hashlib.sha256(condition.tobytes()).hexdigest(), condition_source_split='train',
            precision=frozen['precision'], hjepa_variant=frozen['hjepa']['variant'],
            system_mode='precision_wam_v2_rtc_soft_overlap_c1_fine250', precision_refiner_used_online=True,
            event_clock_used_online=True, event_clock_is_verified_grasp=False,
            hjepa_auxiliary_heads_used_online=False, future_inputs_used_online=False,
            vlm=False, high_level_planner=False, ask=False, memory_writes=False)
        base.serve_native16(channel, policy)
        return 0
    except BaseException as exc:
        try: channel.send(base.envelope('infrastructure_error', 'startup', 0, error=f'{type(exc).__name__}: {exc}'), 1)
        except BaseException: pass
        raise
    finally:
        channel.sock.close()


def configure():
    braking.install_environment()
    from harness import zhuhai_observer_video_v37
    zhuhai_observer_video_v37.install()
    base.zt = precision; base.worker = worker; base.log = log
    base.__file__ = __file__; base.verify_owner = original.verify_owner
    base.read_contract = read_contract; base.Native16PolicyClient = BrakingClient
    base.atomic_json = write_result; base.episode = episode


def main():
    configure()
    return base.main()


if __name__ == '__main__': raise SystemExit(main())
