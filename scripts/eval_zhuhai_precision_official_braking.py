#!/usr/bin/env python3
"""Independent evaluation overlay: immutable Precision-V2 model, Panda braking.

Reuses the frozen physical evaluator, scoring, IPC, observations and clock.
The checkpoint's TRAIN contract is verified unchanged before applying an
explicit evaluation-only postprocessing overlay. No checkpoint is rewritten.
"""
from copy import deepcopy
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from scripts import eval_zhuhai_wam_precision_v2 as original
from harness import zhuhai_official_braking as braking
from scripts.progress1_log import log
base = original.base
precision = original.precision
FROZEN_READ_CONTRACT = base.read_contract


def overlay(contract):
    result = deepcopy(contract)
    result['action_postprocessing'] = braking.protocol()
    result['action_postprocessing_sha256'] = base.digest(result['action_postprocessing'])
    result['evaluation_overlay'] = dict(version=braking.VERSION,
        training_contract_sha256=base.digest(contract), checkpoint_modified=False,
        changes=['official per-axis Panda target limits', 'discrete braking feasibility'],
        unchanged=['model weights', 'scene/scoring', 'seeds', 'spline', 'exec8', 'event clock', 'gripper samples'])
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
        def sample(feats, inputs, seed, step, event_state):
            state, pool = inputs
            seed_value = int(hashlib.sha256(f'{seed}:{step}'.encode()).hexdigest()[:15], 16)
            generator = torch.Generator(device='cuda').manual_seed(seed_value)
            with torch.inference_mode():
                actions, _ = wrapper.sample_actions(feats, torch.from_numpy(state)[None],
                    torch.from_numpy(pool)[None], torch.from_numpy(condition)[None], num_steps=10,
                    generator=generator, event_state=torch.from_numpy(event_state.copy())[None])
                return actions[0].float().cpu().numpy()
        v3 = cfg['model']['v3']
        policy = braking.install_braking(original.clocked_policy(
            V3Context(v3['proprio_lags'], v3['memory_lags'], base.MODEL_VIEWS, state_dim=16),
            encode, sample, contract['action_postprocessing']), contract['action_postprocessing'])
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
            system_mode='precision_wam_v2_official_braking', precision_refiner_used_online=True,
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
    base.zt = precision; base.worker = worker; base.log = log
    base.__file__ = __file__; base.verify_owner = original.verify_owner
    base.read_contract = read_contract; base.Native16PolicyClient = BrakingClient


def main():
    configure()
    return base.main()


if __name__ == '__main__': raise SystemExit(main())
