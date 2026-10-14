#!/usr/bin/env python3
"""Model-free native14 admission: only onboard RGB, one robot HOLD, no third camera."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for key in ('DISPLAY', 'WAYLAND_DISPLAY'):
    os.environ.pop(key, None)
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--seed', type=int, default=922000)
    p.add_argument('--gpu-lock-owner', required=True)
    a = p.parse_args(argv)
    from scripts.eval_robotwin_wam import load_task_config, make_env, close_env, ROBOTWIN
    from harness.robotwin_qpos_adapter import NativeQposAdapter, policy_observation
    from harness.robotwin_eval_evidence import joint_bounds
    from harness.robotwin_policy_worker import atomic_json
    from harness.robotwin_camera_readback import SerialColorReadback
    from scripts.gpu_lock import owner
    import numpy as np
    import cv2
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    env = None
    result = dict(status='running', task=a.task, seed=a.seed, policy_constructed=False,
                  expert_called=False, scored_episodes=0, record_video=False,
                  third_view_camera_created=False, views={}, started_at=time.time())
    try:
        if owner() != a.gpu_lock_owner:
            raise ValueError('external GPU lock mismatch')
        os.chdir(ROBOTWIN)
        cfg, _ = load_task_config('demo_clean', a.task)
        cfg['render_denoiser'] = 'none'
        env = make_env(a.task)
        env.setup_demo(now_ep_num=0, seed=a.seed, is_test=True, **cfg)
        env._serial_color_readback = SerialColorReadback(env.cameras)
        obs = policy_observation(env.get_obs())
        result['native_initial_predicate'] = bool(env.check_success())
        if result['native_initial_predicate']:
            raise ValueError('initial state already satisfies native predicate')
        if len(obs['rgb']) != 3:
            raise ValueError('expected exactly 3 policy RGB views')
        for name, image in obs['rgb'].items():
            if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
                raise ValueError('unexpected RGB format')
            if float(image.std()) < 1:
                raise ValueError('blank onboard RGB')
            if not cv2.imwrite(str(out / (name + '.png')), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
                raise OSError('failed to write onboard still')
            result['views'][name] = dict(shape=list(image.shape), std=float(image.std()))
        command = obs['commanded_joint_gripper_state']
        if command.shape != (14,) or not np.isfinite(command).all():
            raise ValueError('invalid native14 state')
        result['initial_commanded_state'] = command.tolist()
        bounds, _ = joint_bounds(env.robot)
        adapter = NativeQposAdapter(env, bounds)
        before = env.take_action_cnt
        adapter.step(command)
        after = env.take_action_cnt
        updated = policy_observation(env.get_obs())
        if (after != before + 1 or updated['commanded_joint_gripper_state'].shape != (14,)
                or not np.isfinite(updated['commanded_joint_gripper_state']).all()):
            raise ValueError('native HOLD interface failed')
        result.update(status='native_interface_smoke_complete', native_take_action_count_delta=after-before,
                      native_step_limit=env.step_lim, after_hold_native_success=bool(env.check_success()),
                      adapter_topp_diagnostics=adapter.last_topp_diagnostics)
    except BaseException as exc:
        result.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
    finally:
        try:
            close_env(env)
        except BaseException as exc:
            result.update(status='failed', cleanup_error=repr(exc))
        result['finished_at'] = time.time()
        atomic_json(out / 'result.json', result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0 if result['status'] == 'native_interface_smoke_complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
