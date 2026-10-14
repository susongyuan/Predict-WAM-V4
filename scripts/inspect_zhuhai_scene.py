#!/usr/bin/env python3
"""Physical scene smoke check with only native head and wrist observations."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.progress1_log import log
from scripts.collect_zhuhai_data import parser as collection_parser, config_for, verify_owner, atomic_json


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--task', choices=['zhuhai_task1', 'zhuhai_task2'], required=True)
    ap.add_argument('--seed', type=int, default=101001)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--gpu-lock-owner', required=True)
    ap.add_argument('--play', action='store_true')
    ap.add_argument('--randomized', action='store_true')
    args = ap.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    log(f'{args.task}物理烟测seed={args.seed}，专家play={args.play}，仅保存三路机载PNG/规则，无视频', actor='scene_inspect')
    verify_owner(args.gpu_lock_owner)
    parsed = collection_parser().parse_args(['--task', args.task, '--track', 'life', '--mode',
        'randomized' if args.randomized else 'source', '--count', '1', '--seed-start', str(args.seed),
        '--out', str(args.out), '--life-mapping', 'body_text', '--allow-unvalidated-probe'])
    cfg, hashes = config_for(parsed)
    cfg.update(seed=args.seed, now_ep_num=0, save_path=str(args.out), save_freq=None,
               zhuhai_source_seed=args.seed)
    os.chdir(ROOT / 'third_party/RoboTwin')
    from scripts.eval_robotwin_wam import make_env
    from harness.zhuhai_demo_recording import jsonable
    import cv2
    result = dict(task=args.task, seed=args.seed, status='initializing', record_video=False,
                  config_hashes=hashes, started_unix=time.time())
    atomic_json(args.out/'result.json', result)
    env = None

    def capture(stage):
        obs = env.get_obs()
        for name in ('head_camera', 'left_camera', 'right_camera'):
            rgb = obs['observation'][name]['rgb']
            cv2.imwrite(str(args.out/f'{stage}_{name}.png'), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        result[stage + '_rule_evidence'] = jsonable(env.rule_state())
        result[stage + '_info'] = jsonable(env.info)
        cams = env.scene.get_cameras()
        result['camera_names'] = [cam.name for cam in cams]
        result['camera_count'] = len(cams)
        atomic_json(args.out/'result.json', result)

    try:
        env = make_env(args.task)
        env.setup_demo(**cfg)
        capture('initial')
        if args.play:
            env._take_picture = lambda: None
            info = env.play_once()
            capture('final')
            result.update(plan_success=bool(env.plan_success), strict_success=bool(env.check_success()),
                          expert_info=jsonable(info), joint_legality=jsonable(env.planned_joints_legal()))
        result.update(status='completed', wall_seconds=time.time()-result['started_unix'])
    except BaseException as exc:
        result.update(status='failed', error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
        if env is not None and hasattr(env, 'judge') and env.judge is not None:
            try:
                result['failure_rule_evidence'] = jsonable(env.rule_state())
            except Exception:
                pass
    finally:
        atomic_json(args.out/'result.json', result)
        if env is not None:
            try:
                env.close_env(clear_cache=True)
            except Exception as exc:
                result['cleanup_error'] = str(exc)
                atomic_json(args.out/'result.json', result)
    print(json.dumps({key: value for key, value in result.items() if key in
        ['status','error','strict_success','plan_success','wall_seconds','camera_names','camera_count']}), flush=True)
    return 0 if result['status']=='completed' and (not args.play or result['strict_success']) else 2


if __name__ == '__main__':
    raise SystemExit(main())
