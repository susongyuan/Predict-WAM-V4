#!/usr/bin/env python3
"""Collect genuine physical expert demonstrations; never produce videos.

Use --plan-only for a CPU-only contract/interface check. Actual collection must
run in .venvs/robotwin-sim after the coordinator grants its external GPU lock.
Success means the declared simulation rules, not competition certification.
"""
from __future__ import annotations

import argparse
import ast
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.progress1_log import log

FREQUENCY = 10.0
MINIMUM_FREE = 100 * 1024 ** 3
GPU_LOCK = Path('/tmp/project2_gpu.lock')


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n')
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify_owner(owner):
    if not owner or not GPU_LOCK.exists() or GPU_LOCK.read_text().strip() != owner:
        raise RuntimeError('External GPU lock owner mismatch; coordinator must grant the lock first')


def check_disk(path):
    available = shutil.disk_usage(path).free
    if available < MINIMUM_FREE:
        raise RuntimeError(f'Disk reserve breached: {available} bytes free, require {MINIMUM_FREE}')
    return available


def cpu_interface(task):
    path = ROOT / 'third_party/RoboTwin/envs' / f'{task}.py'
    if not path.is_file():
        return {'path': str(path), 'available': False, 'reason': 'scene not implemented'}
    tree = ast.parse(path.read_text())
    classes = [item for item in tree.body if isinstance(item, ast.ClassDef) and item.name == task]
    methods = {item.name for cls in classes for item in cls.body
               if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))}
    missing = sorted({'setup_demo', 'play_once', 'check_success', 'rule_state'} - methods)
    return {'path': str(path), 'available': bool(classes) and not missing,
            'missing_methods': missing, 'sha256': sha256(path)}


def validate_evidence(evidence):
    if not isinstance(evidence, dict):
        raise ValueError('rule_state must return a dict')
    now = evidence.get('physics_time_seconds')
    if isinstance(now, bool) or not isinstance(now, (int, float)) or not math.isfinite(now) or now < 0:
        raise ValueError('rule_state needs finite nonnegative physics_time_seconds')
    score = evidence.get('score')
    if not isinstance(score, dict) or type(score.get('full_task_success')) is not bool:
        raise ValueError('rule_state.score.full_task_success must be explicit bool')
    if not isinstance(score.get('base_score'), (int, float)) or not math.isfinite(score['base_score']):
        raise ValueError('rule_state.score.base_score must be finite')
    if not score.get('rule_source') or not isinstance(score.get('items'), dict) or not score['items']:
        raise ValueError('rule evidence must identify its source and all scored items')
    if not isinstance(evidence.get('objects'), dict) or not evidence['objects']:
        raise ValueError('rule evidence must preserve physical object states')
    if not isinstance(score.get('simulation_tolerances'), dict):
        raise ValueError('rule evidence must declare simulation tolerances')
    # Reject non-JSON/NaN evidence before indexing an episode as successful.
    json.dumps(evidence, allow_nan=False)
    return float(now), score


class PhysicsSampler:
    """At most 10 Hz; ignore duplicate move-boundary callbacks before rendering."""
    def __init__(self, env, recorder):
        self.env, self.recorder = env, recorder
        self.last_time = None
        self.accepted_times = []
        self.filtered_callbacks = 0

    def capture(self, *, force=False):
        # The scene observer updates this clock before our callback. Most
        # physics steps need no RGB rendering or JSON serialization.
        clock = getattr(self.env, 'physics_time', None)
        if not force and clock is not None and self.last_time is not None:
            if float(clock) - self.last_time < 1 / FREQUENCY - 1e-9:
                self.filtered_callbacks += 1
                return False
        evidence = self.env.rule_state()
        now, _ = validate_evidence(evidence)
        if self.last_time is not None and now < self.last_time - 1e-9:
            raise ValueError('physics clock moved backwards')
        if self.last_time is not None and now <= self.last_time + 1e-9:
            self.filtered_callbacks += 1
            return False
        if not force and self.last_time is not None and now - self.last_time < 1 / FREQUENCY - 1e-9:
            self.filtered_callbacks += 1
            return False
        self.recorder.capture(self.env.get_obs(), now, evidence)
        self.last_time = now
        self.accepted_times.append(now)
        return True


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--task', required=True, choices=['zhuhai_task1', 'zhuhai_task2'])
    ap.add_argument('--track', required=True, choices=['life', 'production'])
    ap.add_argument('--mode', required=True, choices=['source', 'randomized'])
    ap.add_argument('--count', required=True, type=int)
    ap.add_argument('--seed-start', required=True, type=int)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--gpu-lock-owner')
    ap.add_argument('--allow-unvalidated-probe', action='store_true')
    ap.add_argument('--life-mapping', choices=['body_text', 'scoring_table'])
    ap.add_argument('--source-manifest', type=Path)
    ap.add_argument('--plan-only', action='store_true')
    ap.add_argument('--max-attempts', type=int,
                    help='Cannot exceed count*10; defaults to count*10')
    ap.add_argument('--max-consecutive-failures', type=int, default=3,
                    help='Stop and retain evidence after this many failed attempts')
    ap.add_argument('--task-config', default='demo_clean')
    return ap


def source_families(args):
    if args.mode == 'source':
        return []
    if not args.source_manifest:
        raise ValueError('Randomized collection requires --source-manifest for registered family grouping')
    manifest = json.loads(args.source_manifest.read_text())
    contract = manifest.get('contract', {})
    if (contract.get('task'), contract.get('track'), contract.get('mode')) != (args.task, args.track, 'source'):
        raise ValueError('Source manifest task/track/mode mismatch')
    families = [item for item in manifest.get('attempts', []) if item.get('status') == 'success']
    if not families:
        raise ValueError('Source manifest has no successful physical demonstrations')
    return families


def build_contract(args, interface):
    dependencies = [ROOT / 'harness' / f'{args.task}_rules.py',
                    ROOT / 'harness/zhuhai_expert_robot.py',
                    ROOT / 'third_party/RoboTwin/envs/zhuhai_camera.py']
    result = dict(schema='zhuhai_physical_source_v1', task=args.task, track=args.track,
                  mode=args.mode, seed_start=args.seed_start, task_config=args.task_config,
                  life_mapping=args.life_mapping, allow_unvalidated_probe=args.allow_unvalidated_probe,
                  robot='fixed_aloha_agilex', frequency_hz=FREQUENCY,
                  expert_planner_mode='mplib_expert', need_plan=True, eval_mode=False,
                  max_consecutive_failures=args.max_consecutive_failures,
                  record_video=False, third_view=False, minimum_free_bytes=MINIMUM_FREE,
                  source_scene=interface, recorder_sha256=sha256(ROOT/'harness/zhuhai_demo_recording.py'),
                  runner_sha256=sha256(__file__),
                  success_scope='strict declared simulation proxy, not competition certification')
    result['scene_dependency_sha256'] = {str(path): sha256(path) for path in dependencies if path.is_file()}
    if args.source_manifest:
        result['source_manifest'] = dict(path=str(args.source_manifest.resolve()),
                                         sha256=sha256(args.source_manifest))
    return result


def config_for(args):
    from scripts.eval_robotwin_wam import load_task_config
    config, files = load_task_config(args.task_config, args.task)
    config = copy.deepcopy(config)
    config.update(need_plan=True, eval_mode=False, planner_mode='mplib_expert',
                  save_data=False, collect_data=False, render_freq=0,
                  eval_video_log=False, eval_video_save_dir=None, render_denoiser='none',
                  zhuhai_track=args.track, zhuhai_randomized=args.mode == 'randomized',
                  zhuhai_life_mapping=args.life_mapping,
                  zhuhai_allow_unvalidated_probe=args.allow_unvalidated_probe)
    config['data_type'].update(third_view=False, observer=False, rgb=True, qpos=True,
                              depth=False, pointcloud=False, mesh_segmentation=False,
                              actor_segmentation=False)
    config['camera'].update(collect_head_camera=True, collect_wrist_camera=True)
    if args.mode == 'randomized':
        config['domain_randomization'].update(random_background=True, cluttered_table=False,
            clean_background_rate=0.25, random_head_camera_dis=0.015,
            random_table_height=0.008, random_light=True, crazy_random_light_rate=0.0)
    return config, {str(path): sha256(path) for path in files}


def initial_manifest(path, contract, count):
    if path.exists():
        result = json.loads(path.read_text())
        if result['contract'] != contract:
            raise ValueError('Resume contract changed: use a new output directory; successful data is immutable')
        for item in result['attempts']:
            if item.get('status') == 'success':
                file = Path(item['data_path'])
                if not file.exists() or sha256(file) != item['data_sha256']:
                    raise ValueError(f'Previously successful demonstration missing or changed: {file}')
            elif item.get('status') == 'running':
                item['status'] = 'interrupted'
                item['failure_reason'] = 'Earlier process stopped; retained partial evidence, seed not silently replayed'
        result['requested_success_count'] = count
    else:
        result = dict(contract=contract, requested_success_count=count, attempts=[], status='ready')
    atomic_json(path, result)
    return result


def collect(args, contract, families):
    from scripts.eval_robotwin_wam import make_env
    from harness.zhuhai_demo_recording import StreamDemoRecorder, jsonable
    config, config_hashes = config_for(args)
    contract = dict(contract, config_hashes=config_hashes,
                    resolved_expert_configuration=jsonable(config))
    manifest_path = args.out / 'manifest.json'
    manifest = initial_manifest(manifest_path, contract, args.count)
    maximum = args.max_attempts or args.count * 10
    successes = sum(item.get('status') == 'success' for item in manifest['attempts'])
    consecutive_failures = 0
    seeds = {item['seed'] for item in manifest['attempts']}
    seed = max(seeds, default=args.seed_start - 1) + 1
    while successes < args.count and len(manifest['attempts']) < maximum:
        log(f'采集操作前：{args.task}/{args.track}/{args.mode} seed={seed}；'
            f'已成功{successes}/{args.count}，视频关闭，核对GPU锁与100GiB磁盘余量', actor='collect_zhuhai_data')
        verify_owner(args.gpu_lock_owner)
        for path, expected in contract.get('scene_dependency_sha256', {}).items():
            if sha256(path) != expected:
                raise RuntimeError(f'Scene dependency changed during collection: {path}')
        if sha256(contract['source_scene']['path']) != contract['source_scene']['sha256']:
            raise RuntimeError('Scene source changed during collection; retain existing evidence and use a new run')
        free = check_disk(args.out)
        index = len(manifest['attempts'])
        parent = families[index % len(families)] if families else None
        family_id = parent['family_id'] if parent else f'{args.track}:{args.task}:source:{seed}'
        name = f'episode_seed_{seed:010d}'
        data_path = args.out / 'data' / f'{name}.hdf5'
        if data_path.exists() or data_path.with_suffix('.hdf5.partial').exists():
            raise FileExistsError(f'Unindexed evidence at {data_path}; do not overwrite, inspect or use a new output')
        metadata = dict(task=args.task, track=args.track, mode=args.mode, seed=seed,
                        family_id=family_id, record_video=False, minimum_free_bytes=free,
                        parent_source_seed=parent['seed'] if parent else None,
                        family_relation='independent physical rollout; grouping only' if parent else 'source seed',
                        domain_randomization=config['domain_randomization'], contract=contract)
        attempt = dict(seed=seed, family_id=family_id, metadata=metadata, status='running',
                       source_parent=parent['seed'] if parent else None,
                       domain=args.mode, started_unix=time.time())
        manifest['attempts'].append(attempt)
        manifest['status'] = 'collecting'
        atomic_json(manifest_path, manifest)
        env = recorder = sampler = None
        final = None
        try:
            env = make_env(args.task)
            if not callable(getattr(env, 'rule_state', None)):
                raise ValueError('Scene must implement strict auditable rule_state()')
            parameters = copy.deepcopy(config)
            parameters.update(seed=seed, now_ep_num=index, save_path=str(args.out),
                              zhuhai_source_seed=parent['seed'] if parent else seed)
            env.setup_demo(**parameters)
            initial = jsonable(env.rule_state())
            initial_time, initial_score = validate_evidence(initial)
            attempt['initial_rule_evidence'] = initial
            if initial_score['full_task_success'] or initial_score['base_score'] != 0:
                raise ValueError('Initial scene already has nonzero rule score; reject invalid task initialization')
            if env.eval_mode or not env.need_plan or env.robot.planner_mode != 'mplib_expert':
                raise ValueError('Expert must use declared MPLib Cartesian planning with need_plan=True/eval_mode=False')
            recorder = StreamDemoRecorder(data_path, metadata=metadata, frequency=FREQUENCY)
            sampler = PhysicsSampler(env, recorder)
            sampler.capture(force=True)
            if callable(getattr(env.scene, '_callback', None)):
                original_callback = env.scene._callback

                def observed_step():
                    original_callback()
                    sampler.capture()

                env.scene._callback = observed_step
                env._take_picture = lambda: None
                env.save_freq = None
                attempt['sampling_method'] = 'strict_10hz_physics_step_callback'
            elif args.allow_unvalidated_probe:
                env.save_freq = max(1, round((1/FREQUENCY)/float(env.scene.get_timestep())))
                env._take_picture = sampler.capture
                attempt['sampling_method'] = 'probe_only_nonuniform_move_callbacks_filtered_to_at_most_10hz'
            else:
                raise ValueError('Formal collection requires a physical-step observer callback')
            verify_owner(args.gpu_lock_owner)
            info = env.play_once()
            verify_owner(args.gpu_lock_owner)
            final = jsonable(env.rule_state())
            final_time, score = validate_evidence(final)
            joints_ok, excess = env.planned_joints_legal()
            predicate = bool(env.check_success())
            successful = bool(env.plan_success and predicate and score['full_task_success'] and joints_ok)
            sampler.capture(force=True)
            if final_time <= initial_time:
                successful = False
            attempt.update(final_rule_evidence=final, expert_info=jsonable(info),
                           plan_success=bool(env.plan_success), strict_predicate=predicate,
                           joints_legal=bool(joints_ok), joint_excess=float(excess),
                           observation_count=recorder.length,
                           accepted_physics_times=sampler.accepted_times,
                           filtered_duplicate_or_fast_callbacks=sampler.filtered_callbacks,
                           save_freq_physics_steps=env.save_freq)
            finished = recorder.finish(final_evidence=final, successful=successful)
            if successful:
                attempt.update(status='success', path=str(finished.resolve()), data_path=str(finished.resolve()),
                               data_sha256=sha256(finished))
                successes += 1
                consecutive_failures = 0
            else:
                failed = args.out / 'failures' / name / finished.name
                failed.parent.mkdir(parents=True, exist_ok=True)
                finished.rename(failed)
                attempt.update(status='failed', failure_reason='Expert failed one or more strict admission criteria',
                               evidence_path=str(failed.resolve()), evidence_sha256=sha256(failed))
        except BaseException as exc:
            attempt.update(status='interrupted' if isinstance(exc, (KeyboardInterrupt, SystemExit)) else 'failed',
                           failure_reason=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
            if recorder is not None and not recorder.closed:
                partial = recorder.abort()
                failed = args.out/'failures'/name/partial.name
                failed.parent.mkdir(parents=True, exist_ok=True)
                if partial.exists():
                    partial.rename(failed)
                    attempt['partial_evidence_path'] = str(failed.resolve())
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
        finally:
            if env is not None:
                try:
                    env.close_env(clear_cache=True)
                except BaseException as exc:
                    attempt['cleanup_error'] = f'{type(exc).__name__}: {exc}'
            attempt['finished_unix'] = time.time()
            if final is not None:
                attempt['final_rule_evidence'] = final
            manifest['success_count'] = successes
            evidence_dir = args.out / 'attempt_evidence'
            evidence_dir.mkdir(exist_ok=True)
            atomic_json(evidence_dir/f'{name}.json', attempt)
            atomic_json(manifest_path, manifest)
        print(json.dumps(dict(seed=seed, status=attempt['status'], success_count=successes,
                              target=args.count), ensure_ascii=False), flush=True)
        if attempt['status'] != 'success':
            consecutive_failures += 1
        if consecutive_failures >= args.max_consecutive_failures:
            manifest.update(status='halted_consecutive_failures',
                            halted_reason=f'{consecutive_failures} consecutive failures; inspect retained evidence before continuing')
            atomic_json(manifest_path, manifest)
            log(f'{args.task}/{args.mode}连续{consecutive_failures}次失败，停止扩量并保全证据；成功{successes}/{args.count}',
                actor='collect_zhuhai_data')
            return 2
        seed += 1
    manifest['status'] = 'complete' if successes >= args.count else 'attempt_limit_reached'
    atomic_json(manifest_path, manifest)
    return 0 if manifest['status'] == 'complete' else 2


def main(argv=None):
    ap = parser()
    args = ap.parse_args(argv)
    if args.count <= 0 or args.seed_start < 0:
        ap.error('--count must be positive and --seed-start nonnegative')
    if args.max_attempts is not None and not 1 <= args.max_attempts <= args.count * 10:
        ap.error('--max-attempts must be in [1, count*10]')
    if args.max_consecutive_failures <= 0:
        ap.error('--max-consecutive-failures must be positive')
    if args.track == 'life' and args.task == 'zhuhai_task2' and not args.life_mapping:
        ap.error('Life task2 requires explicit --life-mapping because the PDF conflicts')
    if not args.plan_only and not args.gpu_lock_owner:
        ap.error('Actual collection requires externally granted --gpu-lock-owner')
    log(f'采集runner预检：{args.task}/{args.track}/{args.mode}，'
        f'count={args.count}，plan_only={args.plan_only}', actor='collect_zhuhai_data')
    interface = cpu_interface(args.task)
    contract = build_contract(args, interface)
    families = source_families(args)
    if args.plan_only:
        print(json.dumps(dict(status='cpu_plan_only', contract=contract,
            registered_source_families=len(families), scene_ready=interface['available'],
            simulation_started=False, requested_count=args.count), ensure_ascii=False, indent=2))
        return 0
    if not interface['available']:
        raise RuntimeError(f'Scene interface unavailable: {interface}')
    verify_owner(args.gpu_lock_owner)
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    check_disk(args.out)
    # Prevent two coordinators from mutating the same manifest simultaneously.
    with (args.out/'.collection.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous_cwd = Path.cwd()
        os.chdir(ROOT / 'third_party/RoboTwin')
        try:
            return collect(args, contract, families)
        finally:
            os.chdir(previous_cwd)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f'collection stopped: {type(error).__name__}: {error}', file=sys.stderr)
        raise SystemExit(2)
