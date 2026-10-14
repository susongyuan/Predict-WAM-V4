#!/usr/bin/env python3
"""CPU checks of exec16 cadence, dispatched samples, and runtime binding."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from harness.zhuhai_robot_spec import HOME_ACTION
from harness.zhuhai_exec16_deployment import Exec16SmoothedChunkPolicy, protocol, overlay
from harness import zhuhai_action_smoothing as smoothing
from scripts import eval_precision_exec16_final5 as q
from scripts import eval_zhuhai_wam_precision_exec16 as runtime
from scripts.test_eval_precision_v4 import fixture


class Exec16Tests(unittest.TestCase):
    def test_real_queue_cadence_samples_braking_and_history(self):
        class Context:
            def reset(self): self.states = []
            def push(self, state, pool): self.states.append(state.copy())
            def inputs(self): return self.states[-1], np.zeros(1)
        calls = []
        home = np.asarray(HOME_ACTION, dtype=np.float64)
        raw = np.tile(home, (32, 1))
        raw[:, smoothing.ARM] += np.linspace(0., .04, 32)[:, None]
        raw[:, smoothing.GRIPPER] = (np.arange(32)[:, None] % 3 == 0)
        def sample(feats, inputs, seed, step, clock):
            calls.append(step)
            return raw.copy()
        settings = protocol()
        policy = Exec16SmoothedChunkPolicy(Context(), lambda rgb: (None, np.zeros(1)), sample, settings)
        policy.reset('test', 932101)
        gold = q.e.braking.BrakingRateLimiter(settings['rate_limit'])
        gold.reset(home)
        smoothed = smoothing.smooth_chunk(raw, settings)
        rows = []
        observation = dict(rgb={name: np.zeros((2, 2, 3), np.uint8) for name in runtime.base.VIEWS},
                           measured_joint_gripper_state=home.tolist(), commanded_joint_gripper_state=home.tolist())
        for step in range(35):
            reply = policy.observe('test', step, observation)
            target, stats = gold.step(smoothed[step % 16])
            self.assertEqual(reply['limited_action'], target.tolist())
            self.assertEqual(reply['rate_limit_stats'], stats)
            self.assertEqual(reply['model_raw_action'], raw[step % 16].tolist())
            self.assertEqual(reply['replanned'], step % 16 == 0)
            rows.append(dict(policy=reply, smoothed_action=reply['smoothed_action']))
        self.assertEqual(calls, [0, 16, 32])
        self.assertEqual(len(policy.context.states), 35)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'actions.jsonl'
            path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            self.assertEqual(q.verify_exec16_actions(directory, settings)['replans'], 3)
            rows[8]['policy']['replanned'] = True
            path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            with self.assertRaises(q.e.EvidenceError):
                q.verify_exec16_actions(directory, settings)

    def test_served_v4_and_original_v2_pass_strict_binding(self):
        for step, family in ((130000, 'v4'), (80000, 'v2')):
            with self.subTest(family=family):
                raw, frozen = fixture(step)
                rt = raw['policy_ready']['runtime']
                frozen['time'] = deepcopy(rt['time'])
                frozen['precision'] = deepcopy(rt['precision'])
                before = deepcopy(frozen)
                deployed = overlay(frozen)
                self.assertEqual(frozen, before)
                rt.update(action_postprocessing=deployed['action_postprocessing'],
                          action_postprocessing_sha256=deployed['action_postprocessing_sha256'])
                actual = (dict(q.old.BACKBONE_BINDING, executed_prefix=16) if family == 'v4' else
                          dict(version=2, refined_flow_steps=8, dense_kv_tokens=588, queries_per_view=16,
                               global_visual_tokens=48, patch_projection_input_dim=1026, executed_prefix=16))
                loaded = dict(family=family, actual=actual, identity=rt['precision_v4_actual'],
                              alpha=.02, spec=q.v4.PRECISION_V4_SPEC)
                with patch.object(runtime, '_loaded', loaded), patch.object(runtime, '_serve'):
                    runtime.serve(None, SimpleNamespace(runtime=rt))
                if family == 'v2':
                    rt['format'] = q.v2.FORMAT
                    rt.pop('precision_v4_actual', None)
                    rt.pop('chunk_vtt16', None)
                raw.update(deployment=rt['deployment'], deployment_sha256=rt['deployment_sha256'],
                           action_postprocessing=rt['action_postprocessing'],
                           action_postprocessing_sha256=rt['action_postprocessing_sha256'])
                q.validate_binding(raw, step, 932101, 'a' * 64, deployed)
                rt['deployment'] = dict(rt['deployment'], execution_prefix=8)
                with self.assertRaises(q.e.EvidenceError):
                    q.validate_binding(raw, step, 932101, 'a' * 64, deployed)

    def test_recording_defaults_and_no_extra_camera(self):
        with patch.object(runtime, '_write') as write:
            runtime.write_json('result.json', dict(record_video=True, third_view=True, control={}))
            saved = write.call_args.args[1]
            self.assertIs(saved['record_video'], False)
            self.assertIs(saved['third_view'], False)
            self.assertEqual(saved['control']['execution_prefix'], 16)

    def test_plan_binds_existing_sources_and_all_fifteen_attempts(self):
        training = q.BASE / 'wam_precision_v4_task2_50k_20261013_exec16/train'
        contract = json.loads((training / 'manifest.json').read_text())['contract']
        checkpoints = [dict(path=str(training / f'checkpoint_step_{step:06d}.pt'),
                            step=step, sha256='a' * 64, size=1, label='v4') for step in (130000, 127500)]
        baseline = dict(path='baseline.pt', step=80000, sha256='b' * 64, size=1, label='original_v2')
        with tempfile.TemporaryDirectory(dir=q.BASE) as directory, \
                patch.object(q.old, 'audit_training', return_value=(contract, checkpoints, {})), \
                patch.object(q, 'baseline_checkpoint', return_value=(baseline, contract)):
            plan = q.make_plan(SimpleNamespace(training=training, out=Path(directory) / 'evaluation',
                                               gpu_lock_owner='cpu-test'))
        self.assertEqual(plan['planned_episodes'], 15)
        self.assertEqual([item['seed'] for item in plan['commands']], list(q.SEEDS) * 3)
        self.assertTrue(all(str(q.RUNTIME) in item['command'] for item in plan['commands']))
        self.assertIn(str(ROOT / 'harness/zhuhai_exec16_deployment.py'), plan['frozen_sources'])
        self.assertIs(plan['record_video'], False)


if __name__ == '__main__':
    unittest.main()
