#!/usr/bin/env python3
"""CPU-only tests for a distinct 30k+20k stage. Synthetic model; no dataset iteration or CUDA.

Run with CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PYTHONDONTWRITEBYTECODE=1 .conda-envs/lilawam/bin/python <this file>.
"""
from __future__ import annotations
import copy
import importlib.util
import json
import math
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts'))
import train_wam_v3 as T
HERE = Path(__file__).resolve().parent
PLAN = HERE / 'continuation_plan.json'
PARENT = ROOT / T.CONTINUATION_PARENT
CFG = json.loads((PARENT.parent / 'config.json').read_text())


def seed(value):
    torch.random.default_generator.manual_seed(value)
    random.seed(value)
    np.random.seed(value)


def context():
    return T.read_continuation_plan(PLAN, CFG, ROOT / json.loads(PLAN.read_text())['parent']['config'])


def historical_contract(ctx):
    manifest = ctx['parent_manifest']
    sources = dict(manifest['source_sha256'])
    for name in (manifest['args']['init_from'], manifest['args']['resume_source'], manifest['cache_assurance']['ledger']):
        sources.pop(str(T.resolve(name)), None)
    for row in ctx['plan']['source_migrations']:
        sources[str(T.resolve(row['path']))] = row['to_sha256']
    labels = json.loads((PARENT.parent / 'train_labels.json').read_text())
    n = sum(row['anchors'] for row in labels['coverage_by_skill'].values())
    indices = list(range(n))
    assert T.digest_json(indices) == manifest['train_indices_sha256']
    return {'config': copy.deepcopy(CFG),
            'evaluation_plan': json.loads((PARENT.parent / 'validation_plan.json').read_text()),
            'train_indices': indices, 'sources': sources, 'sampler_rule': 'weighted_epoch_seed_v1', 'workers': 0}


class TinyWrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device('cpu')
        self.action_model = torch.nn.Linear(2, 1)
        self.order = []

    def forward(self, batch, lambdas):
        self.order.append(batch['index'])
        noise = torch.randn(()) * 0.05 + random.random() * 0.05 + float(np.random.rand()) * 0.05
        loss = (self.action_model(batch['input']).squeeze() - batch['target'] - noise).square()
        return loss, {'mse': float(loss.detach())}, {}


def synthetic_run(target, checkpoint=None, stage=False):
    """Uses the real train_loop/AdamW/LambdaLR at real global counters on a 3-parameter model."""
    # Real invocations load a new checkpoint object. CPU optimizer.load_state_dict can alias
    # input tensors, so independent test invocations must not share an in-memory fixture.
    checkpoint = copy.deepcopy(checkpoint)
    wrapper = TinyWrapper()
    if checkpoint:
        wrapper.action_model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    optimizer, scheduler = T.make_optimizer(wrapper.action_model, CFG['training'], T.CONTINUATION_SCHEDULE if stage else None)
    inputs = [{'index': i, 'input': torch.tensor([i / 10, 1.]), 'target': torch.tensor(i / 100)} for i in range(9)]
    def loader(epoch):
        generator = torch.Generator().manual_seed(T.noise_seed(71, epoch, 0, 'train-sampler-epoch'))
        sampler = torch.utils.data.WeightedRandomSampler(torch.arange(1, 10, dtype=torch.double), 127,
                                                         replacement=True, generator=generator)
        return [inputs[i] for i in sampler]
    saved = {}
    def save(tag, step, metadata):
        saved[tag] = copy.deepcopy({'step': step, 'model_state_dict': wrapper.action_model.state_dict(),
                                  'config': CFG, 'mode': 'long', 'initialization_sha256': T.V2_SHA256, **metadata})
    result = T.train_loop(wrapper, optimizer, scheduler, loader, max_steps=target, accum=2, grad_clip=1,
                          lambdas={}, validate_fn=lambda step: {T.METRIC: 1 + step / 100000},
                          save_fn=save, emit=lambda row: None,
                          resume_state=checkpoint['resume_state'] if checkpoint else None)
    return wrapper, optimizer, scheduler, saved, result


class StageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        seed(71)
        cls.parent_wrapper, _, _, saved, _ = synthetic_run(30000)
        cls.parent = saved['last']

    def assert_tree_equal(self, a, b):
        if isinstance(a, torch.Tensor):
            self.assertTrue(torch.equal(a.cpu(), b.cpu()))
        elif isinstance(a, np.ndarray):
            np.testing.assert_array_equal(a, b)
        elif isinstance(a, dict):
            self.assertEqual(set(a), set(b))
            for key in a:
                self.assert_tree_equal(a[key], b[key])
        elif isinstance(a, (tuple, list)):
            self.assertEqual(len(a), len(b))
            for x, y in zip(a, b):
                self.assert_tree_equal(x, y)
        else:
            self.assertEqual(a, b)

    def test_parent_30k_first_update_preserves_lr_optimizer_rng_cursor(self):
        parent = self.parent
        self.assertEqual(parent['step'], 30000)
        state = parent['resume_state']
        self.assertEqual({float(v['step']) for v in state['optimizer_state_dict']['state'].values()}, {30000.})
        wrapper = TinyWrapper()
        wrapper.action_model.load_state_dict(parent['model_state_dict'])
        opt, scheduler = T.make_optimizer(wrapper.action_model, CFG['training'], T.CONTINUATION_SCHEDULE)
        T.restore_training_state(wrapper, opt, scheduler, state)
        self.assert_tree_equal(opt.state_dict(), state['optimizer_state_dict'])
        self.assert_tree_equal(scheduler.state_dict(), state['scheduler_state_dict'])
        self.assert_tree_equal(T.capture_rng('cpu'), state['rng'])
        self.assertEqual(opt.param_groups[0]['lr'], 5e-6)
        self.assertEqual(scheduler.last_epoch, 30000)
        result = synthetic_run(30001, parent, stage=True)
        self.assertEqual({float(v['step']) for v in result[1].state.values()}, {30001.})
        self.assertEqual(result[2].last_epoch, 30001)
        self.assertAlmostEqual(result[1].param_groups[0]['lr'], 5e-6 + 1.5e-5 / 500, places=15)
        self.assertEqual(len(result[0].order), 2)
        self.assertNotEqual(result[3]['last']['resume_state']['cursor'], state['cursor'])

    def test_segment_at_35000_vs_uninterrupted_to_50000_bitwise(self):
        seed(999)
        whole, opt, scheduler, saved, _ = synthetic_run(50000, self.parent, stage=True)
        rng_whole = T.capture_rng('cpu')
        seed(444)
        first, _, _, saved_first, report = synthetic_run(35000, self.parent, stage=True)
        self.assertEqual(report['completed_updates'], 35000)
        self.assertEqual(saved_first['last']['step'], 35000)
        self.assertIn('step_035000', saved_first)
        seed(222)
        order = first.order[:]
        boundary = saved_first['last']
        for target in (40000, 45000, 50000):
            second, opt2, scheduler2, saved2, report2 = synthetic_run(target, boundary, stage=True)
            order.extend(second.order)
            boundary = saved2['last']
            self.assertIn(f'step_{target:06d}', saved2)
        self.assertEqual(report2['completed_updates'], 50000)
        self.assertEqual(whole.order, order)
        self.assert_tree_equal(whole.action_model.state_dict(), second.action_model.state_dict())
        self.assert_tree_equal(opt.state_dict(), opt2.state_dict())
        self.assert_tree_equal(scheduler.state_dict(), scheduler2.state_dict())
        self.assert_tree_equal(rng_whole, T.capture_rng('cpu'))
        self.assert_tree_equal(saved['last']['resume_state'], saved2['last']['resume_state'])
        self.assertEqual({float(v['step']) for v in opt2.state.values()}, {50000.})
        self.assertEqual(scheduler2.last_epoch, 50000)
        self.assertAlmostEqual(opt2.param_groups[0]['lr'], 5e-6, places=15)
        self.assertEqual(saved2['last']['step'], 50000)

    def test_best_model_only_cannot_replace_full_last_or_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            wrapper, _, _, saved, _ = synthetic_run(30001, self.parent, stage=True)
            meta = copy.deepcopy(saved['last']); meta.pop('model_state_dict'); meta.pop('step')
            last = Path(T.save_checkpoint(tmp, 'last', wrapper.action_model, 30001, meta))
            last_sha = T.sha256(last)
            best = T.save_checkpoint(tmp, 'best', wrapper.action_model, 30001, {'config': CFG, 'mode': 'long'})
            self.assertEqual(last_sha, T.sha256(last))
            ck, _ = T.load_resume_source(last, CFG, 'long', 'cpu')
            self.assertEqual(ck['resume_state']['scheduler_state_dict']['last_epoch'], 30001)
            with self.assertRaisesRegex(ValueError, 'safe optimizer-boundary'):
                T.load_resume_source(best, CFG, 'long', 'cpu')
            with patch.object(torch, 'load', side_effect=AssertionError('must not unpickle')):
                with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
                    T.load_resume_source(last, CFG, 'long', 'cpu', '0' * 64)


class MigrationTests(unittest.TestCase):
    def test_real_plan_reconstructs_parent_contract_and_new_contract(self):
        ctx = context()
        contract = historical_contract(ctx)
        digest = T.continuation_contract(contract, ctx, {})
        self.assertNotEqual(digest, ctx['plan']['parent']['training_contract_sha256'])
        ck = {'continuation': T.continuation_identity(ctx), 'training_contract_sha256': digest}
        self.assertEqual(digest, T.continuation_contract(contract, ctx, ck))
        self.assertEqual(ctx['parent_manifest']['config'], CFG)

    def test_no_other_source_inventory_data_model_stats_workers_or_budget_changes(self):
        ctx = context(); base = historical_contract(ctx)
        cases = []
        for path in ('harness/wam_v3.py', 'harness/wam_behavior.py', 'harness/orchestrator.py',
                     'data/behavior-1k/2026-challenge-demos/wam_v3_stats_task000.json'):
            bad = copy.deepcopy(base); bad['sources'][str(ROOT / path)] = '0' * 64; cases.append(bad)
        bad = copy.deepcopy(base); bad['sources'].pop(next(iter(bad['sources']))); cases.append(bad)
        bad = copy.deepcopy(base); bad['sources']['/unknown_extra_source'] = 'a' * 64; cases.append(bad)
        bad = copy.deepcopy(base); bad['workers'] = 1; cases.append(bad)
        bad = copy.deepcopy(base); bad['train_indices'].reverse(); cases.append(bad)
        for section, field, value in [('training', 'max_steps', 50000), ('training', 'batch_size', 8),
                                     ('dataset', 'root', '/different'), ('model', 'proprio_mask', 'positions')]:
            bad = copy.deepcopy(base); bad['config'][section][field] = value; cases.append(bad)
        for bad in cases:
            with self.subTest(change=str(bad.keys())):
                with self.assertRaises(ValueError):
                    T.continuation_contract(bad, ctx, {})

    def test_plan_rejects_bad_hash_parent_source_schedule_and_config(self):
        original = json.loads(PLAN.read_text())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'plan.json'
            cases = []
            bad = copy.deepcopy(original); bad['source_migrations'][0]['to_sha256'] = '0' * 64; cases.append(bad)
            bad = copy.deepcopy(original); bad['source_migrations'][0]['from_sha256'] = '0' * 64; cases.append(bad)
            bad = copy.deepcopy(original); bad['source_migrations'][0]['path'] = 'harness/wam_v3.py'; cases.append(bad)
            bad = copy.deepcopy(original); bad['source_migrations'].append(bad['source_migrations'][0]); cases.append(bad)
            bad = copy.deepcopy(original); bad['parent']['sha256'] = '0' * 64; cases.append(bad)
            bad = copy.deepcopy(original); bad['schedule']['additional_steps'] = 30000; cases.append(bad)
            bad = copy.deepcopy(original); bad['parent']['manifest_sha256'] = '0' * 64; cases.append(bad)
            bad = copy.deepcopy(original); bad['config_overrides'] = {'batch_size': 32}; cases.append(bad)
            for bad in cases:
                T.atomic_json(path, bad)
                with self.assertRaises(ValueError):
                    T.read_continuation_plan(path, CFG, T.resolve(original['parent']['config']))
            T.atomic_json(path, original)
            changed = copy.deepcopy(CFG); changed['training']['learning_rate'] = 2e-5
            with self.assertRaisesRegex(ValueError, 'config/model/data/stats'):
                T.read_continuation_plan(path, changed, T.resolve(original['parent']['config']))

    def test_real_parent_cpu_mmap_verifies_full_boundary_without_cuda(self):
        ctx = context()
        ck, digest = T.load_resume_source(PARENT, CFG, 'long', 'cuda', T.CONTINUATION_PARENT_SHA)
        T.verify_continuation_checkpoint(ck, PARENT, digest, ctx)
        self.assertEqual(len(ck['resume_state']['optimizer_state_dict']['state']), 322)
        self.assertEqual(ck['resume_state']['cursor']['epoch'], 8)
        self.assertEqual(ck['resume_state']['cursor']['batch_skip'], 4000)
        self.assertFalse(torch.cuda.is_initialized())
        ck['step'] = 29999
        with self.assertRaisesRegex(ValueError, 'pinned parent'):
            T.verify_continuation_checkpoint(ck, PARENT, digest, ctx)
        ck['step'] = 30000
        ck['resume_state']['optimizer_state_dict']['param_groups'][0]['lr'] = 2e-5
        with self.assertRaisesRegex(ValueError, 'optimizer counts/LR'):
            T.verify_continuation_checkpoint(ck, PARENT, digest, ctx)

    def test_lr_history_unchanged_and_new_stage_separate_from_original_50k(self):
        model = TinyWrapper().action_model
        _, old = T.make_optimizer(model, CFG['training'])
        _, new = T.make_optimizer(model, CFG['training'], T.CONTINUATION_SCHEDULE)
        for step in (0, 1, 2000, 5000, 15000, 29999, 30000):
            self.assertEqual(old.lr_lambdas[0](step), new.lr_lambdas[0](step))
        self.assertEqual(T.continuation_lr(30000, T.CONTINUATION_SCHEDULE), 5e-6)
        self.assertEqual(T.continuation_lr(30500, T.CONTINUATION_SCHEDULE), 2e-5)
        self.assertEqual(T.continuation_lr(50000, T.CONTINUATION_SCHEDULE), 5e-6)
        legacy50 = copy.deepcopy(CFG['training']); legacy50.update(max_steps=50000, schedule_steps=50000)
        _, original50 = T.make_optimizer(model, legacy50)
        self.assertNotEqual(original50.lr_lambdas[0](30000), new.lr_lambdas[0](30000))
        for step in (29999, 50001):
            with self.assertRaises(ValueError):
                T.continuation_lr(step, T.CONTINUATION_SCHEDULE)

    def test_new_approval_required_pending_refused_no_auto_gpu(self):
        ctx = context()
        old = json.loads((ROOT / 'outputs/validation/task0_v3_audit_20260928/approval_30k_staged.json').read_text())
        args = T.parse_args(['--run-name', T.CONTINUATION_PREFIX + 's035000', '--mode', 'long',
                             '--continuation-plan', str(PLAN), '--config', ctx['plan']['parent']['config'],
                             '--resume-source', str(PARENT), '--max-steps', '35000', '--long-run-approval', '/unused'])
        self.assertEqual(T.validate_config(CFG, args), 35000)
        with self.assertRaisesRegex(ValueError, 'new continuation approval'):
            T.verify_continuation_approval(old, args, ctx)
        with tempfile.TemporaryDirectory() as tmp:
            evidence = Path(tmp) / 'synthetic-test-only'; evidence.write_text('not real approval; unit-test fixture')
            valid = {**old, 'continuation': T.continuation_identity(ctx), 'additional_steps': 20000,
                     'global_budget': 50000, 'scope': 'four_5k_segments',
                     'checks': {name: {'passed': True, 'path': str(evidence), 'sha256': T.sha256(evidence)}
                                for name in ('continuation_cpu', 'continuation_review', 'continuation_gpu_smoke')}}
            for target in (35000, 40000, 45000, 50000):
                args.max_steps = target
                T.verify_continuation_approval(valid, args, ctx)
            args.max_steps = 35000
            invalid = copy.deepcopy(valid); invalid['checks']['continuation_gpu_smoke']['passed'] = False
            with self.assertRaisesRegex(ValueError, 'not approved'):
                T.verify_continuation_approval(invalid, args, ctx)
            for target in (30001, 50001, 34000):
                args.max_steps = target
                with self.assertRaises(ValueError):
                    T.verify_continuation_approval(valid, args, ctx)
            smoke = {**valid, 'scope': 'gpu_smoke_only'}
            args.max_steps = 30002; T.verify_continuation_approval(smoke, args, ctx)
            args.max_steps = 35000
            with self.assertRaises(ValueError):
                T.verify_continuation_approval(smoke, args, ctx)
            pending = Path(tmp) / 'pending.json'
            T.atomic_json(pending, {**valid, 'approved': False})
            args.long_run_approval = str(pending)
            with patch.object(torch.cuda, 'set_per_process_memory_fraction', side_effect=AssertionError('CUDA forbidden')):
                with self.assertRaisesRegex(ValueError, 'approval is absent'):
                    T.verify_long_approval(args, old['config_sha256'], ctx)
        self.assertFalse(torch.cuda.is_initialized())

    def test_cli_legacy_defaults_and_continuation_require_explicit_resume_prefix_target(self):
        args = T.parse_args(['--run-name', 'legacy'])
        self.assertIsNone(args.continuation_plan)
        self.assertEqual(T.validate_config(CFG, args), 2000)
        args.continuation_plan = str(PLAN)
        with self.assertRaisesRegex(ValueError, 'long mode'):
            T.validate_config(CFG, args)
        args.mode = 'long'; args.resume_source = str(PARENT); args.long_run_approval = 'pending'
        with self.assertRaisesRegex(ValueError, 'independent run prefix'):
            T.validate_config(CFG, args)
        args.run_name = T.CONTINUATION_PREFIX + 'test'; args.max_steps = 35000
        self.assertEqual(T.validate_config(CFG, args), 35000)
        args.max_steps = 50001
        with self.assertRaisesRegex(ValueError, 'target must be'):
            T.validate_config(CFG, args)


if __name__ == '__main__':
    torch.set_num_threads(1)
    with patch.object(torch.cuda, '_lazy_init', side_effect=AssertionError('CPU-only suite forbids CUDA init')):
        unittest.main(verbosity=2)
