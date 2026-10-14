#!/usr/bin/env python3
"""CPU checks of staged updates and exact saved optimizer/RNG boundaries."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

os.environ['CUDA_VISIBLE_DEVICES'] = ''
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from scripts import train_zhuhai_wam_precision_v4 as trainer
from scripts.test_wam_precision_v4 import make_pair
from harness.zhuhai_precision_v4_training import FORMAT


SCHEDULE = dict(steps=8, predictor_steps=2, gate_steps=4, backbone_lr=1e-6, new_lr=1e-3)
TRAINING = dict(weight_decay=.01, betas=(.9, .99))
CONTRACT = dict(format=FORMAT, v4_training=dict(schedule=SCHEDULE))


def update(wrapper, optimizer, batch, rng, step):
    phase = trainer.phase_for(step, SCHEDULE['predictor_steps'], SCHEDULE['gate_steps'])
    trainer.configure_phase(optimizer, phase, step, SCHEDULE)
    optimizer.zero_grad(set_to_none=True)
    sample = dict(batch, action_sequence=batch['action_sequence'] + float(rng.uniform(-.01, .01)))
    loss, _, _ = wrapper(sample)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return float(loss.detach())


class TrainerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized())

    def test_stage_freeze_gate_learning_and_original_frozen_parameters(self):
        _, wrapper, batch = make_pair()
        model = wrapper.action_model
        model.output_proj.bias.requires_grad_(False)
        frozen = [(n, p) for n, p in model.named_parameters() if not p.requires_grad]
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        optimizer = trainer.make_optimizer(model, TRAINING, SCHEDULE)
        identities = [id(p) for g in optimizer.param_groups for p in g['params']]
        self.assertTrue(frozen)
        self.assertTrue(all(id(p) not in identities for _, p in frozen))
        rng = np.random.default_rng(41)
        update(wrapper, optimizer, batch, rng, 0)
        update(wrapper, optimizer, batch, rng, 1)
        self.assertEqual(float(model.chunk_vtt16.alpha), 0.)
        for name, p in model.named_parameters():
            if not name.startswith('chunk_vtt16.predictor.'):
                torch.testing.assert_close(p, before[name], rtol=0, atol=0)
        self.assertTrue(any(not torch.equal(p, before[n]) for n, p in model.named_parameters()
                            if n.startswith('chunk_vtt16.predictor.')))
        update(wrapper, optimizer, batch, rng, 2)
        self.assertNotEqual(float(model.chunk_vtt16.alpha), 0.)
        for name, p in model.named_parameters():
            if not name.startswith('chunk_vtt16.'):
                torch.testing.assert_close(p, before[name], rtol=0, atol=0)
        update(wrapper, optimizer, batch, rng, 3)
        update(wrapper, optimizer, batch, rng, 4)
        self.assertTrue(all(not p.requires_grad and torch.equal(p, before[n]) for n, p in frozen))
        self.assertEqual(identities, [id(p) for g in optimizer.param_groups for p in g['params']])
        self.assertTrue(any(not torch.equal(p, before[n]) for n, p in model.named_parameters()
                            if not n.startswith('chunk_vtt16.') and p.requires_grad))

    def test_full_state_resume_next_update_exact_at_both_stage_boundaries(self):
        for boundary in (2, 4):
            with self.subTest(boundary=boundary), tempfile.TemporaryDirectory() as directory:
                out = Path(directory)
                _, original, batch = make_pair()
                optimizer = trainer.make_optimizer(original.action_model, TRAINING, SCHEDULE)
                rng = np.random.default_rng(51)
                for step in range(boundary):
                    update(original, optimizer, batch, rng, step)
                with patch('torch.cuda.get_rng_state_all', return_value=[]):
                    trainer.save_checkpoint(out, original, CONTRACT, boundary,
                                            optimizer=optimizer, rng=rng.bit_generator.state)
                expected_loss = update(original, optimizer, batch, rng, boundary)
                saved = trainer.load_resume(out, CONTRACT)
                _, resumed, _ = make_pair()
                resumed.action_model.load_state_dict(saved['model_state_dict'], strict=True)
                resumed_optimizer = trainer.make_optimizer(resumed.action_model, TRAINING, SCHEDULE)
                resumed_rng = np.random.default_rng(91)
                with patch('torch.cuda.set_rng_state_all'):
                    trainer.restore_state(resumed_optimizer, resumed_rng, saved)
                actual_loss = update(resumed, resumed_optimizer, batch, resumed_rng, boundary)
                self.assertEqual(actual_loss, expected_loss)
                for name, value in original.action_model.state_dict().items():
                    torch.testing.assert_close(value, resumed.action_model.state_dict()[name], rtol=0, atol=0)
                expected = optimizer.state_dict()
                actual = resumed_optimizer.state_dict()
                self.assertEqual(expected['param_groups'], actual['param_groups'])
                for key, states in expected['state'].items():
                    for name, value in states.items():
                        torch.testing.assert_close(value, actual['state'][key][name], rtol=0, atol=0)

    def test_immutable_milestone_rejects_a_different_model(self):
        with tempfile.TemporaryDirectory() as directory:
            _, wrapper, _ = make_pair()
            out = Path(directory)
            path = trainer.save_checkpoint(out, wrapper, CONTRACT, 0)
            original = path.read_bytes()
            self.assertEqual(trainer.save_checkpoint(out, wrapper, CONTRACT, 0), path)
            with torch.no_grad():
                wrapper.action_model.chunk_vtt16.alpha.add_(.1)
            with self.assertRaisesRegex(ValueError, 'current model boundary'):
                trainer.save_checkpoint(out, wrapper, CONTRACT, 0)
            self.assertEqual(path.read_bytes(), original)

    def test_interrupted_tail_preserved_before_log_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            line = lambda n: (json.dumps(dict(step=80000+n, updates=n)) + '\n').encode()
            committed = line(1) + line(2)
            tail = line(3) + b'{"updates":4'
            path = out / 'log.jsonl'; path.write_bytes(committed + tail)
            trainer.reconcile_resume_log(out, 2)
            self.assertEqual(path.read_bytes(), committed)
            archives = list(out.glob('log_uncommitted_after_*.jsonl'))
            self.assertEqual(len(archives), 1)
            self.assertEqual(archives[0].read_bytes(), tail)
            trainer.reconcile_resume_log(out, 2)
            path.write_bytes(line(1))
            with self.assertRaisesRegex(ValueError, 'missing committed'):
                trainer.reconcile_resume_log(out, 2)
            self.assertEqual(path.read_bytes(), line(1))

    def test_interrupted_receipt_recovers_previous_optimizer_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            _, wrapper, batch = make_pair()
            optimizer = trainer.make_optimizer(wrapper.action_model, TRAINING, SCHEDULE)
            rng = np.random.default_rng(51)
            update(wrapper, optimizer, batch, rng, 0)
            with patch('torch.cuda.get_rng_state_all', return_value=[]):
                trainer.save_checkpoint(out, wrapper, CONTRACT, 1,
                    optimizer=optimizer, rng=rng.bit_generator.state)
            expected = trainer.load_resume(out, CONTRACT)
            update(wrapper, optimizer, batch, rng, 1)
            write = trainer.v2.write_json
            def interrupt(path, value, **kwargs):
                if path.name == 'checkpoint_last.pt.sha256.json':
                    raise OSError('simulated crash before receipt commit')
                return write(path, value, **kwargs)
            with patch('torch.cuda.get_rng_state_all', return_value=[]), \
                    patch.object(trainer.v2, 'write_json', side_effect=interrupt):
                with self.assertRaisesRegex(OSError, 'simulated crash'):
                    trainer.save_checkpoint(out, wrapper, CONTRACT, 2,
                        optimizer=optimizer, rng=rng.bit_generator.state)
            recovered = trainer.load_resume(out, CONTRACT)
            self.assertEqual(recovered['updates'], 1)
            self.assertEqual(len(recovered['resume_recovery']['preserved']), 2)
            for key, value in expected['model_state_dict'].items():
                torch.testing.assert_close(recovered['model_state_dict'][key], value, rtol=0, atol=0)
            wrapper.action_model.load_state_dict(recovered['model_state_dict'])
            with patch('torch.cuda.set_rng_state_all'):
                trainer.restore_state(optimizer, rng, recovered)
            with patch('torch.cuda.get_rng_state_all', return_value=[]):
                trainer.save_checkpoint(out, wrapper, CONTRACT, 1,
                    optimizer=optimizer, rng=rng.bit_generator.state)
            self.assertEqual(trainer.load_resume(out, CONTRACT)['updates'], 1)

    def test_diagnostic_reports_flow_mse_and_never_samples_heldout(self):
        class Dataset:
            def __len__(self): return 64
            def __getitem__(self, index): return dict(episode='episode', anchor=index)
        class Wrapper(torch.nn.Module):
            device = torch.device('cpu')
            def forward(self, batch):
                return torch.tensor(1.), dict(mse=.25, chunk_vtt_loss=.5), {}
        wrapper = Wrapper()
        ds = Dataset()
        result = trainer.diagnostic(wrapper, dict(train={trainer.v2.TASK: ds},
            val={trainer.v2.TASK: ds}, heldout={}), 2)
        self.assertEqual(set(result), {'train', 'val'})
        self.assertEqual(result['val']['means']['mse'], .25)
        self.assertTrue(wrapper.training)

    def test_missing_milestone_receipt_recovers_from_committed_full_state(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            _, wrapper, batch = make_pair()
            optimizer = trainer.make_optimizer(wrapper.action_model, TRAINING, SCHEDULE)
            rng = np.random.default_rng(51)
            for step in range(2):
                update(wrapper, optimizer, batch, rng, step)
            with patch('torch.cuda.get_rng_state_all', return_value=[]):
                trainer.save_checkpoint(out, wrapper, CONTRACT, 2,
                    optimizer=optimizer, rng=rng.bit_generator.state)
            with patch.object(trainer.v2, 'write_json', side_effect=OSError('receipt interrupted')):
                with self.assertRaisesRegex(OSError, 'receipt interrupted'):
                    trainer.save_checkpoint(out, wrapper, CONTRACT, 2)
            path = out / 'checkpoint_step_080002.pt'
            original = path.read_bytes()
            saved = trainer.load_resume(out, CONTRACT)
            _, resumed, _ = make_pair()
            resumed.action_model.load_state_dict(saved['model_state_dict'], strict=True)
            trainer.save_checkpoint(out, resumed, CONTRACT, 2)
            self.assertEqual(path.read_bytes(), original)
            receipt = path.with_name(path.name + '.sha256.json')
            self.assertTrue(json.loads(receipt.read_text())['recovered_missing_receipt'])
            receipt.unlink()
            with torch.no_grad():
                resumed.action_model.chunk_vtt16.alpha.add_(.1)
            with self.assertRaisesRegex(ValueError, 'current model boundary'):
                trainer.save_checkpoint(out, resumed, CONTRACT, 2)
            self.assertFalse(receipt.exists())
            self.assertEqual(path.read_bytes(), original)

    def test_final_saved_boundary_can_finish_after_status_write_interruption(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            contract = dict(CONTRACT, source_files={trainer.__file__: trainer.v2.sha256_file(trainer.__file__)})
            _, wrapper, batch = make_pair()
            optimizer = trainer.make_optimizer(wrapper.action_model, TRAINING, SCHEDULE)
            rng = np.random.default_rng(51)
            for step in range(SCHEDULE['steps']):
                update(wrapper, optimizer, batch, rng, step)
            with patch('torch.cuda.get_rng_state_all', return_value=[]):
                trainer.save_checkpoint(out, wrapper, contract, 8,
                    optimizer=optimizer, rng=rng.bit_generator.state)
            write = trainer.v2.write_json
            def interrupt(path, value, **kwargs):
                if path.name == 'status.json':
                    raise OSError('status interrupted')
                return write(path, value, **kwargs)
            with patch.object(trainer, 'diagnostic', return_value={'checked': True}), \
                    patch.object(trainer.v2, 'write_json', side_effect=interrupt), patch.object(trainer, 'log'):
                with self.assertRaisesRegex(OSError, 'status interrupted'):
                    trainer.finish_run(out, wrapper, contract, 8, {}, 1)
            saved = trainer.load_resume(out, contract)
            self.assertEqual(saved['updates'], 8)
            before = {name: p.clone() for name, p in wrapper.action_model.state_dict().items()}
            torch_rng = torch.get_rng_state().clone()
            with patch.object(trainer, 'diagnostic', side_effect=AssertionError('do not repeat completed diagnostic')), \
                    patch.object(trainer, 'log'):
                trainer.finish_run(out, wrapper, contract, saved['updates'], {}, 1)
            status = json.loads((out / 'status.json').read_text())
            self.assertEqual((status['status'], status['updates'], status['step']), ('completed', 8, 80008))
            torch.testing.assert_close(torch.get_rng_state(), torch_rng, rtol=0, atol=0)
            for name, value in wrapper.action_model.state_dict().items():
                torch.testing.assert_close(value, before[name], rtol=0, atol=0)

    def test_resume_rejects_changed_optimizer_group_or_nonfinite_moments(self):
        _, wrapper, batch = make_pair()
        optimizer = trainer.make_optimizer(wrapper.action_model, TRAINING, SCHEDULE)
        rng = np.random.default_rng(4)
        update(wrapper, optimizer, batch, rng, 0)
        saved = dict(optimizer_state_dict=copy.deepcopy(optimizer.state_dict()),
                     rng=copy.deepcopy(rng.bit_generator.state), torch_rng=torch.get_rng_state(), cuda_rng=[])
        saved['optimizer_state_dict']['param_groups'][0]['base_lr'] *= 10
        with self.assertRaisesRegex(ValueError, 'parameter groups'):
            trainer.restore_state(optimizer, rng, saved)
        saved['optimizer_state_dict'] = copy.deepcopy(optimizer.state_dict())
        next(iter(saved['optimizer_state_dict']['state'].values()))['exp_avg'].fill_(float('nan'))
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            trainer.restore_state(optimizer, rng, saved)


if __name__ == '__main__':
    unittest.main()
