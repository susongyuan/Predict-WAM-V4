"""CPU fixtures only; no real services, GPU libraries, or checkpoints."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from scripts import monitor_wam_v3_stages as m


class MonitorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.out = self.root / 'outputs/training/audit_stages'
        self.train = self.root / 'outputs/training/audit_s005000'
        self.phys = self.root / 'outputs/behavior/closed_loop/audit_s005000_dev10_k1'
        self.put(self.out / 'plan.json', {'total_optimizer_steps': 30000,
                 'checkpoints': list(range(5000, 30001, 5000)), 'physical': {'instance': 10}})
        self.status('attach_training')
        self.train.mkdir(parents=True)
        (self.train / 'log.jsonl').write_text('{"step":4900,"event":"train"}\n')

    def put(self, path, obj):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj))

    def status(self, phase, **extra):
        self.put(self.out / 'status.json', {'phase': phase, 'step_target': 5000,
                 'training_name': 'audit_s005000', **extra})

    def sample(self, available=True):
        return m.sample(self.root, 'audit', {'available': available})

    def test_training5k_progress(self):
        row = self.sample()
        self.assertEqual(row['training']['observed_step'], 4900)
        self.assertEqual(row['training']['step_target'], 5000)
        self.assertEqual(row['training']['total_optimizer_steps'], 30000)
        self.assertFalse(row['budget_complete'])
        self.assertFalse(row['terminal'])

    def test_physical_pending_never_infers_success(self):
        self.status('physical_test')
        self.put(self.train / 'status.json', {'status': 'budget_complete'})
        self.put(self.phys / 'manifest.json', {'evaluation_complete': False})
        self.put(self.phys / 'eval/policy_metadata.json', {'device': 'cuda'})
        (self.phys / 'resources.jsonl').write_text('{"gpu":[{"vram_used_mib":13449}]}\n')
        row = self.sample()
        self.assertFalse(row['budget_complete'])
        self.assertFalse(row['physical']['evaluation_complete_reported'])
        self.assertNotIn('success', row['physical'])
        self.assertEqual(row['physical']['resources_last']['gpu'][0]['vram_used_mib'], 13449)

    def test_controller_disappearance(self):
        row = self.sample(False)
        self.assertEqual(row['phase'], 'attention_required')
        self.assertEqual(row['exit_code'], 2)
        self.assertTrue(row['terminal'])
        self.assertFalse(row['budget_complete'])

    def test_infrastructure_terminal(self):
        self.status('stopped_infrastructure_failure', error='CUDA out of memory')
        row = self.sample(False)
        self.assertEqual(row['phase'], 'stopped_infrastructure_failure')
        self.assertEqual(row['exit_code'], 3)
        self.assertTrue(row['terminal'])

    def test_budget_requires_status_and_plan_agreement(self):
        for steps in [5000, '30000', None]:
            self.status('budget_complete', steps=steps)
            row = self.sample(False)
            self.assertFalse(row['budget_complete'])
            self.assertEqual(row['phase'], 'attention_required')
        self.status('budget_complete', steps=30000)
        row = self.sample(False)
        self.assertTrue(row['budget_complete'])
        self.assertEqual(row['exit_code'], 0)
        self.put(self.out / 'plan.json', {'total_optimizer_steps': 30000, 'checkpoints': [5000]})
        self.assertFalse(self.sample()['budget_complete'])

    def test_partial_tail_and_bounded_read(self):
        path = self.train / 'log.jsonl'
        path.write_bytes(b'x' * 200000 + b'\n{"step":4999}\n{"step":5000')
        self.assertEqual(m.tail_json(path, 128), {'step': 4999})
        path.write_bytes(b'{"step":1}\ninvalid\n{"step":2}\n')
        self.assertEqual(m.tail_json(path), {'step': 2})

    def test_atomic_snapshot_append_only_deduplicated(self):
        row = self.sample()
        m.publish(self.out, row)
        events = self.out / 'monitor_events.jsonl'
        original = events.read_bytes()
        m.publish(self.out, self.sample())
        self.assertEqual(events.read_bytes(), original)
        self.status('physical_test')
        m.publish(self.out, self.sample())
        self.assertTrue(events.read_bytes().startswith(original))
        self.assertEqual(len(events.read_text().splitlines()), 2)
        self.assertEqual(json.loads((self.out / 'monitor_status.json').read_text())['phase'], 'physical_test')
        self.assertFalse(list(self.out.glob('.monitor_status.*')))

    def test_systemctl_show_only_and_prefix_binding(self):
        stdout = ('Id=p2-v3-30k-staged-20260929.service\nLoadState=loaded\n'
                  'ActiveState=active\nMainPID=123\nExecStart={ argv[]=python3 '
                  'scripts/run_wam_v3_stages.py --prefix audit --continue-existing ; }\n')
        with patch.object(m.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, stdout, '')) as run:
            self.assertTrue(m.controller_state('audit')['available'])
            self.assertFalse(m.controller_state('different')['available'])
            self.assertEqual(run.call_args.args[0][:3], ['systemctl', '--user', 'show'])
        with patch.object(m.subprocess, 'run', side_effect=subprocess.TimeoutExpired('systemctl', 10)):
            self.assertFalse(m.controller_state('audit')['available'])

    def test_terminal_and_missing_controller_exit_without_sleep(self):
        for phase, available, expected in [('physical_test', False, 2),
                                           ('stopped_infrastructure_failure', False, 3),
                                           ('budget_complete', False, 0)]:
            with self.subTest(phase=phase):
                self.status(phase, steps=30000)
                with patch.object(m, 'ROOT', self.root), patch.object(m, 'controller_state', return_value={'available': available}), patch.object(m.time, 'sleep') as sleep:
                    self.assertEqual(m.main(['--prefix', 'audit']), expected)
                    sleep.assert_not_called()
                snapshot = json.loads((self.out / 'monitor_status.json').read_text())
                self.assertTrue(snapshot['terminal'])

    def test_training_completion_event_and_partial_event_history(self):
        m.publish(self.out, self.sample())
        events = self.out / 'monitor_events.jsonl'
        with events.open('ab') as f:
            f.write(b'{"interrupted":')
        original = events.read_bytes()
        self.put(self.train / 'status.json', {'status': 'budget_complete'})
        m.publish(self.out, self.sample())
        self.assertTrue(events.read_bytes().startswith(original))
        self.assertEqual(m.tail_json(events)['phase'], 'attach_training')
        self.assertFalse(self.sample()['budget_complete'])

    def test_once_writes_and_does_not_sleep(self):
        with patch.object(m, 'ROOT', self.root), patch.object(m, 'controller_state', return_value={'available': True}), patch.object(m.time, 'sleep') as sleep:
            self.assertEqual(m.main(['--prefix', 'audit', '--once']), 0)
            sleep.assert_not_called()
        self.assertTrue((self.out / 'monitor_status.json').exists())


if __name__ == '__main__':
    unittest.main()
