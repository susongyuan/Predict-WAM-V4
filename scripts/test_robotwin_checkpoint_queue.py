import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from scripts.run_robotwin_checkpoint_queue import command, main, unit_state

class QueueTests(unittest.TestCase):
    def test_offscreen_entry_is_explicit_and_whitelisted(self):
        j=dict(training_run_dir='/tmp/training-new-task',unit='cpu-test',run_dir='/tmp/new',checkpoint='/tmp/ckpt',seed=10000)
        self.assertIn('scripts/eval_robotwin_wam.py', command(j))
        j['evaluator_entry']='scripts/eval_robotwin_offscreen.py'
        self.assertIn(j['evaluator_entry'],command(j))
        j['evaluator_entry']='scripts/unknown.py'
        with self.assertRaisesRegex(ValueError,'unsupported evaluator entry'):command(j)
    def test_child_command_protocol(self):
        j=dict(training_run_dir='/tmp/training-new-task',unit='cpu-test',run_dir='/tmp/new',checkpoint='/tmp/ckpt',seed=10000)
        c=command(j)
        for token in ['--property=RuntimeMaxSec=1560','--property=TimeoutStopSec=30','--property=KillMode=control-group','700','none','120']:
            self.assertIn(token,c)
        self.assertEqual(c[c.index('--seeds')+1],'10000')
        self.assertEqual(c[c.index('--system-mode')+1], 'pure')
    def test_transparent_onboard_command(self):
        j=dict(training_run_dir='/tmp/training-new-task',unit='cpu-test-system',run_dir='/tmp/new',checkpoint='/tmp/ckpt',seed=10010,
               system_mode='transparent',record_onboard=True)
        c=command(j)
        self.assertEqual(c[c.index('--system-mode')+1], 'transparent')
        self.assertIn('--record-onboard', c)
    def test_unknown_mode_rejected(self):
        with self.assertRaisesRegex(ValueError, 'unsupported system_mode'):
            command(dict(training_run_dir='/tmp/training-new-task',unit='cpu-test', run_dir='/tmp/new', checkpoint='/tmp/ckpt', seed=10010,
                         system_mode='full'))
    def test_middle_vtt_artifact_and_serial_readback_command(self):
        j=dict(training_run_dir='/tmp/training-new-task',unit='cpu-test-middle',run_dir='/tmp/new',checkpoint='/tmp/ckpt',seed=10010,
               system_mode='middle',record_onboard=True,vtt_adapter='/tmp/adapter.pt',
               vtt_bank='/tmp/bank',camera_readback='serial',execution_mode='native8')
        c=command(j)
        for option,value in [('--system-mode','middle'),('--vtt-adapter','/tmp/adapter.pt'),
                             ('--vtt-bank','/tmp/bank'),('--camera-readback','serial')]:
            self.assertEqual(c[c.index(option)+1],value)
    def test_existing_run_refused_before_launch(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);m=p/'manifest.json';m.write_text(json.dumps({'jobs':[{'unit':'x','run_dir':d,'checkpoint':str(p/'missing')}]}))
            with patch('scripts.run_robotwin_checkpoint_queue.subprocess.run') as run:
                with self.assertRaisesRegex(ValueError,'existing run'):main(['--manifest',str(m),'--execute'])
                run.assert_not_called()
    def test_duplicate_unit_refused(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);ck=p/'c';ck.touch();j=dict(training_run_dir='/tmp/training-new-task',unit='x',run_dir=str(p/'new'),checkpoint=str(ck),seed=10000);m=p/'m';m.write_text(json.dumps({'jobs':[j,j]}))
            with self.assertRaisesRegex(ValueError,'duplicate unit'):main(['--manifest',str(m)])
    def test_parse_inactive_not_success_assumption(self):
        with patch('scripts.run_robotwin_checkpoint_queue.subprocess.run') as run:
            run.return_value.stdout='ActiveState=failed\nResult=timeout\nMainPID=0\n'
            self.assertEqual(unit_state('x')['Result'],'timeout')
    def test_invalid_status_filename_refused(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);ck=p/'c';ck.touch();m=p/'m.json'
            m.write_text(json.dumps({'queue_status_filename':'../escape.json','jobs':[
                dict(training_run_dir='/tmp/training-new-task',unit='x',run_dir=str(p/'new'),checkpoint=str(ck),seed=10000)]}))
            with self.assertRaisesRegex(ValueError,'queue_status_filename'):
                main(['--manifest',str(m),'--execute'])

if __name__=='__main__':unittest.main()
