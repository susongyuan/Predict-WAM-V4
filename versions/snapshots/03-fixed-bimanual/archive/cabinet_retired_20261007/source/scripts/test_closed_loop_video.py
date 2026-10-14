#!/usr/bin/env python3
"""CPU-only tests: fake ffprobe/ffmpeg, temporary tiny files, no old video decode."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
import check_closed_loop_video as checker


class VideoTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.run = self.base / 'historical_run'
        (self.run / 'eval/json').mkdir(parents=True)
        (self.run / 'eval/videos').mkdir()
        self.row = {'task': 'loading_the_car', 'instance_id': 311, 'rollout_id': 0,
                    'steps': 3, 'success': False}
        self.episode = self.run / 'eval/json/loading_the_car_311_0.json'
        self.video = self.run / 'eval/videos/loading_the_car_311_0.mp4'
        self.episode.write_text(json.dumps(self.row))
        self.video.write_bytes(b'fake-video-not-decoded')
        self.sidecar = self.video.with_name(self.video.name + '.json')
        self.metadata = {'status': 'completed', 'view': 'viewer', 'environment_steps': 3,
                         'frames': 4, 'success': False, 'complete_frame_coverage': True}
        self.sidecar.write_text(json.dumps(self.metadata))
        self.stream = {'width': 1280, 'height': 720, 'nb_read_frames': '4'}
        self.decode_rc, self.decode_stderr = 0, ''
        self.calls = []
        self.mock = patch.object(checker.subprocess, 'run', side_effect=self.fake)
        self.mock.start()
        self.addCleanup(self.mock.stop)

    def fake(self, argv, **kwargs):
        self.calls.append(argv)
        self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'], '')
        self.assertEqual(argv[argv.index('-threads') + 1], '1')
        self.assertGreater(kwargs['timeout'], 0)
        if argv[0] == 'ffprobe':
            self.assertIn('-count_frames', argv)
            self.assertEqual(argv[-1], str(self.video))
            return subprocess.CompletedProcess(argv, 0, json.dumps({'streams': [self.stream]}), '')
        self.assertEqual(argv[0], 'ffmpeg')
        self.assertEqual(argv[argv.index('-hwaccel') + 1], 'none')
        self.assertIn('-xerror', argv)
        self.assertEqual(argv[argv.index('-i') + 1], str(self.video))
        return subprocess.CompletedProcess(argv, self.decode_rc, '', self.decode_stderr)

    def check(self):
        return checker.check_run(self.run)

    def test_failed_task_can_have_verified_video_and_run_is_unchanged(self):
        before = {p: p.read_bytes() for p in self.run.rglob('*') if p.is_file()}
        result = self.check()
        self.assertTrue(result['all_verified'])
        self.assertEqual(result['episodes'][0]['expected_frames'], 4)
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(before, {p: p.read_bytes() for p in self.run.rglob('*') if p.is_file()})

    def test_missing_or_mismatched_sidecar_fails(self):
        for key, value in (('status', 'interrupted'), ('view', 'robot'), ('environment_steps', 2),
                           ('frames', 3), ('complete_frame_coverage', False), ('success', True),
                           ('frames', True), ('environment_steps', True)):
            self.sidecar.write_text(json.dumps({**self.metadata, key: value}))
            self.assertFalse(self.check()['all_verified'], (key, value))
        self.sidecar.unlink()
        self.assertFalse(self.check()['all_verified'])

    def test_pyav_full_decode_and_malformed_result(self):
        valid = {'width':1280, 'height':720, 'frames':4, 'pyav_version':'fixture'}
        for change, expected in (({}, True), ({'frames':3}, False), ({'height':448}, False), ({'frames':True}, False)):
            result = subprocess.CompletedProcess([], 0, json.dumps({**valid, **change}), '')
            with patch.object(checker, '_run', return_value=result) as run:
                report = checker.check_run(self.run, pyav_python='/existing/python')
                self.assertEqual(report['all_verified'], expected)
                self.assertIn('container.decode(stream)', run.call_args[0][0][3])
        result = subprocess.CompletedProcess([], 1, '', 'corrupt bitstream')
        with patch.object(checker, '_run', return_value=result):
            self.assertFalse(checker.check_run(self.run, pyav_python='/existing/python')['all_verified'])

    def test_higher_resolution_passes(self):
        self.stream.update(width=1920, height=1080)
        self.assertTrue(self.check()['all_verified'])

    def test_lower_dimension_fails(self):
        for width, height in ((1279, 720), (1280, 719)):
            self.stream.update(width=width, height=height)
            self.assertIn('dimensions_below_1280x720', self.check()['episodes'][0]['issues'])

    def test_frame_counts_missing_wrong_or_header_only(self):
        for count in ('N/A', None, '3', '5', True, '4.0'):
            self.stream['nb_read_frames'] = count
            self.stream['nb_frames'] = '4'
            self.assertFalse(self.check()['all_verified'], count)

    def test_decode_error_nonzero_or_error_stderr(self):
        for rc, err in ((1, ''), (0, 'corrupt frame')):
            self.decode_rc, self.decode_stderr = rc, err
            self.assertFalse(self.check()['all_verified'])

    def test_no_fallback_to_unrelated_video(self):
        self.video.rename(self.video.with_name('unrelated.mp4'))
        result = self.check()
        self.assertFalse(result['all_verified'])
        self.assertIn('missing_or_empty_bound_video', result['episodes'][0]['issues'])
        self.assertFalse(self.calls)

    def test_filename_binding_mismatch(self):
        self.episode.rename(self.episode.with_name('other.json'))
        self.assertIn('episode_filename_identity_mismatch', self.check()['episodes'][0]['issues'])
        self.assertFalse(self.calls)

    def test_duplicate_identity_invalidates_both(self):
        self.episode.with_name('duplicate.json').write_text(json.dumps(self.row))
        result = self.check()
        self.assertFalse(result['all_verified'])
        self.assertTrue(all('duplicate_episode_identity' in e['issues'] for e in result['episodes']))
        self.assertFalse(self.calls)

    def test_invalid_identity_and_steps(self):
        for key, value in (('task', '../escape'), ('instance_id', True), ('rollout_id', None),
                           ('steps', True), ('steps', 0), ('steps', None)):
            self.episode.write_text(json.dumps({**self.row, key: value}))
            self.assertFalse(self.check()['all_verified'])
        self.assertFalse(self.calls)

    def test_missing_tools_timeout_and_bad_json(self):
        for error in (FileNotFoundError('ffprobe'), subprocess.TimeoutExpired('ffprobe', 1)):
            with patch.object(checker.subprocess, 'run', side_effect=error):
                self.assertFalse(self.check()['all_verified'])
        self.episode.write_text('{')
        self.assertFalse(self.check()['all_verified'])

    def test_decode_tool_missing(self):
        probe = subprocess.CompletedProcess([], 0, json.dumps({'streams': [self.stream]}), '')
        with patch.object(checker.subprocess, 'run', side_effect=[probe, FileNotFoundError('ffmpeg')]):
            self.assertFalse(self.check()['all_verified'])

    def test_probe_errors_and_missing_stream(self):
        for rc, stdout, stderr in ((1, '{}', ''), (0, '{}', 'decode error'),
                                   (0, '{', ''), (0, '{"streams": []}', '')):
            with patch.object(checker.subprocess, 'run', return_value=subprocess.CompletedProcess([], rc, stdout, stderr)):
                self.assertFalse(self.check()['all_verified'])

    def test_no_episodes_is_not_verified(self):
        self.episode.unlink()
        self.assertEqual(self.check()['status'], 'not_verified')
        self.assertFalse(self.calls)

    def test_cli_stdout_and_explicit_new_output(self):
        with contextlib.redirect_stdout(io.StringIO()) as stdout:
            self.assertEqual(checker.main([str(self.run)]), 0)
        self.assertTrue(json.loads(stdout.getvalue())['all_verified'])
        target = self.base / 'report.json'
        self.assertEqual(checker.main([str(self.run), '--out', str(target)]), 0)
        self.assertTrue(json.loads(target.read_text())['all_verified'])
        for bad in (target, self.run / 'summary.json'):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                checker.main([str(self.run), '--out', str(bad)])
        self.assertFalse((self.run / 'summary.json').exists())

    def test_not_verified_report_exit_code(self):
        self.video.unlink()
        with contextlib.redirect_stdout(io.StringIO()) as stdout:
            self.assertEqual(checker.main([str(self.run)]), 2)
        self.assertEqual(json.loads(stdout.getvalue())['status'], 'not_verified')

    def test_external_symlink_is_not_bound(self):
        external = self.base / 'external.mp4'
        external.write_bytes(b'fake')
        self.video.unlink()
        self.video.symlink_to(external)
        self.assertIn('video_resolves_outside_run', self.check()['episodes'][0]['issues'])
        self.assertFalse(self.calls)


if __name__ == '__main__':
    unittest.main()
