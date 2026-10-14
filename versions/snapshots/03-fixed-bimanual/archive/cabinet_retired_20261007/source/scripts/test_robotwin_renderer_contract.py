#!/usr/bin/env python3
"""CPU-only renderer/stop contracts from live production ASTs.

Run: python3 -B scripts/test_robotwin_renderer_contract.py
Standard library only; no production imports, Scene, GPU, files, or downloads.
AST slices deliberately test renderer wiring, not full simulator initialization.
"""
import argparse
import ast
import contextlib
import copy
import io
from pathlib import Path
import signal
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / 'scripts/eval_robotwin_wam.py'
BASE = ROOT / 'third_party/RoboTwin/envs/_base_task.py'
EVIDENCE = ROOT / 'harness/robotwin_eval_evidence.py'


def node(path, name):
    matches = [n for n in ast.walk(ast.parse(path.read_text()))
               if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name]
    if len(matches) != 1:
        raise AssertionError(f'{path}: expected exactly one {name}, got {len(matches)}')
    return copy.deepcopy(matches[0])


def execute(nodes, path, namespace):
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    exec(compile(module, str(path), 'exec'), namespace)
    return namespace


def expression(value, namespace):
    return eval(compile(ast.Expression(value), '<live production expression>', 'eval'), namespace)


def field(dictionary, key):
    matches = [v for k, v in zip(dictionary.keys, dictionary.values)
               if isinstance(k, ast.Constant) and k.value == key]
    if len(matches) != 1:
        raise AssertionError(f'expected one dictionary field {key}')
    return matches[0]


class RendererContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parse = staticmethod(execute([node(EVAL, 'validate_pure'), node(EVAL, 'parse_args')], EVAL,
                                        {'argparse': argparse, '__doc__': '', 'InfrastructureError': RuntimeError})['parse_args'])

    def args(self, *extra):
        # Parse only: these sentinel paths are never opened or created.
        return self.parse(['--run-dir', 'unused-output', '--checkpoint', 'unused-checkpoint',
                           '--training-run-dir', 'unused-training', '--plan-only', *extra])

    def test_cli_default(self):
        self.assertEqual(self.args().render_denoiser, 'oidn')

    def test_cli_explicit_choices(self):
        for mode in ('oidn', 'none', 'optix'):
            with self.subTest(mode=mode):
                self.assertEqual(self.args('--render-denoiser', mode).render_denoiser, mode)

    def test_cli_rejects_invalid_choices(self):
        for mode in ('auto', 'OIDN', '', 'cuda'):
            with self.subTest(mode=mode), contextlib.redirect_stderr(io.StringIO()) as err:
                with self.assertRaises(SystemExit) as raised:
                    self.args('--render-denoiser', mode)
                self.assertEqual(raised.exception.code, 2)
                self.assertIn('invalid choice', err.getvalue())
                self.assertIn('--render-denoiser', err.getvalue())

    def test_plan_renderer_and_task_kwargs_agree(self):
        plan = node(EVAL, 'plan_evaluation')
        assignments = [n for n in plan.body if isinstance(n, ast.Assign)
                       and any(ast.unparse(t) == "config['render_denoiser']" for t in n.targets)]
        self.assertEqual(len(assignments), 1)
        result = next(n.value for n in plan.body if isinstance(n, ast.Return))
        for mode in (None, 'oidn', 'none', 'optix'):
            with self.subTest(mode=mode):
                args = SimpleNamespace() if mode is None else SimpleNamespace(render_denoiser=mode)
                ns = {'args': args, 'config': {}}
                execute(assignments, EVAL, ns)
                renderer = expression(field(result, 'renderer'), ns)
                config = expression(field(result, 'task_kwargs'), ns)
                expected = mode or 'oidn'
                self.assertIs(config, ns['config'])
                self.assertEqual(config['render_denoiser'], expected)
                self.assertEqual(renderer, {'shader': 'rt', 'samples_per_pixel': 32,
                                           'path_depth': 8, 'denoiser': expected,
                                           'native_default': 'oidn',
                                           'protocol_deviation': expected != 'oidn'})

    def init_renderer(self, kwargs):
        method = node(BASE, '_init_task_env_')
        # Execute the actual contiguous assignment/validation/forwarding block,
        # excluding seed initialization, torch and all native simulator work.
        start = next(i for i, n in enumerate(method.body) if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == 'denoiser' for t in n.targets))
        calls = [(i, n) for i, n in enumerate(method.body)
                 if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                 and ast.unparse(n.value.func) == 'self.setup_scene']
        self.assertEqual(len(calls), 1)
        end, call = calls[0]
        self.assertGreater(end, start)
        self.assertEqual([k.arg for k in call.value.keywords], ['render_denoiser'])
        target = SimpleNamespace(setup_scene=Mock())
        try:
            execute(method.body[start:end + 1], BASE, {'kwags': kwargs, 'self': target})
        except ValueError:
            target.setup_scene.assert_not_called()
            raise
        target.setup_scene.assert_called_once_with(render_denoiser=kwargs.get('render_denoiser', 'oidn'))
        return target.setup_scene.call_args.kwargs

    def setup_renderer(self, kwargs):
        method = node(BASE, 'setup_scene')
        calls = [n for n in method.body if isinstance(n, ast.Expr)
                 and isinstance(n.value, ast.Call)
                 and ast.unparse(n.value.func) == 'sapien.render.set_ray_tracing_denoiser']
        self.assertEqual(len(calls), 1)
        setter = Mock()
        execute(calls, BASE, {'kwargs': kwargs,
                             'sapien': SimpleNamespace(render=SimpleNamespace(set_ray_tracing_denoiser=setter))})
        setter.assert_called_once_with(kwargs.get('render_denoiser', 'oidn'))

    def test_native_default_and_explicit_forwarding(self):
        for kwargs in ({}, *({'render_denoiser': m} for m in ('oidn', 'none', 'optix'))):
            with self.subTest(kwargs=kwargs):
                self.setup_renderer(self.init_renderer(kwargs))

    def test_setup_scene_direct_default(self):
        self.setup_renderer({})

    def test_native_invalid_value_rejected_before_scene(self):
        for mode in ('auto', '', None):
            with self.subTest(mode=mode):
                # Invalid inputs must raise at validation, before setup_scene.
                with self.assertRaisesRegex(ValueError, 'unsupported explicit render_denoiser'):
                    self.init_renderer({'render_denoiser': mode})


class RunStopContractTests(unittest.TestCase):
    def setUp(self):
        self.clock = Mock(return_value=100.0)
        self.ns = execute([node(EVIDENCE, n) for n in
                           ('WallTimeout', 'EvaluationCancelled', 'RunStop')], EVIDENCE,
                          {'time': SimpleNamespace(monotonic=self.clock), 'signal': signal})
        self.stop = self.ns['RunStop'](60)

    def test_normal_call_forwards_arguments_and_result(self):
        result = object()
        fn = Mock(return_value=result)
        self.assertIs(self.stop.call(fn, 3, marker='ok'), result)
        fn.assert_called_once_with(3, marker='ok')

    def test_swallowed_cancellation_blocks_current_and_next_call(self):
        for signum in (signal.SIGTERM, signal.SIGINT):
            with self.subTest(signum=signum):
                stop = self.ns['RunStop'](60)
                swallowed = []

                def native_bare_except():
                    try:
                        stop.terminate(signum, None)  # no OS signal or handler installed
                    except BaseException as exc:
                        swallowed.append(exc)
                    return 'must not escape call boundary'

                with self.assertRaises(self.ns['EvaluationCancelled']) as raised:
                    stop.call(native_bare_except)
                self.assertEqual(len(swallowed), 1)
                self.assertIs(raised.exception, swallowed[0])
                self.assertIs(stop.reason, swallowed[0])
                for _ in range(2):
                    next_call = Mock()
                    with self.assertRaises(self.ns['EvaluationCancelled']) as again:
                        stop.call(next_call)
                    self.assertIs(again.exception, swallowed[0])
                    next_call.assert_not_called()

    def test_swallowed_timeout_is_also_sticky(self):
        def native():
            try:
                self.stop.expire('test deadline')
            except BaseException:
                pass
        with self.assertRaises(self.ns['WallTimeout']):
            self.stop.call(native)
        next_call = Mock()
        with self.assertRaises(self.ns['WallTimeout']):
            self.stop.call(next_call)
        next_call.assert_not_called()

    def test_absolute_deadline_blocks_call_without_sleep(self):
        self.clock.return_value = 160.0
        fn = Mock()
        with self.assertRaises(self.ns['WallTimeout']):
            self.stop.call(fn)
        fn.assert_not_called()


if __name__ == '__main__':
    unittest.main(verbosity=2)
