#!/usr/bin/env python3
"""CPU-only runtime provenance tests; AST-isolated, no models, GPU or subprocesses."""
import ast
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest


class RuntimeProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        source = Path(__file__).with_name('behavior_closed_loop.py')
        tree = ast.parse(source.read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == 'runtime_artifact_paths')
        namespace = {'ROOT': self.root, 'CONFIG': {'source_dir': 'isaac', 'task': {'id': 0}},
                     'Path': Path, 'json': json,
                     'task_profile': SimpleNamespace(TASK_ID=0, VTT_DIR=self.root / 'vtt')}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
        self.collect = namespace['runtime_artifact_paths']
        (self.root / 'stats.json').write_text(json.dumps({'provenance': {'split_file': 'split.json'}}))
        self.cfg = {'dataset': {'stats_path': 'stats.json'},
                    'model': {'vision_encoder': {'checkpoint_path': 'dino'}}}
        self.args = SimpleNamespace(checkpoint=str(self.root / 'main.pt'), orchestrate=None,
                                    program=None, memory_vtt=None, env_wrapper=None)
        self.argv = ['--robot-config', str(self.root / 'robot.yaml')]
        self.v3_source = (self.root / 'harness/wam_v3.py').resolve()

    def v3(self, stats='v3_stats.json'):
        cfg = copy.deepcopy(self.cfg)
        cfg['model']['v3'] = {'stats_path': stats}
        return cfg

    def test_v2_does_not_require_v3_files(self):
        paths = self.collect(self.args, self.argv, self.cfg)
        self.assertNotIn(self.v3_source, paths)
        self.assertIn((self.root / 'stats.json').resolve(), paths)

    def test_main_v3_source_and_stats_are_fingerprinted(self):
        paths = self.collect(self.args, self.argv, self.v3())
        self.assertIn(self.v3_source, paths)
        self.assertIn((self.root / 'v3_stats.json').resolve(), paths)

    def test_routed_manip_v3_is_fingerprinted(self):
        self.args.orchestrate = json.dumps({'manip_checkpoint': str(self.root / 'manip.pt')})
        paths = self.collect(self.args, self.argv, self.cfg, self.v3('manip_history.json'))
        self.assertIn(self.v3_source, paths)
        self.assertIn((self.root / 'manip_history.json').resolve(), paths)

    def test_both_v3_models_keep_distinct_stats_and_one_source(self):
        self.args.orchestrate = json.dumps({'manip_checkpoint': str(self.root / 'manip.pt')})
        paths = self.collect(self.args, self.argv, self.v3('main_history.json'),
                             self.v3('manip_history.json'))
        self.assertEqual(paths.count(self.v3_source), 1)
        for name in ('main_history.json', 'manip_history.json'):
            self.assertIn((self.root / name).resolve(), paths)

    def test_invalid_v3_without_stats_fails_closed(self):
        cfg = self.v3()
        cfg['model']['v3'] = {}
        with self.assertRaises(KeyError):
            self.collect(self.args, self.argv, cfg)

    def test_absolute_stats_path_is_preserved(self):
        stats = (self.root / 'external/history.json').resolve()
        paths = self.collect(self.args, self.argv, self.v3(str(stats)))
        self.assertIn(stats, paths)

    def test_existing_v2_history_stats_are_preserved(self):
        self.cfg['model']['proprio_history_stats'] = 'legacy_history.json'
        paths = self.collect(self.args, self.argv, self.cfg)
        self.assertIn((self.root / 'legacy_history.json').resolve(), paths)


if __name__ == '__main__':
    unittest.main()
