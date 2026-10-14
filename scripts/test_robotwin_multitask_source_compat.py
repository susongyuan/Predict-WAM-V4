#!/usr/bin/env python3
"""CPU regression on actual five-task cache/SOURCE/seed metadata, plus tampering.

No GPU, checkpoint tensor loading, training interruption, or real-file mutation.
Only synthetic corruption fixtures use temporary directories.
"""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zlib

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import eval_robotwin_multitask_source_compat as compat
from harness.robotwin_policy_worker import InfrastructureError, atomic_json, identity, sha256

RUN = ROOT / 'outputs/runs/multitask5_wam_v3_vtt_20261008'
TRAIN = RUN / 'train/manifest.json'
TASKS = ('put_bottles_dustbin', 'place_cans_plasticbox', 'open_laptop', 'click_bell', 'press_stapler')


def synthetic(root, tokens=b'1\n2\n', count=2):
    task = 'click_bell'
    seed = root / f'source/{task}/aloha_agilex/seed.txt'
    seed.parent.mkdir(parents=True)
    seed.write_bytes(tokens)
    (seed.parent / 'data').mkdir()
    receipt = {'task': task, 'archive': 'demo_clean.zip', 'status': 'complete',
        'expected_sha256': 'a' * 64, 'sha256': 'a' * 64, 'expected_bytes': 100, 'received_bytes': 100,
        'zip_crc_verified': True,
        'members': [{'path': str(seed.relative_to(root)), 'bytes': len(tokens), 'zip_crc32': f'{zlib.crc32(tokens):08x}'}]
            + [{'path': f'source/{task}/aloha_agilex/data/episode_{i}.hdf5', 'bytes': 1, 'zip_crc32': '00000000'} for i in range(count)]}
    source = root / 'SOURCE.json'
    atomic_json(source, receipt)
    cache = {'source': receipt, 'source_sha256': sha256(source), 'data_root': str(seed.parent / 'data')}
    return cache, seed, source


def reseal(cache, source):
    atomic_json(source, cache['source'])
    cache['source_sha256'] = sha256(source)


class CompatTests(unittest.TestCase):
    def test_real_training_manifest_five_sources_crc_size_zero_overlap(self):
        if not TRAIN.is_file():
            self.skipTest('registered immutable training metadata unavailable')
        audit = compat.audit_training_manifest(TRAIN)
        self.assertTrue(audit['all_zero_overlap'])
        self.assertTrue(audit['all_seed_crc_size_verified'])
        self.assertEqual(set(audit['tasks']), set(TASKS))
        expected = {'put_bottles_dustbin': (143, '72cd6b42'), 'place_cans_plasticbox': (141, 'c35d80d9'),
                    'open_laptop': (141, '76c3959b'), 'click_bell': (140, '9d46d798'), 'press_stapler': (142, '02fc41a4')}
        for task, row in audit['tasks'].items():
            self.assertEqual(row['seed_count'], 50)
            self.assertEqual(row['overlap'], [])
            integrity = row['seed_integrity']
            self.assertEqual((integrity['bytes_verified'], integrity['crc32_verified']), expected[task])
        self.assertEqual(audit['tasks']['put_bottles_dustbin']['lookup'], 'unchanged_native_SOURCE.episodes')
        self.assertFalse(audit['cuda_initialized'])
        self.assertFalse(audit['weights_loaded'])

    def test_real_full_task_plan_seed_path_and_50_compat_commands(self):
        # Use immutable completed CPU-smoke metadata solely to exercise task_plan
        # (no load_contract/final-checkpoint admission, no tensor loading). No
        # mocking of cache, SOURCE, dataset_seed_inventory, assets or task_plan.
        manifest_path = RUN / 'smoke/manifest.json'
        checkpoint = RUN / 'smoke/checkpoint_step_000002.pt'
        if not manifest_path.is_file() or not checkpoint.is_file():
            self.skipTest('completed smoke metadata unavailable')
        manifest = json.loads(manifest_path.read_text())
        contract = manifest['contract']
        receipt = json.loads(checkpoint.with_name(checkpoint.name + '.sha256.json').read_text())
        caches = {}
        for item in contract['cache_files']:
            if Path(item['path']).name == 'cache_config.json':
                value = json.loads(Path(item['path']).read_text())
                caches[value['source']['task']] = {'path': item['path'], 'value': value}
        bundle = {'contract': contract, 'receipt': receipt, 'tasks': contract['tasks'], 'caches': caches,
                  'manifest': str(manifest_path), 'checkpoint': str(checkpoint)}
        args = compat.evaluation.parse_args(['--checkpoint', str(checkpoint), '--multitask-manifest', str(RUN / 'tasks.json'),
            '--out', '/tmp/not-created-by-source-compat-test', '--gpu-lock-owner', 'test-only'])
        original_seed = compat.evaluation.native.dataset_seed_inventory
        with compat.installed():
            plan = compat.evaluation.queue_plan(args, bundle)
            self.assertEqual(len(plan['queue']), 50)
            self.assertEqual(plan['source_compat']['entrypoint']['sha256'], compat.ENTRY_SHA256)
            for task, row in plan['tasks'].items():
                self.assertEqual(row['source_compat']['seed_inventory_audit']['seed_count'], 50)
                self.assertEqual(row['seed_overlap'], [])
                self.assertIn(compat.ENTRY_SHA256, [p['sha256'] for p in row['source_files']])
                self.assertEqual(row['task_config'], 'demo_clean')
                # Also exercise exactly the per-episode plan path used by child.
                child_plan = compat.evaluation.task_plan(args, bundle, task, seed=922001, assets={})
                self.assertEqual(child_plan['seeds'], [922001])
                self.assertEqual(child_plan['source_compat']['entrypoint']['sha256'], compat.ENTRY_SHA256)
            for job in plan['queue']:
                command = compat.evaluation.child_command(args, job['task'], job['seed'], Path('/tmp/not-created'))
                self.assertEqual(command[2], str(compat.ENTRY))
                self.assertEqual(command[-2:], ['--compat-sha256', compat.ENTRY_SHA256])
                original = compat._ORIGINAL_CHILD_COMMAND(args, job['task'], job['seed'], Path('/tmp/not-created'))
                command[2] = original[2]
                self.assertEqual(command[:-2], original)
        self.assertIs(compat.evaluation.native.dataset_seed_inventory, original_seed)

    def test_bound_source_must_match_frozen_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache, seed, source = synthetic(Path(tmp))
            self.assertEqual(compat.dataset_seed_inventory(cache), ({1, 2}, {seed}))
            source.write_text(source.read_text() + ' ')
            with self.assertRaisesRegex(InfrastructureError, 'SOURCE bytes'):
                compat.dataset_seed_inventory(cache)

    def test_same_size_seed_crc_tamper_and_truncation_rejected(self):
        for payload, reason in ((b'3\n4\n', 'CRC32'), (b'1\n', 'size mismatch')):
            with self.subTest(reason=reason), tempfile.TemporaryDirectory() as tmp:
                cache, seed, _ = synthetic(Path(tmp))
                seed.write_bytes(payload)
                with self.assertRaisesRegex(InfrastructureError, reason):
                    compat.dataset_seed_inventory(cache)

    def test_bad_members_or_archive_receipt_rejected(self):
        mutations = (
            lambda source: source['members'].append(copy.deepcopy(source['members'][0])),
            lambda source: source['members'][0].update(path='../seed.txt'),
            lambda source: source['members'][0].update(path='source/other/aloha_agilex/seed.txt'),
            lambda source: source['members'][0].update(bytes=0),
            lambda source: source['members'][0].update(zip_crc32='bad'),
            lambda source: source.update(zip_crc_verified=False),
            lambda source: source.update(sha256='b' * 64),
            lambda source: source.update(received_bytes=99),
        )
        for mutate in mutations:
            with self.subTest(mutation=mutate), tempfile.TemporaryDirectory() as tmp:
                cache, _, source = synthetic(Path(tmp))
                mutate(cache['source']); reseal(cache, source)
                with self.assertRaises(InfrastructureError):
                    compat.dataset_seed_inventory(cache)

    def test_complete_valid_uint32_seed_sequence_required(self):
        for payload, count in ((b'1\n1\n', 2), (b'1\n', 2), (b'-1\n2\n', 2),
                               (b'4294967296\n2\n', 2), (b'one\n2\n', 2), (b'922001\n2\n', 2)):
            with self.subTest(payload=payload), tempfile.TemporaryDirectory() as tmp:
                cache, _, _ = synthetic(Path(tmp), payload, count)
                with self.assertRaises(InfrastructureError):
                    compat.dataset_seed_inventory(cache)

    def test_symlink_escape_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            local = root / 'dataset'; local.mkdir()
            cache, seed, _ = synthetic(local)
            external = root / 'outside.txt'; external.write_bytes(seed.read_bytes())
            seed.unlink(); seed.symlink_to(external)
            with self.assertRaisesRegex(InfrastructureError, 'escapes'):
                compat.dataset_seed_inventory(cache)

    def test_wrapper_restores_globals_on_failure_and_main_delegates(self):
        before = [compat.evaluation.native.dataset_seed_inventory, compat.evaluation.task_plan,
                  compat.evaluation.queue_plan, compat.evaluation.child_command]
        def fake_main(argv):
            self.assertEqual(argv, ['--checkpoint', 'unused'])
            self.assertIs(compat.evaluation.native.dataset_seed_inventory, compat.dataset_seed_inventory)
            raise ValueError('synthetic admission failure')
        with patch.object(compat.evaluation, 'main', side_effect=fake_main), self.assertRaises(ValueError):
            compat.main(['--checkpoint', 'unused'])
        after = [compat.evaluation.native.dataset_seed_inventory, compat.evaluation.task_plan,
                 compat.evaluation.queue_plan, compat.evaluation.child_command]
        self.assertEqual(before, after)
        with self.assertRaisesRegex(InfrastructureError, 'source SHA'):
            compat.main(['--compat-sha256', '0' * 64])

    def test_cpu_import_and_real_audit_subprocess_without_torch(self):
        if not TRAIN.is_file():
            self.skipTest('registered immutable training metadata unavailable')
        code = ('import sys; from scripts import eval_robotwin_multitask_source_compat as c; '
                f'assert c.main(["--audit-training-manifest", {str(TRAIN)!r}]) == 0; '
                'assert "torch" not in sys.modules; assert "sapien" not in sys.modules')
        result = subprocess.run([sys.executable, '-B', '-c', code], cwd=ROOT, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        value = json.loads(result.stdout)
        self.assertEqual(value['status'], 'cpu_seed_inventory_verified')
        self.assertTrue(value['all_zero_overlap'])


if __name__ == '__main__':
    unittest.main()
