#!/usr/bin/env python3
"""CPU software tests, never physical success evidence."""
from io import BytesIO
from pathlib import Path
import sys
import tempfile
import unittest
import json
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch

from harness.zhuhai_dataset import FrameFeatureStore, HybridTaskDataset, SourceEpisode, grouped_split
from harness.zhuhai_demo_recording import StreamDemoRecorder, data_contract, native_observation
from harness.zhuhai_hybrid_cache import HybridFeatureCache
from scripts import collect_isaac_zhuhai as collector
from scripts.test_zhuhai_pipeline import fixture_contract, fixture_row, rule_state


class DatasetTest(unittest.TestCase):
    def test_episode_local_alignment_and_terminal(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "episode_0000000.hdf5"
            recorder = StreamDemoRecorder(path, metadata={"task": "cpu_test_only"}, frequency=10)
            for frame in range(5):
                rgb = np.full((240, 320, 3), frame * 20, np.uint8)
                commanded = np.asarray(collector.robot_spec.HOME_ACTION, np.float32)
                commanded[0] += .1 + frame*.01
                measured = np.asarray(collector.robot_spec.HOME_ACTION, np.float32)
                measured[0] += frame*.001
                measured[7] = .8
                observation = {"measured_joint_gripper_state": measured, "joint_action": {
                    "left_arm": commanded[:7], "right_arm": commanded[8:15],
                    "left_gripper": commanded[7], "right_gripper": commanded[15]},
                    "observation": {name: {"rgb": rgb} for name in
                                    ("head_camera", "left_camera", "right_camera")}}
                recorder.capture(observation, frame * .1, rule_state(False, frame*.1))
            recorder.finish(final_evidence=rule_state(True, .4), successful=True)
            def producer(episode, frame):
                assert len(episode.rgb(frame)) == 3
                value = torch.full((3, 3, 201, 1024), frame * .1, dtype=torch.bfloat16)
                feat = value.view(torch.int16).numpy().view(np.uint16)
                stream = BytesIO()
                np.savez(stream, feat=feat, pool=np.full((3, 2048), frame, np.float32))
                return stream.getvalue()
            cache = HybridFeatureCache(Path(tmp) / "cache", budget_bytes=0, min_free_bytes=0)
            store = FrameFeatureStore(cache, producer, encoder_sha256="a" * 64)
            dataset = HybridTaskDataset("cpu_test_only", [{"path": str(path), "split": "train"}],
                                        store, future_offset=2, condition=np.ones(1024))
            sample = dataset[2]
            self.assertEqual(sample["state"].shape, (7, 16))
            self.assertEqual(sample["action_sequence"].shape, (32, 16))
            self.assertEqual(sample["pool"].shape, (3, 4, 2048))
            self.assertEqual(int(sample["action_valid_mask"].sum()), 2)
            self.assertTrue(sample["future_valid_mask"])
            self.assertAlmostEqual(float(sample["state"][0, 0]), collector.robot_spec.HOME_ACTION[0]+.002, places=6)
            self.assertAlmostEqual(float(sample["state"][1, 0]), collector.robot_spec.HOME_ACTION[0], places=6)
            self.assertAlmostEqual(float(sample["action_sequence"][0, 0]), collector.robot_spec.HOME_ACTION[0]+.13, places=6)
            self.assertAlmostEqual(float(sample["state"][0, 7]), .8, places=6)
            self.assertEqual(float(sample["action_sequence"][0, 7]), 1.)
            self.assertTrue(np.all(sample["pool"][:, 1:] == 0.))
            self.assertFalse(dataset[3]["future_valid_mask"])
            prototype = dataset.train_prototype()
            self.assertEqual(prototype.shape, (1024,))
            self.assertGreater(float(np.linalg.norm(prototype)), 0.)
            for split in ("val", "heldout"):
                dataset.episodes[0].row["split"] = split
                with self.assertRaises(ValueError):
                    dataset.train_prototype()

    def test_relatives_stay_in_same_split(self):
        rows = [{"family_id": str(family), "path": f"{family}_{variant}"}
                for family in range(4) for variant in range(3)]
        splits = grouped_split(rows, counts=(2, 1, 1))
        memberships = {}
        for split, entries in splits.items():
            for row in entries:
                memberships.setdefault(row["family_id"], set()).add(split)
        self.assertTrue(all(len(value) == 1 for value in memberships.values()))

    def test_full_source_family_split_600_per_task_1200_total(self):
        total = 0
        for task in ('zhuhai_task1', 'zhuhai_task2'):
            rows = [dict(family_id=f'life:{task}:source:{family}',
                         path=f'{task}_{family}_{variant}', domain='source' if variant == 0 else 'randomized')
                    for family in range(100) for variant in range(6)]
            splits = grouped_split(rows)
            self.assertEqual({split: len(items) for split, items in splits.items()},
                             dict(train=480, val=60, heldout=60))
            groups = {split: {r['family_id'] for r in items} for split, items in splits.items()}
            self.assertEqual({split: len(ids) for split, ids in groups.items()}, dict(train=80, val=10, heldout=10))
            self.assertFalse(groups['train'] & groups['val'] or groups['train'] & groups['heldout'] or groups['val'] & groups['heldout'])
            for split, items in splits.items():
                for family in groups[split]:
                    relatives = [r for r in items if r['family_id'] == family]
                    self.assertEqual(len(relatives), 6)
                    self.assertEqual(sum(r['domain'] == 'source' for r in relatives), 1)
            total += sum(len(items) for items in splits.values())
        self.assertEqual(total, 1200)

    def test_family_split_is_deterministic_under_row_reordering(self):
        rows = [dict(family_id=f'family:{i}', path=f'{i}_{j}') for i in range(100) for j in range(6)]
        mapping = lambda s: {row['path']: split for split, items in s.items() for row in items}
        self.assertEqual(mapping(grouped_split(rows)), mapping(grouped_split(list(reversed(rows)))))
        with self.assertRaises(ValueError):grouped_split([r for r in rows if r['family_id'] != 'family:99'])

    def test_existing_training_admission_accepts_explicit_500_override(self):
        # Exercise only the CPU manifest/split function, not checkpoint/model IO.
        from harness.zhuhai_training import task_sources
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            refs = {}
            active_robot_label = collector.robot_binding()['robot']
            for mode, count in collector.PRODUCTION_QUOTAS.items():
                directory = root/mode; directory.mkdir()
                contract = fixture_contract(mode=mode)
                contract['robot'] = active_robot_label
                contract['source_scene'] = dict(path=str(Path(__file__).resolve()), sha256=collector.sha256(__file__))
                if mode == 'randomized':
                    contract['source_manifest'] = dict(path=str(refs['source']), sha256=collector.sha256(refs['source']))
                rows = [fixture_row(directory, contract,
                    (1000 if mode == 'source' else 10000)+i,
                    parent=None if mode == 'source' else 1000+i%100) for i in range(count)]
                refs[mode] = directory/'manifest.json'
                collector.write_once(refs[mode], dict(contract=contract, status='complete',
                    requested_success_count=count, success_count=count, attempts=rows))
            tasks = root/'tasks.json'
            collector.write_once(tasks, {'tasks': [dict(task='zhuhai_task1',
                source_manifest=str(refs['source']), randomized_manifest=str(refs['randomized']))]})
            splits, inventory = task_sources(tasks, source_count=100, randomized_count=500)
            self.assertEqual(inventory['zhuhai_task1']['quotas'], dict(source=100, randomized=500))
            self.assertEqual({k: len(v) for k, v in splits['zhuhai_task1'].items()}, dict(train=480, val=60, heldout=60))
            self.assertFalse(torch.cuda.is_initialized())


class CollectorHdfTests(unittest.TestCase):
    """Fake robot + real recorder: CPU evidence integrity, NEVER expert success."""
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        self.args = SimpleNamespace(out=self.root, task='zhuhai_task1', track='life', mode='source',
            count=1, seed_start=10, max_attempts=3, max_consecutive_failures=3,
            source_manifest=None, gpu_lock_owner='CPU-MOCK-ONLY', life_mapping='body_text')
        self.contract = fixture_contract(count=1)
        self.plan_success = True
        self.raise_expert = False
        self.extra_camera = False
        self.empty_camera = False
        self.unsafe_initial = False
        self.unsafe_after_step = False
        self.missing_safety = False
        self.bad_trace = False
        self.measured_gripper = .8
        test = self
        class FakeEnv:
            action_dim = 16
            arm_dof = 7
            def __init__(self, owner, output_dir):
                self.output_dir = output_dir
                self.joint_limits = np.asarray(collector.robot_spec.DUAL_JOINT_LIMITS)
                self.robot_contract = collector.robot_spec.robot_contract()
                self.on_control_step = None
            def setup(self, *args, **kwargs):
                self.physics_time = 0.
                self.full_score = False
                self.command = np.asarray(collector.robot_spec.HOME_ACTION, np.float64)
                self.measured = self.command.copy(); self.measured[0] -= .025; self.measured[7] = test.measured_gripper
                self.camera_evidence = dict(all_geometry_in_frame=True, all_named_entities_visible=True)
            def rule_state(self):
                value = rule_state(self.full_score, self.physics_time)
                if test.unsafe_initial or (test.unsafe_after_step and self.physics_time > 0):
                    value['physics_safety'].update(passed=False, events=[{'CPU_fixture_unsafe': True}])
                if test.missing_safety:value.pop('physics_safety')
                return value
            def get_action(self):return self.command.copy()
            def actual_state(self):return self.measured.copy()
            def step_action(self, action, interval_seconds=.1):
                self.command = collector.robot_spec.validate_action(action)
                self.physics_time += interval_seconds
                self.measured = self.command.copy(); self.measured[0] -= .025; self.measured[7] = test.measured_gripper
                if self.on_control_step:self.on_control_step(self)
            def get_obs(self):
                rgb = np.arange(32*32*3, dtype=np.uint8).reshape(32, 32, 3)
                cameras = {name: {'rgb': rgb} for name in collector.ONBOARD_CAMERAS}
                if test.extra_camera:cameras['evaluation_third_view'] = {'rgb': rgb}
                if test.empty_camera:cameras['right_camera'] = {'rgb': np.zeros_like(rgb)}
                return dict(observation=cameras, measured_joint_gripper_state=self.actual_state(),
                    joint_action=dict(left_arm=self.command[:7], right_arm=self.command[8:15],
                        left_gripper=self.command[7], right_gripper=self.command[15]))
            def close(self):pass
        class FakeExpert:
            def __init__(self, env):self.env = env; self.trace = []; self.motion_commands = []; self.plan_success = test.plan_success
            def play_once(self):
                self.env.full_score = True
                command = self.env.get_action(); command[0] += .05
                self.trace.append(dict(test_only=True, target_action=command.tolist()))
                if test.bad_trace:self.trace.append({'nonfinite_fixture': float('nan')})
                self.motion_commands.append(command.tolist())
                self.env.step_action(command, .1)
                if test.raise_expert:raise RuntimeError('synthetic interruption, no physical result')
                return dict(plan_success=test.plan_success, test_only=True)
        runtime = ModuleType('harness.isaac_zhuhai_runtime'); runtime.IsaacZhuhaiEnv = FakeEnv
        expert = ModuleType('harness.isaac_zhuhai_expert'); expert.SmoothExpert = FakeExpert
        watcher = ModuleType('scripts.zhuhai_resource_watch')
        watcher.snapshot = lambda: dict(advice='resources_within_declared_reserves', CPU_fixture_only=True)
        self.watcher = watcher
        for context in (patch.dict(sys.modules, {'harness.isaac_zhuhai_runtime': runtime,
                'harness.isaac_zhuhai_expert': expert, 'scripts.zhuhai_resource_watch': watcher}),
                patch.object(collector, 'verify_owner'), patch.object(collector, 'check_disk'),
                patch.object(collector, 'log'), patch('builtins.print')):
            context.start(); self.addCleanup(context.stop)

    def manifest(self):return json.loads((self.root/'manifest.json').read_text())

    def test_real_hdf_is_three_onboard_archives_not_video(self):
        self.assertEqual(collector.collect(self.args, self.contract, []), 0)
        row = self.manifest()['attempts'][0]
        episode = SourceEpisode({'path': row['data_path'], 'data_sha256': row['data_sha256']})
        self.assertEqual(len(episode.rgb(0)), 3)
        import h5py
        with h5py.File(row['data_path'], 'r') as handle:
            self.assertEqual(set(handle['vision']), {'cam_head', 'cam_left_wrist', 'cam_right_wrist'})
            self.assertFalse(handle.attrs['record_video'])
        self.assertFalse(list(self.root.rglob('*.mp4')) or list(self.root.rglob('*.avi')))
        self.assertFalse(torch.cuda.is_initialized())
        collect_rows = collector.validate_manifest(self.root/'manifest.json', task='zhuhai_task1', mode='source', count=1)
        self.assertEqual(len(collect_rows), 1)

    def test_failed_plan_with_full_score_never_counts_as_success(self):
        self.plan_success = False
        self.assertEqual(collector.collect(self.args, self.contract, []), 2)
        manifest = self.manifest()
        self.assertEqual(manifest['success_count'], 0)
        self.assertEqual(len(manifest['attempts']), 3)
        paths = []
        for row in manifest['attempts']:
            self.assertEqual(row['status'], 'failed')
            self.assertNotIn('data_path', row)
            path = Path(row['evidence_path']); self.assertTrue(path.is_file()); paths.append(path)
            with self.assertRaisesRegex(ValueError, 'Failed/partial'):SourceEpisode({'path': str(path)})
        self.assertEqual(len(set(paths)), 3)
        self.assertEqual(len(list((self.root/'attempt_evidence').glob('*/result.json'))), 3)
        self.assertEqual(len(list((self.root/'resource_checks').glob('*.json'))), 3)
        before = {p: p.read_bytes() for p in self.root.rglob('*') if p.is_file() and p.name != 'manifest.json'}
        self.assertEqual(collector.collect(self.args, self.contract, []), 2)
        self.assertEqual(before, {p: p.read_bytes() for p in before})

    def test_partial_expert_exceptions_preserve_hdf_but_are_not_training_data(self):
        self.raise_expert = True
        self.assertEqual(collector.collect(self.args, self.contract, []), 2)
        self.assertEqual(self.manifest()['success_count'], 0)
        for row in self.manifest()['attempts']:
            path = Path(row['partial_evidence_path'])
            self.assertTrue(path.exists())
            self.assertTrue(path.name.endswith('.hdf5.partial'))
            with self.assertRaisesRegex(ValueError, 'Failed/partial'):SourceEpisode({'path': str(path)})

    def test_fourth_camera_is_rejected_before_recording_success(self):
        self.extra_camera = True
        self.assertEqual(collector.collect(self.args, self.contract, []), 2)
        self.assertEqual(self.manifest()['success_count'], 0)
        self.assertTrue(all('Exactly head' in row['failure_reason'] for row in self.manifest()['attempts']))

    def test_resource_pause_does_not_consume_seed_or_overwrite_snapshot(self):
        self.watcher.snapshot = lambda: {'advice': 'pause_CPU_fixture_only'}
        self.assertEqual(collector.collect(self.args, self.contract, []), 3)
        self.assertEqual(self.manifest()['attempts'], [])
        first = next((self.root/'resource_checks').glob('*.json')); before = first.read_bytes()
        self.watcher.snapshot = lambda: {'advice': 'resources_within_declared_reserves'}
        self.assertEqual(collector.collect(self.args, self.contract, []), 0)
        self.assertEqual(self.manifest()['attempts'][0]['seed'], self.args.seed_start)
        self.assertEqual(first.read_bytes(), before)
        self.assertEqual(len(list((self.root/'resource_checks').glob('*.json'))), 2)

    def test_changed_hdf_hash_is_rejected(self):
        self.assertEqual(collector.collect(self.args, self.contract, []), 0)
        row = self.manifest()['attempts'][0]
        with self.assertRaisesRegex(ValueError, 'admitted demonstration hash'):
            SourceEpisode({'path': row['data_path'], 'data_sha256': 'a'*64})

    def test_unsafe_initial_or_missing_gate_is_never_success(self):
        self.unsafe_initial = True
        self.assertEqual(collector.collect(self.args, self.contract, []), 2)
        self.assertEqual(self.manifest()['success_count'], 0)
        self.assertTrue(all('physics_safety' in r['failure_reason'] for r in self.manifest()['attempts']))

    def test_missing_physics_safety_fails_closed(self):
        self.missing_safety = True
        self.assertEqual(collector.collect(self.args, self.contract, []), 2)
        self.assertEqual(self.manifest()['success_count'], 0)

    def test_unsafe_control_boundary_is_preserved_in_partial_evidence(self):
        self.unsafe_after_step = True
        self.assertEqual(collector.collect(self.args, self.contract, []), 2)
        self.assertEqual(self.manifest()['success_count'], 0)
        import h5py
        for row in self.manifest()['attempts']:
            self.assertFalse(row['physics_safety_failure_evidence']['physics_safety']['passed'])
            with h5py.File(row['partial_evidence_path'], 'r') as handle:
                self.assertEqual(handle['measured_state'].shape, (2, 16))
                self.assertEqual(handle['target_command'].shape, (2, 16))
                last = json.loads(handle['rule_evidence'].asstr()[-1])
                self.assertFalse(last['physics_safety']['passed'])

    def test_scene_visibility_flag_cannot_replace_actual_three_rgb(self):
        self.empty_camera = True
        self.assertEqual(collector.collect(self.args, self.contract, []), 2)
        self.assertEqual(self.manifest()['success_count'], 0)
        for row in self.manifest()['attempts']:
            self.assertTrue(row['initial_head_camera_evidence']['all_named_entities_visible'])
            self.assertFalse(row['initial_camera_observations']['right_camera']['nonempty_rgb'])
            self.assertTrue(Path(row['partial_evidence_path']).exists())

    def test_success_manifest_records_actual16_statistics_and_trace(self):
        self.assertEqual(collector.collect(self.args, self.contract, []), 0)
        row = self.manifest()['attempts'][0]
        self.assertEqual((row['state_dim'], row['action_dim']), (16, 16))
        self.assertEqual(row['state_semantics'], 'measured_joint_gripper')
        self.assertTrue(row['physics_safety_all_observations_passed'])
        stats = row['episode_statistics']
        self.assertEqual(len(stats['state']['mean']), 16)
        self.assertAlmostEqual(stats['state']['mean'][0], collector.robot_spec.HOME_ACTION[0]-.025, places=6)
        self.assertAlmostEqual(stats['action']['mean'][0], collector.robot_spec.HOME_ACTION[0]+.05, places=6)
        receipt = json.loads(Path(row['expert_receipt_path']).read_text())
        self.assertEqual(len(receipt['motion_commands'][0]), 16)
        self.assertEqual(receipt['state_semantics'], 'measured_joint_gripper')

    def test_reader_rejects_old14_header_and_field_width(self):
        self.assertEqual(collector.collect(self.args, self.contract, []), 0)
        row = self.manifest()['attempts'][0]
        import h5py
        from harness.zhuhai_demo_recording import SOURCE_FORMAT
        with h5py.File(row['data_path'], 'r+') as handle:
            handle.attrs['source_format'] = 'Zhuhai2026_native14_physical_expert'
        with self.assertRaisesRegex(ValueError, 'native14'):
            SourceEpisode({'path': row['data_path']})
        with h5py.File(row['data_path'], 'r+') as handle:
            handle.attrs['source_format'] = SOURCE_FORMAT
            values = handle['state/left_arm_joint_states'][:, :6]
            del handle['state/left_arm_joint_states']
            handle.create_dataset('state/left_arm_joint_states', data=values)
        with self.assertRaisesRegex(ValueError, 'width 7'):
            SourceEpisode({'path': row['data_path']})

    def test_reader_detects_commanded_state_masquerading_as_measurement(self):
        self.assertEqual(collector.collect(self.args, self.contract, []), 0)
        row = self.manifest()['attempts'][0]
        import h5py
        with h5py.File(row['data_path'], 'r+') as handle:
            handle['state/left_arm_joint_states'][:] = handle['target_command'][:-1, :7]
        with self.assertRaisesRegex(ValueError, 'State must be measured'):
            SourceEpisode({'path': row['data_path']})

    def test_measured_gripper_overshoot_is_not_clipped_or_replaced(self):
        self.measured_gripper = 1.025
        self.assertEqual(collector.collect(self.args, self.contract, []), 0)
        row = self.manifest()['attempts'][0]
        episode = SourceEpisode({'path': row['data_path']})
        self.assertAlmostEqual(float(episode.state[0, 7]), 1.025, places=6)
        self.assertEqual(float(episode.action[0, 7]), 1.)

    def test_invalid_expert_trace_cannot_silently_be_successful(self):
        self.bad_trace = True
        self.assertEqual(collector.collect(self.args, self.contract, []), 2)
        self.assertEqual(self.manifest()['success_count'], 0)
        for row in self.manifest()['attempts']:
            self.assertFalse(row['expert_receipt_valid'])
            receipt = json.loads(Path(row['expert_receipt_path']).read_text())
            self.assertTrue(receipt['invalid_expert_receipt'])
            self.assertIn('nan', receipt['raw_receipt_repr'])


class ActualExpertInterfaceTest(unittest.TestCase):
    def test_actual_smooth_expert_uses_native16_public_driver_without_assets_or_isaac(self):
        # Only a short numerical move through the real expert class. No scene,
        # rule simulation, actor state, model, or physical success assertion.
        from harness.isaac_zhuhai_expert import SmoothExpert
        spec = collector.robot_spec
        class Env:
            action_dim = 16
            arm_dof = 7
            manifest = {'CPU_fixture_only': True, 'objects': {}}
            gripper_pose_is_tool = True
            kinematics = SimpleNamespace(limits={arm: np.asarray(spec.JOINT_LIMITS) for arm in ('left', 'right')})
            def __init__(self):self.command=np.asarray(spec.HOME_ACTION, np.float64); self.calls=[]
            def get_action(self):return self.command.copy()
            def actual_state(self):return self.command.copy()
            def gripper_pose(self, arm):return np.eye(4)
            def step_action(self, action, interval_seconds=.1):
                self.command=spec.validate_action(action)
                self.calls.append((self.command.copy(), interval_seconds))
            @property
            def step_native14(self):raise AssertionError('legacy14 driver must not be accessed')
            @property
            def get_native14(self):raise AssertionError('legacy14 state must not be accessed')
            @property
            def actual_native14(self):raise AssertionError('legacy14 state must not be accessed')
        env = Env(); expert = SmoothExpert(env)
        target = env.get_action(); target[0] += .03; target[8] -= .02
        expert.move_joints(target, 'CPU_native16_driver_interface')
        self.assertEqual(expert.dof, 16)
        self.assertEqual(expert.arm_dof, 7)
        self.assertTrue(env.calls)
        self.assertTrue(all(a.shape == (16,) and dt == .1 for a, dt in env.calls))
        self.assertTrue(all(len(a) == 16 for a in expert.motion_commands))
        self.assertEqual(len(expert.trace[-1]['target_action']), 16)
        self.assertNotIn('isaacsim', sys.modules)


if __name__ == "__main__":
    unittest.main()
