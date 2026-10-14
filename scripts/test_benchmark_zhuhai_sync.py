#!/usr/bin/env python3
"""CPU CLI/HDF + no-physics test doubles for independent sync instrumentation.

No Isaac, carb, Torch, GPU, expert or real physics is started. Mock execution
artifacts exist only in temporary directories and are not benchmark evidence.
"""
from __future__ import annotations
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['PYTHONDONTWRITEBYTECODE']='1'
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import h5py
import numpy as np
from scripts import benchmark_zhuhai_sync as b
from harness import zhuhai_robot_spec as spec


class Settings:
    def __init__(self):self.values={k:True for k in b.SETTING_KEYS}
    def get(self,k):return self.values.get(k)
    def set(self,k,v):self.values[k]=v


class Fabric:
    def __init__(self):self.updates=0
    def force_update(self,*args):self.updates+=1


class World:
    def __init__(self):self._physx_fabric_interface=Fabric();self.calls=[]
    def step(self,render=True,update_fabric=False):self.calls.append((render,update_fabric))
    def render(self):
        if self._physx_fabric_interface is not None:
            self._physx_fabric_interface.force_update(.004,0.)


class Rigid:
    def __init__(self):
        self.valid=True
        self.poses=np.array([[1.,2.,3.,.1,.2,.3,.9]])
        self.velocities=np.array([[.2,.3,.4,.5,.6,.7]])
        self._physics_view=SimpleNamespace(get_transforms=lambda:self.poses,get_velocities=lambda:self.velocities)
    def is_physics_handle_valid(self):return self.valid


class Arm:
    def __init__(self,q):
        self.positions=np.asarray(q,dtype=float).reshape(1,9)
        self._articulation_view=SimpleNamespace(is_physics_handle_valid=lambda:True,
            _physics_view=SimpleNamespace(get_dof_positions=lambda:self.positions))


class FakeEnv:
    """API bookkeeping only. Updating arrays here is NOT a physics simulation."""
    episode=None
    fail=False
    def __init__(self,**kwargs):
        self.world=World();self.physics_dt=.004;self.physics_time=0.;self.on_control_step=None
        self.rigid=Rigid();self.body_names=['mock_body'];self.robot_contract=spec.robot_contract()
        self.arm_indices={a:list(range(9)) for a in ('left','right')}
        self.command=np.asarray(spec.HOME_ACTION,dtype=float)
        self.arms={a:Arm([*self.command[o:o+7],.04,.04]) for a,o in [('left',0),('right',8)]}
        self.last_images={v:np.full((8,12,3),50+i,np.uint8) for i,v in enumerate(b.VIEWS)}
        self.steps=0;self.closed=False;self.targets=[]
    def setup(self,**kwargs):self.setup_args=kwargs
    def actual_state(self):
        result=[]
        for arm in ('left','right'):
            q=self.arms[arm].positions[0];result.extend(q[:7]);result.append(q[7:].mean()/.04)
        return np.asarray(result)
    def body_states(self):
        p,v=self.rigid.poses[0],self.rigid.velocities[0]
        return {'mock_body':dict(position_world_m=p[:3].tolist(),quaternion_wxyz=p[[6,3,4,5]].tolist(),
            linear_velocity_m_s=v[:3].tolist(),angular_velocity_rad_s=v[3:].tolist())}
    def step_action(self,action,interval_seconds=.1):
        self.command=np.asarray(action).copy();self.targets.append(self.command.copy())
        for _ in range(25):self.world.step(render=False,update_fabric=True)
        self.physics_time+=interval_seconds
        if self.fail:raise RuntimeError('injected CPU test-double failure; NOT a physics result')
        for arm,offset in [('left',0),('right',8)]:
            self.arms[arm].positions[0]=[*self.command[offset:offset+7],*(self.command[offset+7]*.04,)*2]
        self.body_states();self.steps+=1
        return self.rule_state()
    def rule_state(self):
        return dict(physics_time_seconds=self.physics_time,**self.episode.truth[self.steps])
    def get_action(self):return self.command.copy()
    def capture(self):self.world.render();return self.last_images,{}
    def close(self):self.closed=True


class FastExitApp:
    """Supported close signature; deliberately kills ONLY the CPU test child."""
    config = {'fast_shutdown': True}
    def close(self, wait_for_replicator=True, skip_cleanup=False, exit_code=0):
        assert 'isaacsim' not in sys.modules and 'torch' not in sys.modules and 'carb' not in sys.modules
        print(f'CPU_FAKE_FAST_SHUTDOWN exit_code={exit_code}', flush=True)
        os._exit(exit_code)


class FastExitEnv(FakeEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.app = FastExitApp()
    def close(self):
        self.app.close()  # No caller kwargs: benchmark must forward supported exit_code.


def close_child(case, out, reference=None):
    episode = b.read_episode(b.DEFAULT_SOURCE/'manifest.json', b.DEFAULT_SOURCE/'data/episode_seed_0000921002.hdf5')
    FakeEnv.episode = episode
    env = FastExitEnv()
    mode = 'reference' if case == 'reference' else 'render_sync'
    if case == 'fabric_unavailable':
        env.world._physx_fabric_interface = None
        env.world._extension_manager = SimpleNamespace(is_extension_enabled=lambda name: False)
    if case == 'action_failure':
        env.fail = True
    args = b.parser().parse_args(['--mode',mode,'--steps','2','--out',out])
    args.gpu_lock_owner = 'CPU-test-child'; args.reference_out = reference
    snapshot = b.source_snapshot(episode)
    if reference:
        # Isolate these no-physics mocks from concurrent main-thread source edits.
        snapshot = json.loads((Path(reference)/'result.json').read_text())['current_source_snapshot']
    with mock.patch.object(b, 'source_snapshot', return_value=snapshot):
        b.execute_replay(args, episode, env_factory=lambda **kw: env,
                         settings_factory=Settings, lease_check=lambda owner: None)
    raise AssertionError('Fake SDK close must never return')


class BenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ep=b.read_episode(b.DEFAULT_SOURCE/'manifest.json',b.DEFAULT_SOURCE/'data/episode_seed_0000921002.hdf5')
        cls.snapshot=b.source_snapshot(cls.ep);FakeEnv.episode=cls.ep
    def test_cli_defaults_are_cpu_all_actions_initial_final_only(self):
        args=b.parser().parse_args([])
        self.assertFalse(args.execute);self.assertIsNone(args.steps);self.assertEqual(args.capture_stride,0)
        self.assertEqual(args.mode,'reference')
    def test_real_hdf_after_action_alignment_and_source_hash(self):
        self.assertEqual(len(self.ep.actions),982);self.assertEqual(len(self.ep.times),983)
        self.assertEqual(self.ep.metadata['seed'],921002)
        self.assertEqual(self.ep.truth[-1]['score']['base_score'],30)
        with h5py.File(self.ep.hdf_path,'r') as f:
            np.testing.assert_array_equal(self.ep.next_state,f['measured_state'][1:])
            np.testing.assert_array_equal(self.ep.actions,f['target_command'][1:])
            np.testing.assert_array_equal(self.ep.next_state[-1],f['terminal_observation/measured_state'][:])
        self.assertFalse(self.ep.actions.flags.writeable)
        self.assertEqual(b.sha256(self.ep.hdf_path),'b4c2e4ee0da7b47b37083c1dc06d2bb87510088233511a777fab00d4b87800bd')
    def test_cpu_check_never_imports_isaac_or_calls_gpu_lock(self):
        with contextlib.redirect_stdout(io.StringIO()) as out, mock.patch.object(b,'log'), \
             mock.patch.object(b,'verify_owner',side_effect=AssertionError('no GPU in CPU preflight')):
            self.assertEqual(b.main(['--check-only','--mode','render_sync','--steps','2']),0)
        self.assertEqual(json.loads(out.getvalue())['planned_steps'],2)
        self.assertNotIn('isaacsim',sys.modules);self.assertNotIn('carb',sys.modules);self.assertNotIn('torch',sys.modules)
    def test_invalid_limits_and_execute_without_lock_fail_before_init(self):
        for args in (['--steps','0'],['--capture-stride','-1'],['--execute']):
            with self.subTest(args=args),contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):b.main(args)
    def test_sync_flag_override_and_restoration_both_modes(self):
        for mode in ('reference','render_sync'):
            env=FakeEnv();settings=Settings();patch=b.SyncPatch(env,settings,mode)
            original=env.actual_state().copy();patch.install()
            env.world.step(render=False,update_fabric=True)
            env.world.step(False,True)
            env.world.step(render=True,update_fabric=False)
            self.assertEqual(env.world.calls,[(False,mode=='reference'),(False,mode=='reference'),(True,False)])
            np.testing.assert_array_equal(env.actual_state(),original)
            self.assertEqual(settings.get(b.SETTING_KEYS[0]),mode=='reference')
            patch.restore();self.assertTrue(settings.get(b.SETTING_KEYS[0]));self.assertNotIn('step',env.world.__dict__)
            self.assertNotIn('actual_state',env.__dict__)
    def test_tensor_quaternion_order_and_no_tensor_mutation(self):
        env=FakeEnv();readers=b.TensorReaders(env);before=env.rigid.poses.copy()
        self.assertEqual(readers.body_states()['mock_body']['quaternion_wxyz'],[.9,.1,.2,.3])
        np.testing.assert_array_equal(readers.actual_state(),env.actual_state())
        np.testing.assert_array_equal(before,env.rigid.poses)
        env.rigid.valid=False
        with self.assertRaises(RuntimeError):readers.body_states()
    def test_missing_fabric_refuses_optimized_mode(self):
        env=FakeEnv();env.world._physx_fabric_interface=None
        with self.assertRaises(RuntimeError):b.SyncPatch(env,Settings(),'render_sync').install()
    def test_joint_and_gripper_errors_have_distinct_units(self):
        a=np.zeros(16);a[0]=.02;a[7]=.4
        metrics=b.state_error(a,np.zeros(16))
        self.assertEqual(metrics['max_joint_abs_rad'],.02);self.assertEqual(metrics['max_gripper_abs_fraction'],.4)
    def run_mock(self,out,mode,*,reference=None,fail=False,stride=0):
        env=FakeEnv();env.fail=fail
        args=b.parser().parse_args(['--mode',mode,'--steps','2','--out',str(out),'--capture-stride',str(stride)])
        args.reference_out=reference;args.gpu_lock_owner='CPU-test-double'
        with mock.patch.object(b,'source_snapshot',return_value=self.snapshot), \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            code,result=b.execute_replay(args,self.ep,env_factory=lambda **kw:env,settings_factory=Settings,lease_check=lambda owner:None)
        self.assertTrue(env.closed)
        return code,result,env
    def test_mock_replay_unique_output_raw_targets_and_png_cadence(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'reference';code,result,env=self.run_mock(out,'reference')
            self.assertEqual(code,0);self.assertEqual(result['completed_steps'],2)
            self.assertFalse(result['full_task_replayed'])
            np.testing.assert_array_equal(np.stack(env.targets),self.ep.actions[:2])
            self.assertEqual(len(list((out/'onboard').glob('*.png'))),6)
            self.assertEqual(result['sync_protocol']['calls']['fabric_true_physics_steps'],50)
            self.assertAlmostEqual(result['physics_elapsed_seconds'],.2)
            saved=(out/'result.json').read_bytes()
            with self.assertRaises(FileExistsError):self.run_mock(out,'reference')
            self.assertEqual((out/'result.json').read_bytes(),saved)
    def test_mock_optimized_failure_preserved_no_reference_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'failed';code,result,env=self.run_mock(out,'render_sync',fail=True)
            self.assertEqual(code,2);self.assertEqual(result['status'],'failed');self.assertEqual(result['mode'],'render_sync')
            self.assertIn('failure_rule_evidence',result);self.assertTrue(result['source_hashes_unchanged'])
            self.assertTrue((out/'onboard/final_head_camera.png').is_file())
            self.assertEqual(result['sync_protocol']['calls']['fabric_false_physics_steps'],25)
    def test_mock_pair_comparison_and_sparse_capture_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            ref=Path(tmp)/'ref';self.run_mock(ref,'reference')
            code,result,_=self.run_mock(Path(tmp)/'optimized','render_sync',reference=ref)
            self.assertEqual(code,0);self.assertEqual(result['reference_comparison']['max_joint_abs_rad'],0.)
            self.assertTrue(result['reference_comparison']['final_score_equal'])
            code,result,_=self.run_mock(Path(tmp)/'wrong_cadence','render_sync',reference=ref,stride=1)
            self.assertEqual(code,2);self.assertIn('reference_comparison_error',result)
            self.assertEqual(len(list((Path(tmp)/'wrong_cadence/onboard').glob('*.png'))),9)

    def launch_close_child(self, case, out, reference=None):
        command = [sys.executable, '-B', str(Path(__file__).resolve()), '--close-child', case, str(out)]
        if reference is not None:
            command.append(str(reference))
        completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=20,
                                   env=dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1'))
        self.assertIn('benchmark_result_committed_before_close', completed.stdout,
                      completed.stdout + '\n' + completed.stderr)
        self.assertIn('CPU_FAKE_FAST_SHUTDOWN', completed.stdout)
        self.assertLess(completed.stdout.index('benchmark_result_committed_before_close'),
                        completed.stdout.index('CPU_FAKE_FAST_SHUTDOWN'))
        result = json.loads((out/'result.json').read_text())
        protocol = json.loads((out/'protocol.json').read_text())
        self.assertTrue(result['finalization']['authoritative_replay_result'])
        self.assertTrue(result['finalization']['all_required_fields_committed_before_close'])
        self.assertEqual(result['close']['state'], 'pending_may_not_return')
        self.assertTrue(result['close']['exit_code_forwarded'])
        self.assertFalse(result['close']['return_guaranteed'])
        self.assertNotIn('post_close_observed', result)
        self.assertIsNone(result['timing']['close_wall_seconds'])
        self.assertIn('excludes', result['timing']['total_wall_seconds_scope'])
        self.assertTrue(result['source_hashes_unchanged'])
        self.assertTrue(result['current_code_snapshot_unchanged'])
        self.assertIn('sync_protocol', protocol)
        self.assertEqual(protocol['final_outcome']['status'], result['status'])
        return completed, result

    def test_real_os_exit_success_is_durable_before_close(self):
        with tempfile.TemporaryDirectory() as directory:
            completed, result = self.launch_close_child('reference', Path(directory)/'ref')
            self.assertEqual(completed.returncode, 0)
            self.assertEqual(result['status'], 'completed')
            self.assertEqual(result['completed_steps'], 2)
            self.assertEqual(result['intended_process_exit_code'], 0)
            self.assertGreater(result['timing']['physics_step_wall_seconds'], 0.)
            self.assertIn('final_score', result)
            self.assertIn('final_safety', result)
            self.assertEqual(result['sync_protocol']['calls']['fabric_true_physics_steps'], 50)

    def test_real_os_exit_zero_step_fabric_failure_keeps_diagnosis(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)/'no_fabric'
            completed, result = self.launch_close_child('fabric_unavailable', out)
            self.assertEqual(completed.returncode, 2, completed.stderr)
            self.assertEqual(result['status'], 'failed')
            self.assertEqual(result['completed_steps'], 0)
            self.assertEqual(result['failure_phase'], 'sync_preflight')
            self.assertEqual(result['reason_code'], 'fabric_interface_unavailable')
            self.assertFalse(result['sync_preflight_failure']['fabric_extension_enabled'])
            self.assertFalse(result['sync_preflight_failure']['fabric_interface_has_force_update'])
            self.assertTrue(result['sync_preflight_failure']['render_contains_force_update'])
            self.assertIn('SyncPreflightError', result['traceback'])
            self.assertFalse((out/'steps.jsonl').exists())
            failure = json.loads((out/'failure.json').read_text())
            self.assertEqual(failure['reason_code'], result['reason_code'])
            self.assertIn('fabric_interface_unavailable', completed.stderr)

    def test_real_os_exit_action_failure_keeps_primary_error_and_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            completed, result = self.launch_close_child('action_failure', Path(directory)/'action_error')
            self.assertEqual(completed.returncode, 2)
            self.assertEqual(result['status'], 'failed')
            self.assertEqual(result['failure_phase'], 'replaying_actions')
            self.assertIn('injected CPU test-double failure', result['error'])
            self.assertIn('failure_rule_evidence', result)
            self.assertEqual(result['active_action_index'], 0)
            self.assertAlmostEqual(result['physics_elapsed_seconds'], .1)

    def test_real_os_exit_pair_comparison_is_committed_before_close(self):
        with tempfile.TemporaryDirectory() as directory:
            reference = Path(directory)/'reference'
            self.launch_close_child('reference', reference)
            completed, result = self.launch_close_child('render_sync', Path(directory)/'opt', reference)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn('reference_comparison', result)
            self.assertEqual(result['reference_comparison']['max_joint_abs_rad'], 0.)
            self.assertTrue(result['reference_comparison']['final_score_equal'])

    def test_durable_json_fsyncs_file_and_parent_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'receipt.json'
            with mock.patch.object(b.os, 'fsync', wraps=os.fsync) as synced:
                b.write_json(path, {'committed': True}, durable=True)
            self.assertEqual(synced.call_count, 2)
            self.assertEqual(json.loads(path.read_text()), {'committed': True})


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--close-child':
        close_child(sys.argv[2],sys.argv[3],sys.argv[4] if len(sys.argv)>4 else None)
    else:
        unittest.main(verbosity=2)
