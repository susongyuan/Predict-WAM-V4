#!/usr/bin/env python3
"""Meaningful CPU admission checks; no claimed physical recovery success."""
from pathlib import Path
import json,sys,tempfile,unittest
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import numpy as np
from harness.zhuhai_dagger import DaggerEpisode,DaggerPrecisionDataset,StallTrigger,finish_mask
from harness.zhuhai_demo_recording import StreamDemoRecorder
from harness.zhuhai_dataset import SourceEpisode
from harness.zhuhai_robot_spec import HOME_ACTION,ARM_INDICES
from harness.wam_event_clock import episode_event_features
from scripts.test_zhuhai_pipeline import rule_state


def fixture(directory):
    p=Path(directory)/'mixed.hdf5'
    metadata=dict(task='zhuhai_task2',family_id='TRAIN-fixture',split='train')
    rec=StreamDemoRecorder(p,metadata=metadata,frequency=10.)
    for i in range(9):
        q=np.asarray(HOME_ACTION,np.float32);q[0]+=.001*i
        command=np.asarray(HOME_ACTION,np.float32);command[0]+=.01*i
        command[7]=0. if i<4 else 1.
        rgb=np.full((16,16,3),i,np.uint8)
        rec.capture(dict(rgb={k:rgb for k in ('head_camera','left_camera','right_camera')},
            measured_joint_gripper_state=q,commanded_joint_gripper_state=command),i*.1,rule_state(i==8,i*.1))
    finish_mask(rec,[0,0,0,1,1,1,1,1],['policy']*3+['expert']*5,3)
    rec.finish(final_evidence=rule_state(True,.8),successful=True)
    return dict(path=str(p),split='train',family_id='TRAIN-fixture')


class Store:
    def get(self,ep,t):
        return np.zeros((3,3,201,1024),np.uint16),np.full((3,2048),t,np.float32)


class Tests(unittest.TestCase):
    def test_old_unmasked_loader_rejects_new_format(self):
        with tempfile.TemporaryDirectory() as d:
            row=fixture(d)
            with self.assertRaisesRegex(ValueError,'source format'):SourceEpisode(row)
            ep=DaggerEpisode(row);self.assertEqual(ep.supervised_count,5)

    def test_expert_labels_only_but_full_causal_prefix_retained(self):
        with tempfile.TemporaryDirectory() as d:
            row=fixture(d);ds=DaggerPrecisionDataset('zhuhai_task2',[row],Store(),np.ones(1024))
            self.assertEqual(len(ds),5);s=ds[0];ep=ds.episodes[0]
            self.assertEqual(s['anchor'],3)
            np.testing.assert_array_equal(s['action_sequence'][:5],ep.action[3:])
            np.testing.assert_array_equal(s['state'][1],ep.state[1])
            self.assertEqual(float(s['pool'][0,-1,0]),0.)
            with __import__('h5py').File(row['path'],'r') as f:initial=f['target_command'][0].copy()
            expected=episode_event_features(ep.state[:4],ep.action[:4],initial_target=initial)[-1]
            np.testing.assert_array_equal(s['event_state'],expected)
            self.assertGreater(float(s['event_state'][0]),0.)

    def test_short_suffix_masks_future_and_idm_without_fake_labels(self):
        with tempfile.TemporaryDirectory() as d:
            ds=DaggerPrecisionDataset('zhuhai_task2',[fixture(d)],Store(),np.ones(1024))
            s=ds[len(ds)-1]
            self.assertEqual(int(s['action_valid_mask'].sum()),1)
            self.assertFalse(s['future_valid_mask']);self.assertFalse(s['future_valid_mask_8']);self.assertFalse(s['idm_valid_mask'])
            self.assertTrue(np.all(s['action_sequence'][1:]==0));self.assertTrue(np.all(s['idm_target_raw']==0))

    def test_bad_authority_and_nontrain_rows_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            row=fixture(d)
            with self.assertRaisesRegex(ValueError,'TRAIN'):DaggerEpisode({**row,'split':'heldout'})
            import h5py
            with h5py.File(row['path'],'r+') as f:f['expert_action_mask'][0]=True
            with self.assertRaisesRegex(ValueError,'boundaries'):DaggerEpisode(row)

    def test_progress_gate_waits_and_never_intervenes_after_full_success(self):
        def evidence(grasp=False,place=False,full=False):
            return dict(physics_safety=dict(passed=True),score=dict(full_task_success=full,
                items=dict(bowl=dict(grasp_awarded=grasp,placement_valid=place))))
        gate=StallTrigger();self.assertIsNone(gate.observe(100,evidence()))
        self.assertIsNone(gate.observe(220,evidence(True,True)))
        self.assertIsNone(gate.observe(300,evidence(True,True)))
        self.assertEqual(gate.observe(400,evidence(True,True)),'no_new_verified_grasp_or_placement')
        self.assertIsNone(gate.observe(500,evidence(True,True,True)))

    def test_handoff_brakes_without_resetting_inherited_velocity(self):
        from harness.zhuhai_official_braking import BrakingRateLimiter,protocol
        control=BrakingRateLimiter(protocol()['rate_limit']);q=np.asarray(HOME_ACTION,float)
        control.reset(q);q=q.copy();q[0]+=.2
        control.step(q);before=control.velocity.copy();self.assertGreater(before[0],0.)
        for _ in range(10):
            goal=np.asarray(HOME_ACTION,float);goal[list(ARM_INDICES)]=control.previous
            accepted,stats=control.step(goal)
            self.assertGreaterEqual(stats['minimum_braking_surplus_rad'],-1e-10)
        np.testing.assert_array_equal(control.velocity,np.zeros(14))


if __name__=='__main__':unittest.main()
