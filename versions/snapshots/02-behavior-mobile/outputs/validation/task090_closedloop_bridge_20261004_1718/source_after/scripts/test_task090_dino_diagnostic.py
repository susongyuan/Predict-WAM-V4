"""CPU tests of cached-DINO localization and strictly passive runtime diagnostics."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
import time
import unittest
import tempfile

import numpy as np
from PIL import Image

from harness.task090_dino_localizer import Task090DinoLocalizer
from harness.task090_observation import Task090ObservationProducer
from scripts.build_task090_dino_localizer import cached_frame
from scripts.test_task090_observer_lifecycle import policy, contract, pick, observation

ROOT=Path(__file__).resolve().parents[1]
ARTIFACT=ROOT/'outputs/validation/task090_onboard_grasp_20261002/dino_localizer_v1/prototypes.npz'
CALIBRATION=ROOT/'configs/task090_perception_geometry_v2.json'


class DinoDiagnostic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.localizer=Task090DinoLocalizer(ARTIFACT)
        cls.features=cached_frame(18001,870)
        p=ROOT/'outputs/validation/task090_first_grasp_evidence_20261001/extracted/frames.jsonl'
        with p.open() as file:
            cls.sample=next(r for r in map(json.loads,file) if r['episode']==18001 and r['frame_index']==870)

    def observation(self):
        r=self.sample
        return {'robot_r1::robot_r1:zed_link:Camera:0::rgb':np.asarray(Image.open(r['rgb'])),
                'robot_r1::robot_r1:zed_link:Camera:0::depth_linear':np.load(r['depth_m'],allow_pickle=False),
                'robot_r1::cam_rel_poses':np.asarray(r['camera_pose7'])}

    def test_heldout_peak_is_a_bin_candidate_not_success(self):
        pos,_=self.localizer.scores(self.features)
        i=int(np.argmax(pos)); x=(i%14+.5)*720/14; y=(i//14+.5)*720/14
        self.assertTrue(485<=x<720 and 548<=y<720)
        p=Task090ObservationProducer(CALIBRATION,dino_artifact=ARTIFACT)
        result=p.update(self.observation(),self.sample['state61'],episode=18001,step=870,head_features=self.features)
        self.assertEqual(result['status'],'measurements_available')
        self.assertLess(result['detections'][0]['xyz_base'][2],1.)
        self.assertFalse(result['verified']);self.assertFalse(result['admitted'])
        self.assertFalse(result['detections'][0]['object_presence_verified'])

    def test_prefix_tokens_do_not_change_scores(self):
        before=self.localizer.scores(self.features)
        changed=self.features.copy();changed[:,:5]=1e6
        after=self.localizer.scores(changed)
        np.testing.assert_array_equal(before[0],after[0])
        np.testing.assert_array_equal(before[1],after[1])

    def test_shared_inputs_are_not_modified(self):
        obs=self.observation(); saved={k:v.copy() for k,v in obs.items()}
        state=np.asarray(self.sample['state61']);state_before=state.copy()
        features=self.features.copy();features_before=features.copy()
        p=Task090ObservationProducer(CALIBRATION,dino_artifact=ARTIFACT)
        p.update(obs,state,episode=18001,step=870,head_features=features)
        for k,v in saved.items():np.testing.assert_array_equal(obs[k],v)
        np.testing.assert_array_equal(state,state_before)
        np.testing.assert_array_equal(features,features_before)

    def test_missing_current_features_fail_closed(self):
        p=Task090ObservationProducer(CALIBRATION,dino_artifact=ARTIFACT)
        result=p.update(self.observation(),self.sample['state61'],episode=18001,step=870)
        self.assertEqual(result['reason'],'missing_current_dino_features')
        self.assertFalse(result['verified'])

    def test_wrong_feature_shape_rejected(self):
        with self.assertRaises(ValueError):self.localizer.scores(np.zeros((3,196,1024)))


class PassiveHook(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tree=ast.parse((ROOT/'harness/policy_server.py').read_text())
        method=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_log_task090_measurement')
        snapshot=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_save_onboard_snapshot')
        ns={'time':time,'np':np}
        exec(compile(ast.Module(body=[method,snapshot],type_ignores=[]),'<passive-hook>','exec'),ns)
        cls.log=staticmethod(ns[method.name])
        cls.snapshot=staticmethod(ns[snapshot.name])

    def fake(self, reply):
        rows=[]
        def update(obs,state,**kwargs):
            state[0]=999  # even a mutating test producer must not change the action's original state
            return reply
        fake=policy(steps=[pick()], producer=SimpleNamespace(update=update, max_frame_gap=5, reset=lambda: None))
        fake.step_in_episode=30
        contract(fake, 0)
        fake._emit=rows.append
        fake.queue=[1,2]
        fake._last_post_t1={'original':True}
        return fake,rows

    def test_no_control_return_or_state_mutation(self):
        fake,rows=self.fake({'status':'unknown','verified':False,'admitted':False})
        state=np.zeros(61)
        self.assertIsNone(self.log(fake,observation(),state,[]))
        self.assertEqual(state[0],0);self.assertEqual(fake.queue,[1,2]);self.assertEqual(fake._last_post_t1,{'original':True})
        self.assertTrue(rows[0]['diagnostic_only'])

    def test_success_claim_becomes_diagnostic_error(self):
        fake,rows=self.fake({'verified':True,'admitted':True})
        self.log(fake,observation(),np.zeros(61),[])
        self.assertEqual(rows[0]['measurement']['status'],'unknown')
        self.assertTrue(rows[0]['measurement']['diagnostic_error'])
        self.assertIn('cannot certify success', rows[0]['measurement']['reason'])
        self.assertFalse(rows[0]['measurement']['verified'])

    def test_previous_action_is_passed_without_modification(self):
        fake,rows=self.fake({})
        fake._last_action=np.zeros(23,np.float32);fake._last_action[22]=-1.
        captured=[]
        def update(obs,state,**kwargs):
            captured.append(kwargs['action'].copy());kwargs['action'][22]=999
            return {'status':'unknown','verified':False,'admitted':False}
        fake.onboard_measurement=SimpleNamespace(update=update,requires_applied_action=True,
            max_frame_gap=5,reset=lambda: None)
        self.log(fake,observation(),np.zeros(61),[])
        self.assertEqual(captured[0][22],-1.)
        self.assertEqual(fake._last_action[22],-1.)
        self.assertFalse(rows[0]['measurement']['verified'])

    def test_snapshot_only_sensors_and_no_overwrite(self):
        fake,rows=self.fake({})
        fake._find=lambda obs,suffix,contains=None: next(obs[k] for k in obs if k.endswith(suffix) and (contains is None or contains in k))
        obs={'robot_r1::zed_link::depth_linear':np.ones((8,8),np.float32),
             'robot_r1::cam_rel_poses':np.zeros(21,np.float32), 'privbase_nav::secret':object()}
        with tempfile.TemporaryDirectory() as tmp:
            self.snapshot(fake,Path(tmp),obs,np.zeros(61,np.float32))
            path=Path(tmp)/'sensors/ep2_step000030.npz'
            with np.load(path,allow_pickle=False) as data:
                self.assertEqual(set(data.files),{'head_depth_m','camera_pose','proprio','episode','observation_step','skill_key','scope'})
                self.assertEqual(int(data['observation_step']),30)
            original=path.read_bytes()
            self.snapshot(fake,Path(tmp),obs,np.zeros(61,np.float32))
            self.assertEqual(path.read_bytes(),original)
            self.assertEqual(rows[-1]['event'],'onboard_snapshot_error')

    def test_new_frame_observed_even_outside_old_logging_interval(self):
        fake,rows=self.fake({'status':'unknown','verified':False,'admitted':False})
        fake.step_in_episode=31
        self.log(fake,observation(),np.zeros(61),[])
        self.assertEqual(len(rows),1)
        self.assertNotIn('diagnostic_error',rows[0]['measurement'])
        self.assertEqual(rows[0]['step'],31)


if __name__=='__main__':unittest.main()
