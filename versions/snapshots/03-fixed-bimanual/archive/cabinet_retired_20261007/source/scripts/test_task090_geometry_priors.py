#!/usr/bin/env python3
"""CPU sanity checks for explicit Task90 static-size proposal filters."""
import ast
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from harness.task090_perception import geometry_sanity, detect_from_observation


def box(size, offset=(1, .2, .5)):
    xyz=np.array([[x,y,z] for x in (-.5,.5) for y in (-.5,.5) for z in (-.5,.5)])
    return xyz*np.asarray(size)+offset


class GeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg=json.loads((ROOT/'configs/task090_perception_geometry_v2.json').read_text())
        cls.base=json.loads((ROOT/'configs/task090_perception.json').read_text())

    def prior(self,label):
        return self.cfg['classes'][label]['geometry_prior']

    def test_appearance_parameters_unchanged(self):
        for name, values in self.base['classes'].items():
            self.assertEqual({k:v for k,v in self.cfg['classes'][name].items() if k!='geometry_prior'},values)
        self.assertFalse(self.cfg.get('relations_admitted',False))

    def test_asset_provenance_and_positive_shapes(self):
        for label in self.cfg['classes']:
            p=self.prior(label)
            self.assertEqual(hashlib.sha256((ROOT/p['asset']).read_bytes()).hexdigest(),p['asset_sha256'])
            self.assertTrue(geometry_sanity(box(p['native_bbox_m']),p),label)

    def test_planar_floor_texture_rejected(self):
        for label in ('half','half_pomegranate'):
            self.assertFalse(geometry_sanity(box([.1,.1,.0001],offset=(1,.2,0)),self.prior(label)))

    def test_large_surface_proposals_rejected(self):
        for label in self.cfg['classes']:
            self.assertFalse(geometry_sanity(box([2,1,1]),self.prior(label)))
        self.assertFalse(geometry_sanity(box([.8,.8,.3]),self.prior('trash_can')))

    def test_yaw_rotation_of_fruit_is_not_rejected(self):
        rot=np.array([[2**-.5,-2**-.5,0],[2**-.5,2**-.5,0],[0,0,1]])
        for label in ('half','half_pomegranate'):
            p=self.prior(label)
            self.assertTrue(geometry_sanity(box(p['native_bbox_m'])@rot.T,p))

    def test_nonfinite_or_missing_observation_rejected(self):
        self.assertFalse(geometry_sanity(np.full((5,3),np.nan),self.prior('half')))
        self.assertFalse(geometry_sanity(np.zeros((0,3)),self.prior('half')))
        self.assertEqual(detect_from_observation(None,None,None,None),([],None))
        self.assertTrue(geometry_sanity(box([3,3,3]),None))  # opt-in, legacy proposal behavior retained

    def test_runtime_calibration_is_fingerprinted(self):
        tree=ast.parse((ROOT/'scripts/behavior_closed_loop.py').read_text())
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='runtime_artifact_paths')
        import behavior_preflight
        with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,{},clear=True):
            tmp=Path(tmp);stats=tmp/'stats.json';stats.write_text('{}')
            scope={'ROOT':ROOT,'CONFIG':behavior_preflight.CONFIG,'Path':Path,'json':json,'os':os,
                   'task_profile':SimpleNamespace(TASK_ID=90,VTT_DIR=tmp)}
            exec(compile(ast.Module(body=[fn],type_ignores=[]),'<actual_manifest_function>','exec'),scope)
            args=SimpleNamespace(checkpoint=str(tmp/'ck.pt'),orchestrate=json.dumps({'perception':{'detector':{'calibration_path':str(ROOT/'configs/task090_perception_geometry_v2.json')}}}),program=None,memory_vtt=None,env_wrapper=None)
            cfg={'dataset':{'stats_path':str(stats)},'model':{'vision_encoder':{'checkpoint_path':str(tmp)}}}
            paths=scope['runtime_artifact_paths'](args,['--robot-config',str(tmp/'robot.yaml')],cfg)
            self.assertIn(ROOT/'configs/task090_perception_geometry_v2.json',paths)

    def test_current_training_sources_not_changed_by_runtime_patch(self):
        m=json.loads((ROOT/'outputs/training/task090_v3_50k_20260930_s010000/manifest.json').read_text())
        for p in ('harness/task090_perception.py','scripts/behavior_closed_loop.py','configs/task090_orchestrate.json','configs/task090_perception_geometry_v2.json'):
            self.assertNotIn(str(ROOT/p),m['source_sha256'])


if __name__=='__main__':unittest.main(verbosity=2)
