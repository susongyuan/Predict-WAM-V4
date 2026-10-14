#!/usr/bin/env python3
"""CPU regression: acquire targets after exploration; reject above-bin furniture proposals."""
import ast
import json
from pathlib import Path
import sys
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import harness.docking_controller as dc
from harness.task090_perception import geometry_sanity, observed_floor_plane


def box(size, center):
    return np.array([[x,y,z] for x in (-.5,.5) for y in (-.5,.5) for z in (-.5,.5)])*size+center


class ExplorationTests(unittest.TestCase):
    def fixture(self):
        pose={'trunk':np.zeros(4),'larm':np.zeros(7),'rarm':np.zeros(7)}
        cal={'dock':{'trash_can':{'hand':'right','target':np.array([.6,-.2])}},
             'posture':{'nav_before_bin':pose,'nav_carrying_bin':pose,'pick_start_trash_can':pose},'gripper':{}}
        cfg=dc.DockConfig(plan_dock_pose=False,require_posture=False,avoid_obstacles=False)
        ctl=dc.DockingController(cal,cfg);odom=dc.Odometry();m=dc.ObjectMap(max_age_steps=90)
        ctl.start('trash_can','before_bin',odom);ctl.mode='explore';ctl._explore=(np.zeros(2),0.0);ctl.search_turns=1.1
        for t in range(3):m.update([{'label':'trash_can','xyz_base':[2,0,.5]}],odom,t)
        return ctl,m,odom,np.zeros(61)

    def test_previous_mode_bug_reproduced(self):
        ctl,m,odom,state=self.fixture()
        path=ROOT/'outputs/validation/task090_diagnostic_repairs_20261001/exploration/source_before/docking_controller.py'
        cls=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='DockingController')
        step=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='step')
        scope=dict(vars(dc));exec(compile(ast.Module(body=[step],type_ignores=[]),str(path),'exec'),scope)
        _,status=scope['step'](ctl,state,m,odom)
        self.assertEqual(status['mode'],'explore')

    def test_new_target_leaves_exploration(self):
        ctl,m,odom,state=self.fixture()
        action,status=ctl.step(state,m,odom)
        self.assertEqual(status['mode'],'approach')
        self.assertIsNone(ctl._explore);self.assertEqual(ctl.search_turns,0)
        self.assertEqual(action.shape,(23,));self.assertTrue(np.isfinite(action).all())

    def test_no_target_can_keep_exploring(self):
        ctl,m,odom,state=self.fixture();m.reset();m.advance(0)
        _,status=ctl.step(state,m,odom)
        self.assertEqual(status['mode'],'explore')

    def test_floor_plane_uses_depth_not_a_world_pose(self):
        class Camera:
            def backproject(self,u,v,z):
                x,y=np.asarray(u)*.01,np.asarray(v)*.01
                return np.c_[x,y,.01*x+.02*y-.005]
        p=observed_floor_plane(np.ones((64,64)),Camera())
        self.assertIsNotNone(p)
        np.testing.assert_allclose(p['coefficients'],[.01,.02,-.005],atol=1e-12)
        self.assertIsNone(observed_floor_plane(np.zeros((64,64)),Camera()))

    def test_floor_bin_height_bounds(self):
        cfg=json.loads((ROOT/'configs/task090_perception_floor_v3.json').read_text())
        p=cfg['classes']['trash_can']['geometry_prior'];size=np.array(p['native_bbox_m'])
        floor={'coefficients':np.array([0,0,-.005]),'margin_m':.006}
        self.assertTrue(geometry_sanity(box(size,[1,0,size[2]/2-.005]),p,floor))
        furniture=box(np.array([.3,.4,.3]),[1,0,1.])
        without_floor={k:v for k,v in p.items() if k!='max_height_above_floor_m'}
        self.assertTrue(geometry_sanity(furniture,without_floor))
        self.assertFalse(geometry_sanity(furniture,p,floor))
        self.assertFalse(geometry_sanity(box(size,[1,0,size[2]/2]),p,None))

    def test_budget_has_fixed_controller_derived_allowance(self):
        import math
        cfg=dc.DockConfig()
        target=json.loads((ROOT/'data/skill_vtt_task090/contract_targets.json').read_text())
        median=target['navigation']['move to(trash_can)']['pick up from(trash_can,floors)']['median_frames']
        expected=int(median*1.5)+cfg.lost_steps+math.ceil(2*math.pi*cfg.explore_after_turns/(cfg.w_search*dc.DT))+math.ceil(cfg.explore_distance/(cfg.explore_speed*dc.DT))+math.ceil(math.pi/(cfg.w_max*dc.DT))
        actual=json.loads((ROOT/'configs/task090_orchestrate.json').read_text())
        self.assertEqual(actual['min_budget_steps']['move to'],expected)
        self.assertEqual(expected,1873)

    def test_color_parameters_unchanged(self):
        a=json.loads((ROOT/'configs/task090_perception.json').read_text())
        b=json.loads((ROOT/'configs/task090_perception_floor_v3.json').read_text())
        for key,value in a['classes'].items():
            self.assertEqual(value,{k:v for k,v in b['classes'][key].items() if k!='geometry_prior'})


if __name__=='__main__':unittest.main(verbosity=2)
