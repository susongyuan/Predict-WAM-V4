#!/usr/bin/env python3
"""Small CPU contract checks; NOT a physical/detector success benchmark."""
import io
import json
import os
from pathlib import Path
import sys
import unittest

os.environ.setdefault('P2_TASK_ID', '90')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from harness import task090_perception as perception
from harness.task1_perception_adapter import TaskPerception, detector_module_name
from scripts.build_task090_perception import OUT, read_annotations, save, sha


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x1-x0) * max(0, y1-y0)
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    return intersection / max(1, union)


class PerceptionCPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cv2.setNumThreads(1)
        cls.rows, cls.provenance = read_annotations()
        cls.cfg = json.loads((ROOT/'configs/task090_orchestrate.json').read_text())
        manifest = json.loads(Path(cls.provenance['manifest']).read_text())
        cls.sample = next(s for s in manifest['samples'] if s['episode']==18000
                          and s['frame']==1830 and 'zed' in s['camera'])
        cls.rgb = next(im for a, im in cls.rows if a.get('label')=='half' and a['episode']==18000)
        cls.state = np.asarray(cls.sample['state61'])
        cls.pose = np.asarray(cls.sample['camera_pose7'])

    def test_route_and_fail_closed_modes(self):
        from harness.task090_predicates import validate_mode, Task090Postconditions
        validate_mode(90, self.cfg)
        adapter = TaskPerception(self.cfg['perception'])
        self.assertEqual(adapter.module_name, 'harness.task090_perception')
        self.assertIs(adapter._module(), perception)
        self.assertEqual(detector_module_name(task_id=1), 'harness.object_perception')
        self.assertEqual(detector_module_name(task_id=0), 'harness.radio_perception')
        for override in ({'postconditions':'task1_physical'}, {'candidate_outcome_policy':'advance'},
                         {'manip_executor':'primitive'}, {'skip_failed_objects':True}):
            with self.assertRaises(ValueError):
                validate_mode(90, {**self.cfg, **override})
        self.assertFalse(Task090Postconditions().admission_status()['admitted'])
        self.assertTrue(self.cfg['frozen'])
        self.assertIn('furniture_sink', perception.capability_status()['unsupported'])

    def test_missing_observations_are_unknown(self):
        self.assertEqual(TaskPerception(self.cfg['perception']).detect({}, self.state), ([], None))
        for rgb, depth, pose in ((None, None, None), (self.rgb, None, self.pose),
                                 (self.rgb, np.ones((10,10)), self.pose),
                                 (self.rgb, np.full((720,720), np.nan), self.pose),
                                 (self.rgb, np.ones((720,720)), None),
                                 (self.rgb, np.ones((720,720)), np.zeros(7))):
            self.assertEqual(perception.detect_from_observation(rgb, depth, pose, self.state), ([], None))
        self.assertIsNone(self.sample['depth'])  # TRAIN has no matched depth; never fit metric truth here

    def test_train_roi_resubstitution_and_negatives(self):
        results = []
        for a, image in self.rows:
            ds = perception.appearance_candidates(image)
            # Store full-frame proposals, including known negative overlaps; never drop failures.
            result = {k:a[k] for k in ('episode','frame','rgb_sha256','bbox_xyxy')}
            result['annotation_label'] = a.get('label', 'hard_negative')
            result['proposals'] = [{k:d[k] for k in ('label','bbox_xyxy','pixels')} for d in ds]
            if 'label' in a:
                result['best_matching_iou'] = max([iou(a['bbox_xyxy'], d['bbox_xyxy'])
                                                  for d in ds if d['label']==a['label']] or [0.])
                self.assertGreater(result['best_matching_iou'], .3)
            else:
                x0,y0,x1,y1 = a['bbox_xyxy']
                result['negative_description'] = a['description']
                result['negative_pixel_overlaps'] = [d['label'] for d in ds if np.any(
                    (d['pixel_x']>=x0)&(d['pixel_x']<x1)&(d['pixel_y']>=y0)&(d['pixel_y']<y1))]
            results.append(result)
        save(OUT/'train_roi_recheck.json', {'kind':'TRAIN fit resubstitution; not held-out accuracy',
             'rows':results, 'false_positives_expected':True, 'detector_admitted':False,
             'metric_validation':False, 'uses_dev_test_hidden_truth':False})

    def test_full_frame_not_fixed_roi(self):
        moved = np.roll(self.rgb, -400, axis=1)
        proposals = perception.appearance_candidates(moved)
        target = [87,431,179,510]
        self.assertGreater(max([iou(target,d['bbox_xyxy']) for d in proposals
                                if d['label']=='half'] or [0.]), .3)

    def test_surface_sample_only_with_synthetic_depth(self):
        # Synthetic sensor plumbing ONLY. This constant is not a TRAIN observation/calibration.
        depth = np.ones(self.rgb.shape[:2], np.float32)
        detections, visibility = perception.detect_from_observation(self.rgb, depth, self.pose, self.state)
        self.assertIsNone(visibility)
        self.assertTrue(detections)
        cam = perception.Camera(*perception.camera_from_pose(self.pose), 720, 720)
        for d in detections:
            u,v = d['surface_pixel_uv']
            np.testing.assert_allclose(d['xyz_base'], cam.backproject(u,v,depth[v,u]))
            self.assertEqual(d['point_semantics'], 'observed_surface_sample')
            self.assertFalse(d['center_valid'])
            self.assertFalse(d['metric_calibration_admitted'])
            self.assertFalse(d['grasp_reference_valid'])
            self.assertFalse(d['relations_admitted'])
            self.assertNotIn('relation',d)
            self.assertNotIn('held',d)
            self.assertEqual(d['held_status'],'unknown')
            if d['label']=='half':
                self.assertEqual(d['physical_category'],'half_banana')

    def test_train_program_and_explicit_docking_blockers(self):
        from harness.orchestrator import fixed_program_candidates, budgets_from_program
        from harness.grounding import Catalog, compile_programs
        from harness.docking_controller import load_calibration
        candidates, _ = fixed_program_candidates(ROOT/'configs/task090_fixed_program.json')
        catalog = Catalog(ROOT/'configs/task090_skill_program.json')
        self.assertTrue(all(catalog.allows(s) for s in candidates[0].steps))
        compiled, rejected = compile_programs(candidates, catalog)
        self.assertEqual(rejected, [])
        self.assertEqual(len(compiled), 1)
        self.assertEqual(len(compiled[0].steps), 12)
        self.assertEqual(len(candidates[0].steps),12)
        triples = [s.objects for s in candidates[0].steps if s.skill=='place on next to']
        self.assertEqual(triples,[('trash_can','floors','furniture_sink')])
        self.assertIn('pick up from(half,countertop)',budgets_from_program(ROOT/'configs/task090_skill_program.json'))
        dock = load_calibration()
        self.assertIn('task090',dock['source'])
        missing = sorted({s.objects[0] for s in candidates[0].steps if s.skill=='move to'} - set(dock['dock']))
        self.assertEqual(missing,['furniture_sink','half'])
        save(OUT/'routing_limits.json', {'program_parse_passed':True, 'admitted':False,
             'missing_docking_keys':missing,
             'banana_note':'TRAIN alias half vs docking half_banana requires later checked alias routing; no copied/guessed reference inserted.',
             'sink_note':'No observed sink or TRAIN docking reference; unsupported. Normal flow HOLDs at unverified bin pick before reaching sink.',
             'calibration_note':'Loaded EEF-at-closure references are not matched to detector surface samples.'})


if __name__=='__main__':
    stream=io.StringIO()
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(PerceptionCPU))
    text=stream.getvalue()
    print(text,end='')
    save(OUT/'cpu_validation.json', {'tests_run':result.testsRun,'failures':len(result.failures),
         'errors':len(result.errors),'passed':result.wasSuccessful(),'log':text,
         'mode':'CPU only, synthetic depth plumbing and TRAIN resubstitution',
         'gpu_started':False,'complete_task_validated':False,'postconditions_admitted':False,
         'perception_sha256':sha(ROOT/'harness/task090_perception.py')})
    sys.exit(0 if result.wasSuccessful() else 1)
