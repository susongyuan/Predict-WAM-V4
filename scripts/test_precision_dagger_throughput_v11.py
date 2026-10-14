#!/usr/bin/env python3
"""Actual policy collision exclusion and complete scene-evidence preservation."""
from pathlib import Path
import ast
import copy
import json
import tempfile
import unittest
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from harness.zhuhai_dagger_parallel import GIB,ResourcePressure
from harness.zhuhai_dagger_throughput_v11 import failure_fields,write_result,ResourceAdmission,audited_prior_policy_failure


class InfrastructureError(Exception):pass


class Regression(unittest.TestCase):
    def test_policy_unsafe_excluded_expert_and_unknown_still_halt(self):
        safety=dict(passed=False,events=[dict(reason='non_adjacent_self_collision')])
        exc=InfrastructureError('unsafe actual physics')
        fields,code=failure_fields(exc,262,0,safety,InfrastructureError)
        self.assertEqual((fields['status'],code),('unsafe_policy_episode',0))
        self.assertEqual((fields['accepted_for_training'],fields['quota_credit']),(False,0))
        self.assertFalse(fields['all_observations_safe'])
        for authority,step,actual,exception in [(1,262,safety,exc),(0,0,safety,exc),
                (0,262,None,exc),(0,262,dict(passed=False,events=[]),exc),
                (0,262,safety,ValueError('physics_safety'))]:
            fields,code=failure_fields(exception,step,authority,actual,InfrastructureError)
            self.assertEqual((fields['status'],code),('infrastructure_error',2))
        fields,code=failure_fields(ResourcePressure('reserve'),0,0,None,InfrastructureError)
        self.assertEqual((fields['status'],code),('resource_paused',0))

    def test_live_small_terminal_complete_scene_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);scene=dict(vertices=[list(range(200)) for _ in range(20)])
            row=dict(status='policy_rollout',scene_manifest=scene,seed=1)
            writes=[]
            def atomic(path,value):
                writes.append(str(path));Path(path).write_text(json.dumps(value))
            for step in range(20):
                row['control_steps']=step;write_result(root,row,atomic)
            live=json.loads((root/'result.json').read_text())
            self.assertNotIn('scene_manifest',live)
            self.assertEqual(json.loads((root/'scene_manifest.json').read_text()),scene)
            self.assertEqual(writes.count(str(root/'scene_manifest.json')),1)
            write_result(root,row,atomic,terminal=True)
            final=json.loads((root/'result.json').read_text())
            self.assertEqual(final['scene_manifest'],scene)
            self.assertNotIn('live_scene_manifest_in_sidecar',final)

    def test_measured_third_admission_and_fourth_remains_conditional(self):
        admission=ResourceAdmission(lambda:0.)
        row=dict(available=12*GIB,total=32*GIB,gpu_free_bytes=7*GIB,swap_fraction=.82)
        self.assertTrue(admission.allowed(row,2))
        row['available']=10*GIB;self.assertFalse(admission.allowed(row,2))
        row['available']=16*GIB;self.assertTrue(admission.allowed(row,3))
        row['swap_fraction']=.87;self.assertFalse(admission.allowed(row,3))
        row['swap_fraction']=.95;self.assertFalse(admission.allowed(row,0))
        row['swap_fraction']=.82;row['gpu_free_bytes']=5*GIB
        self.assertFalse(admission.allowed(row,2))

    def test_real_failure_audit_rejects_other_episode_or_expert(self):
        path=ROOT/'outputs/zhuhai2026_tasks12/verification/dagger50_segmented_20261010_v11/policy_failure_audit.json'
        audit=json.loads(path.read_text());raw=json.loads(Path(audit['result_path']).read_text())
        plan=dict(prior_policy_failure_audit=str(path))
        self.assertTrue(audited_prior_policy_failure(plan,raw,audit['result_path'],True))
        self.assertFalse(audited_prior_policy_failure(plan,raw,audit['result_path'],False))
        other=copy.deepcopy(raw);other['expert_control_steps']=1
        with self.assertRaises(ValueError):audited_prior_policy_failure(plan,other,audit['result_path'],True)

    def test_collector_action_path_unchanged(self):
        def tree(path):
            source=path.read_text().split('    except BaseException as exc:\n')[0]
            source=source.replace('from harness.zhuhai_dagger_resource_v10 import outer_failure_fields','')
            source=source.replace('from harness.zhuhai_dagger_throughput_v11 import failure_fields, write_result','')
            source=source.replace('write_result(out, row, base.atomic_json, terminal=True)',"base.atomic_json(out/'result.json', row)")
            source=source.replace('write_result(out, row, base.atomic_json)',"base.atomic_json(out/'result.json', row)")
            return ast.dump(ast.parse(source+'    except BaseException:\n        raise\n'))
        self.assertEqual(tree(ROOT/'scripts/collect_precision_dagger_segmented_v10.py'),
                         tree(ROOT/'scripts/collect_precision_dagger_segmented_v11.py'))


if __name__=='__main__':unittest.main()
