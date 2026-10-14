#!/usr/bin/env python3
"""Regression of the real zero-action resource stop and conservative admission."""
from pathlib import Path
import copy
import json
import tempfile
import unittest
from unittest.mock import patch
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from harness.zhuhai_dagger_parallel import GIB, ResourcePressure
from harness.zhuhai_dagger_resource_v10 import ResourceAdmission, outer_failure_fields, audited_resource_pause
from scripts import supervise_precision_dagger50k_v10 as supervisor


class Regression(unittest.TestCase):
    def test_resource_type_only(self):
        fields, code=outer_failure_fields(ResourcePressure('reserve reached'),0)
        self.assertEqual((fields['status'],code),('resource_paused',0))
        self.assertFalse(fields['accepted_for_training'])
        fields, code=outer_failure_fields(ValueError('ResourcePressure: unsafe physics'),0)
        self.assertEqual((fields['status'],code),('infrastructure_error',2))

    def test_actual_old_evidence_and_exact_identity(self):
        path=ROOT/'outputs/zhuhai2026_tasks12/verification/dagger50_segmented_20261010_v10/resource_stop_audit.json'
        audit=json.loads(path.read_text()); raw=json.loads(Path(audit['result_path']).read_text())
        plan=dict(prior_resource_audit=str(path))
        self.assertTrue(audited_resource_pause(plan,raw,audit['result_path'],True))
        self.assertFalse(audited_resource_pause(plan,raw,audit['result_path'],False))
        unsafe=copy.deepcopy(raw);unsafe['physics_safety_at_failure']=dict(passed=False)
        with self.assertRaises(ValueError):audited_resource_pause(plan,unsafe,audit['result_path'],True)
        other=copy.deepcopy(raw);other['seed']+=1
        with self.assertRaises(ValueError):audited_resource_pause(plan,other,audit['result_path'],True)

    def test_cooldown_and_expansion_without_guard_relaxation(self):
        now=[0.];admission=ResourceAdmission(lambda:now[0])
        row=dict(available=24*GIB,total=32*GIB,gpu_free_bytes=10*GIB,swap_fraction=.89)
        self.assertTrue(admission.allowed(row,0))
        self.assertFalse(admission.allowed(row,1))
        row['swap_fraction']=.50
        self.assertTrue(admission.allowed(row,3))
        row['available']=14*GIB
        self.assertFalse(admission.allowed(row,1))
        row['available']=24*GIB
        admission.pressure();self.assertFalse(admission.allowed(row,0))
        now[0]=181.;self.assertTrue(admission.allowed(row,0))
        self.assertFalse(admission.allowed(row,1))
        now[0]=1801.;self.assertTrue(admission.allowed(row,1))
        row['swap_fraction']=.95;self.assertFalse(admission.allowed(row,0))
        row['swap_fraction']=.93;self.assertFalse(admission.allowed(row,0))

    def test_successful_physical_path_is_identical(self):
        old=(ROOT/'scripts/collect_precision_dagger_segmented_v8.py').read_text()
        new=(ROOT/'scripts/collect_precision_dagger_segmented_v10.py').read_text()
        old=old.split('    except BaseException as exc:\n')[0]
        new=new.replace('from harness.zhuhai_dagger_resource_v10 import outer_failure_fields\n','')
        self.assertEqual(old,new.split('    except BaseException as exc:\n')[0])

    def test_halt_receipt_uses_fresh_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'collection').mkdir();(root/'training').mkdir()
            cm=dict(status='halted',success_count=12,active_participants=[],error='resource infrastructure')
            (root/'collection/manifest.json').write_text(json.dumps(cm))
            (root/'training/status.json').write_text(json.dumps(dict(status='halted',step=30000)))
            plan=dict(schema='regression',out=str(root/'out'),collection=str(root/'collection'),
                training=str(root/'training'),collection_unit='collection',training_unit='training',
                evaluation_out=str(root/'untested'),source_files={},bound_inputs={},maximum_seconds=60)
            state=lambda unit:dict(unit=unit,MainPID='0',ActiveState='failed',SubState='failed')
            with patch.object(supervisor,'unit_state',state),patch.object(supervisor,'snapshot',lambda:{}), \
                 patch.object(supervisor,'publish_documents') as publish,patch.object(supervisor,'log'):
                with self.assertRaises(ValueError):supervisor.execute(plan)
                self.assertEqual(publish.call_args.args[0]['collection'],cm)
            receipt=json.loads((root/'out/status.json').read_text())
            self.assertEqual(receipt['collection']['active_participants'],[])
            self.assertEqual(receipt['evaluation'],{})
            self.assertEqual(receipt['formal_step'],30000)


if __name__=='__main__':unittest.main()
