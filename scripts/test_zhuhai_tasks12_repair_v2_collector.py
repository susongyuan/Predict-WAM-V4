#!/usr/bin/env python3
"""CPU-only checks for the separately versioned task1 v2 collector."""
from contextlib import redirect_stderr
from io import StringIO
import ast
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from scripts import collect_isaac_zhuhai_tasks12_repair as old
from scripts import collect_isaac_zhuhai_tasks12_repair_v2 as new


class V2CollectorTests(unittest.TestCase):
    def args(self, parent):
        inv=Path(parent)/'inventory.json';inv.write_text('{}')
        return new.parser().parse_args(['--task','zhuhai_task1','--mode','randomized','--count','100',
            '--seed-start','974000','--out',str(Path(parent)/'data'),'--prior-inventory',str(inv)])

    def contract(self,args,module=new):
        with patch.object(module,'robot_binding',return_value=dict(robot={},robot_contract={},robot_contract_sha256='test')):
            return module.contract(args)

    def test_only_task1_can_be_started(self):
        with tempfile.TemporaryDirectory() as t, redirect_stderr(StringIO()):
            with self.assertRaises(SystemExit):
                new.parser().parse_args(['--task','zhuhai_task2'])
            self.assertEqual(self.args(t).task,'zhuhai_task1')

    def test_original_live_collector_unchanged(self):
        root=Path(new.ROOT)
        receipt=json.loads((root/'outputs/zhuhai2026_tasks12/production_task2_collection_only_20261009_v11/source_snapshot/receipt.json').read_text())
        key='scripts/collect_isaac_zhuhai_tasks12_repair.py'
        self.assertEqual(old.sha256(old.__file__),receipt['files'][key]['sha256'])
        self.assertNotEqual(old.__file__,new.__file__)

    def test_bind_both_v2_and_inherited_v1(self):
        binding=new.expert_dependency_binding('zhuhai_task1')
        self.assertEqual({Path(p).name for p in binding}, {'isaac_zhuhai_task1_expert_repair_v1.py','isaac_zhuhai_task1_expert_repair_v2.py'})
        self.assertTrue(all(len(h)==64 for h in binding.values()))

    def test_source_physics_contract_not_relabelled(self):
        with tempfile.TemporaryDirectory() as t:
            a=self.args(t); before=self.contract(a,old); after=self.contract(a,new)
        for field in new.SHARED_CONTRACT_FIELDS:
            self.assertEqual(before[field],after[field],field)
        self.assertNotEqual(before['expert_variant'],after['expert_variant'])
        self.assertNotEqual(before['collection_implementation'],after['collection_implementation'])
        self.assertIn('expert_dependency_sha256',after)

    def test_missing_or_changed_dependency_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            c=self.contract(self.args(t))
        with patch.object(new,'validate_native16_contract'):
            new.validate_header(c,'zhuhai_task1',verify_files=False)
            for change in ('missing','changed'):
                bad=copy.deepcopy(c)
                if change=='missing':bad.pop('expert_dependency_sha256')
                else:bad['expert_dependency_sha256'][str(new.ROOT/'harness/isaac_zhuhai_task1_expert_repair_v1.py')]='0'*64
                with self.assertRaisesRegex(ValueError,'inherited v1 changed'):
                    new.validate_header(bad,'zhuhai_task1',verify_files=False)

    def test_quotas_failures_resource_and_recycle_policies_preserved(self):
        for name in ('PRODUCTION_QUOTAS','PRIOR_SUCCESS_COUNTS','PRIOR_COLLECTOR_PATHS','RESOURCE_POLICY','WORKER_RECYCLING','MINIMUM_FREE_BYTES','SHARED_CONTRACT_FIELDS'):
            self.assertEqual(getattr(old,name),getattr(new,name),name)
        self.assertEqual(new.PRIOR_SUCCESS_COUNTS['zhuhai_task1'],48)
        self.assertEqual(new.PRODUCTION_QUOTAS['randomized'],100)

    def test_existing_validator_and_writer_bodies_preserved(self):
        def functions(file):
            return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(Path(file).read_text()).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
        a,b=functions(old.__file__),functions(new.__file__)
        allowed={'parser','contract','validate_header'}
        changed={k for k in a if a[k]!=b[k]}
        # The runtime import resides in collect() in this collector; inspect its
        # AST narrowly below instead of accepting arbitrary writer changes.
        for name in changed-allowed:
            before=a[name].replace('isaac_zhuhai_task1_expert_repair_v1','isaac_zhuhai_task1_expert_repair_v2')
            self.assertEqual(before,b[name],name)
        for name in ('_success_rows','prior_rows','residual_families','source_families','validate_manifest','consecutive_failures'):
            self.assertEqual(a[name],b[name],name)


if __name__=='__main__':
    unittest.main()
