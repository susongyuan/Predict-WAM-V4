#!/usr/bin/env python3
"""CPU AST-isolated status-RPC binding test; no simulator/model imports."""
import ast
from pathlib import Path
import sys
from types import SimpleNamespace, ModuleType
import unittest
from unittest.mock import Mock, patch
SOURCE=Path(__file__).resolve().parents[1]/"third_party/BEHAVIOR-1K-isaac6/OmniGibson/omnigibson/eval/evaluator.py"
class PolicyEvidenceTests(unittest.TestCase):
    def check(self, task, policy, status=None):
        tree=ast.parse(SOURCE.read_text())
        cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='Evaluator')
        func=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='policy_evidence')
        ns={'TASK_NAMES_TO_INDICES':{'turning_on_radio':0,'picking_up_trash':1,'loading_the_car':13}}
        exec(compile(ast.Module(body=[func],type_ignores=[]),str(SOURCE),'exec'),ns)
        client=Mock();client.get_server_metadata.return_value={'policy':policy,'seed':0,'rng_reset_behavior':'once'}
        client._packer.pack.side_effect=lambda x:x
        client._ws.recv.return_value={'episode':2,'step':3225} if status is None else status
        self.client=client
        obj=SimpleNamespace(policy=SimpleNamespace(policy=client),cfg=SimpleNamespace(task=SimpleNamespace(name=task)))
        mod=ModuleType('omnigibson.eval.utils.network_utils');mod.unpackb=lambda x:x
        with patch.dict(sys.modules,{'omnigibson.eval.utils.network_utils':mod}):
            return ns['policy_evidence'](obj)
    def test_current_task_status_read_for_0_1_13(self):
        for task,i in [('turning_on_radio',0),('picking_up_trash',1),('loading_the_car',13)]:
            row=self.check(task,f'lilawam_behavior_task{i}')
            self.assertEqual((row['policy_episode'],row['policy_step']),(2,3225))
            self.client._ws.send.assert_called_once_with({'status':True})
    def test_other_policy_not_queried(self):
        row=self.check('turning_on_radio','lilawam_behavior_task13')
        self.assertIsNone(row['policy_episode']);self.client._ws.send.assert_not_called()
    def test_bad_status_fails(self):
        for status in [{'episode':True,'step':3},{'episode':2,'step':-1},{'episode':2}]:
            with self.assertRaises(ValueError):self.check('turning_on_radio','lilawam_behavior_task0',status)
if __name__=='__main__':unittest.main()
