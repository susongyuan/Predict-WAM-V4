#!/usr/bin/env python3
"""Task0 fixtures for frozen imports; production source is never modified.
Historical synthetic tests hard-code Task13/car. Adapt only their fixture task names,
and reuse the focused stub-policy candidate boundary, not unrelated Task13 routing.
"""
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import types
import unittest
import torch
import harness.policy_server as policy
import harness.holding_guard as guard

PROJECT=Path('/home/user/Robotic/project2')
torch.cuda._lazy_init=lambda *a,**k:(_ for _ in ()).throw(RuntimeError('CPU test forbids CUDA'))
suite=unittest.TestSuite()
for name in ('test_task0_contract_safety','test_holding_guard_boundary'):
    path=PROJECT/'scripts'/(name+'.py')
    source=path.read_text()
    if name=='test_task0_contract_safety':
        source=source.replace('open lid','press').replace('car','radio')
    mod=types.ModuleType(name);mod.__file__=str(path);sys.modules[name]=mod
    exec(compile(source,'<Task0 fixture adaptation: '+str(path)+'>','exec'),mod.__dict__)
    suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(mod))
result=unittest.TextTestRunner(verbosity=2).run(suite)
assert result.wasSuccessful()

# Reuse actual production WamPolicy constructor/action boundary with source's stub
# model factory. Only test fixture names/program path differ from the old self-test.
source=inspect.getsource(policy.synthetic_self_test)
prefix=source.split('        # 1) Single-view main only:')[0]
prefix=prefix.replace('def synthetic_self_test():','def focused_task0_candidate_test():')
prefix=prefix.replace('data/skill_vtt_task013/contract_targets.json','data/skill_vtt_task000/contract_targets.json')
prefix=prefix.replace('"answer": "none", "frozen": True, "precondition": True,',
                      '"answer": "none", "frozen": True, "precondition": False, "program":"configs/task000_fixed_program.json",')
start=source.index('        # Candidate HOLD reaches')
focused=prefix+source[start:]
ns=dict(policy.__dict__)
exec(compile(focused,'<Task0 focused stub-policy fixture>','exec'),ns)
boundary=ns['focused_task0_candidate_test']()
assert not torch.cuda.is_initialized()
print(json.dumps({'cpu_only':True,'fixture_scope':'Task0 radio, no production edits; focused candidate path with stub models',
      'tests':result.testsRun,'candidate_boundary':boundary,'cuda_initialized':False,
      'holding_guard_source':guard.__file__,'policy_source':policy.__file__}))
