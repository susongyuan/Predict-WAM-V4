#!/usr/bin/env python3
"""Offline evidence audit, never a policy or replacement task scorer."""
import argparse
import json
from pathlib import Path
import numpy as np

def audit(path):
    report={'source':str(path),'rows':0,'usable':0,'unknown':0,'mismatches':[],
            'native_true_steps':[],'diagnostic_error_rows':0,'scope':'saved prefix only; not task completion or policy input'}
    for line in Path(path).read_text().splitlines():
        r=json.loads(line);report['rows']+=1
        d=r.get('native_diagnostics') or {};p=d.get('native_predicate_evidence') or {}
        if d.get('errors'):report['diagnostic_error_rows']+=1
        names=['object_position_world','cabinet_functional_point_world','origin_z','selected_gripper_command']
        if any(p.get(k) is None for k in names):report['unknown']+=1;continue
        types=p['operand_dtypes'];kinds=p['operand_scalar_kinds']
        def scalar(name,value):
            return value if kinds.get(name)=='python' else np.asarray(value,dtype=types[name])[()]
        obj=np.array(p['object_position_world'],dtype=types['object_position_world'])
        target=np.array(p['cabinet_functional_point_world'],dtype=types['cabinet_functional_point_world'])
        origin=scalar('origin_z',p['origin_z']);side=p['selected_gripper_arm']
        g=scalar(side+'_gripper_command',p['selected_gripper_command'])
        # Frozen native task expression, independent of saved conditions/recomputed bool.
        dz=obj[2]-origin
        value=bool(np.all(abs(obj[:2]-target[:2])<np.array([.05,.05])) and dz>.007 and dz<.12 and g>.8)
        report['usable']+=1
        if p.get('native_predicate') is None:report['unknown']+=1
        elif value != p['native_predicate']:
            report['mismatches'].append({'control_step':r.get('control_step'),'settle_step':r.get('settle_step'),'recomputed':value,'native':p['native_predicate']})
        if p.get('native_predicate') is True:report['native_true_steps'].append(r.get('control_step'))
    return report
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('source');parser.add_argument('--output',required=True);a=parser.parse_args()
    report=audit(a.source);Path(a.output).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
    raise SystemExit(bool(report['mismatches']))
