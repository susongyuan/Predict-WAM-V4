#!/usr/bin/env python3
"""Read only first-frame structure of the downloaded candidate with restricted pickle globals.
No arbitrary pickle classes, GPU, scene, policy, conversion, or fabricated RGB.
"""
import hashlib, json, pickle
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT/'data/benchmarks/robotwin_fruit/pick_apple_messy_early_raw'
OUT = ROOT/'outputs/verification/robotwin_fruit_20261007'
class ArrayUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        allowed = {('numpy.core.multiarray', '_reconstruct'): np._core.multiarray._reconstruct,
                   ('numpy', 'ndarray'): np.ndarray, ('numpy', 'dtype'): np.dtype}
        if (module, name) not in allowed:
            raise pickle.UnpicklingError(f'not allowlisted: {module}.{name}')
        return allowed[module, name]
def main():
    rows=[]
    root=RAW/'source/pick_apple_messy_pkl'
    for ep in sorted(root.iterdir(),key=lambda p:int(p.name[7:])):
        frames=sorted(ep.glob('*.pkl'),key=lambda p:int(p.stem))
        with frames[0].open('rb') as f: obs=ArrayUnpickler(f).load()
        cameras=obs['observation']
        image_fields={v:[k for k in d if k.lower() in ('rgb','color','colors','image','images')] for v,d in cameras.items()}
        rows.append({'episode':ep.name,'frames':len(frames),'first_frame':str(frames[0].relative_to(ROOT)),
          'sha256':hashlib.sha256(frames[0].read_bytes()).hexdigest(),
          'camera_fields':{v:list(d) for v,d in cameras.items()},'image_fields':image_fields,
          'pointcloud_shape':list(obs['pointcloud'].shape),'joint_action_shape':list(obs['joint_action'].shape),
          'target_visible_in_policy_first_rgb':None,'reason':'No RGB image payload in first-frame pickle; calibration matrices and sampled pointcloud are not the WAM RGB observation.'})
    receipt={'status':'not_admitted','candidate_task':'pick_apple_messy','user_task_match_confirmed':False,
      'download_verified':json.loads((RAW/'SOURCE.json').read_text())['status']=='complete',
      'episodes':len(rows),'frames':sum(r['frames'] for r in rows),
      'episodes_with_first_rgb':sum(any(r['image_fields'].values()) for r in rows),
      'first_frame_target_visibility':'unverifiable_missing_rgb','native_simulation_performed':False,
      'training_started':False,'rows':rows}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'candidate_first_frame_audit.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k!='rows'},ensure_ascii=False,indent=2))
if __name__=='__main__': main()
