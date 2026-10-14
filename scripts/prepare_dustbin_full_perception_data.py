#!/usr/bin/env python3
"""CPU-only TRAIN RGB annotation pack; no labels, tracks or admission are fabricated."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from PIL import Image,ImageDraw
from harness.robotwin_dataset import RoboTwinEpisode,POLICY_VIEWS
from harness.robotwin_policy_worker import atomic_json,sha256


def main():
    split_path=ROOT/'data/benchmarks/robotwin2/put_bottles_dustbin_50_v1/split_manifest.json'
    split=json.loads(split_path.read_text())
    out=ROOT/'outputs/construction/dustbin_full_perception_train_20261007'
    out.mkdir(parents=True,exist_ok=False)
    names=split['splits']['train']
    if len(names)!=40 or set(names)&set(split['splits']['val']+split['splits']['heldout']):
        raise ValueError('disjoint forty-TRAIN split required')
    chosen=names[:24]
    protocol=dict(task='put_bottles_dustbin',source='official TRAIN onboard RGB only',
                  split_sha256=sha256(split_path),selection_rule='first 24 TRAIN episodes in frozen split order',
                  fit_episodes=chosen[:16],reserved_check_episodes=chosen[16:],
                  fractions=[0,.25,.5,.75,1],views=list(POLICY_VIEWS),
                  semantic_classes=['clear_blue_label_bottle','yellow_bottle','dark_red_label_bottle','dustbin'],
                  annotation_schema='bbox_xyxy_pixels plus visible/occluded/out_of_view/uncertain and optional sensor track link; unannotated is not a negative',
                  intended_model='frozen DINO patch features plus small CPU/GPU target head and temporal association',
                  runtime_uses_evaluator_gt=False,third_view_used=False,
                  fit_or_admission_run=False,physical_success_verifier=False,
                  warning='Pixel boxes and tracks alone cannot prove grasp, release, full containment or stability. Reserved check labels must remain unseen by fitting/tuning.')
    atomic_json(out/'protocol.json',protocol)
    cases=[]
    initial=[]
    for i,name in enumerate(chosen):
        path=Path(split['data_root'])/name
        ep=RoboTwinEpisode(path)
        directory=out/ep.episode_id;directory.mkdir()
        frames=sorted(set(round((len(ep)-1)*f) for f in protocol['fractions']))
        for frame in frames:
            sample=ep.chunk(frame,1)
            for view in POLICY_VIEWS:
                image_path=directory/f'{view}_frame{frame:06d}.png'
                Image.fromarray(sample['images'][view]).save(image_path)
                cases.append(dict(episode=name,frame=frame,view=view,
                    partition='fit' if i<16 else 'reserved_check',image=str(image_path),
                    image_sha256=sha256(image_path),source_hdf_sha256=sha256(path) if frame==0 and view==POLICY_VIEWS[0] else None,
                    annotation_status='unannotated',objects=None))
                if frame==0 and view==POLICY_VIEWS[0]:initial.append((name,image_path))
    for start in (0,16):
        rows=initial[start:start+16]
        sheet=Image.new('RGB',(1280,4*270),'white');draw=ImageDraw.Draw(sheet)
        for j,(name,path) in enumerate(rows):
            x=(j%4)*320;y=(j//4)*270
            sheet.paste(Image.open(path),(x,y+24));draw.text((x+4,y+4),name,fill='black')
        sheet.save(out/('fit_initial_gallery.png' if start==0 else 'reserved_initial_gallery.png'))
    atomic_json(out/'annotation_template.json',dict(format='onboard-pixel-annotations-v1',cases=cases,
        status='unannotated',admitted=False))
    atomic_json(out/'result.json',dict(status='prepared_unannotated',episodes=24,fit=16,reserved_check=8,
        RGB_images=len(cases),heldout_used=False,model_trained=False,provider_admitted=False))
    print(json.dumps(dict(status='prepared_unannotated',images=len(cases),path=str(out))))


if __name__=='__main__':main()
