#!/usr/bin/env python3
import argparse,json,time,sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from PIL import Image,ImageDraw
from harness.robotwin_integrated_vlm import Backend
from harness.robotwin_target_binding import PROMPT,SIZE,parse_box
from scripts.probe_robotwin_target_binding import iou

def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu-lock-owner',required=True);p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out);out.mkdir(exist_ok=False);base=ROOT/'outputs/validation/target_binding_train_20261006'
    original=json.loads((base/'selection.json').read_text());gold=json.loads((base/'annotations.json').read_text());reserve=json.loads((base/'reserved_annotations.json').read_text());cases=[]
    for i,n in enumerate(original['episodes']):cases.append({'episode':n,'partition':'development','gold':gold['annotations'][i]['head_bbox_1000']})
    for item in reserve['annotations']:cases.append({'episode':item['episode'],'partition':'reserved_check','gold':item['head_bbox_1000']})
    protocol={'prompt':PROMPT,'coordinate_contract':'absolute_pixels_448x336_actual_processor_grid_must_be_1x24x32','cases':cases,'split':'TRAIN_only','new_reserved_check_not_used_in_prompt_development':True,'no_gold_in_model_input':True,'only_visible_body_boxes_not_task_success':True,'source':'Qwen official coordinates https://qwenlm.github.io/blog/qwen2.5-vl/'}
    (out/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n');backend=Backend(a.gpu_lock_owner);records=[]
    for i,c in enumerate(cases):
        img=Image.open(base/c['episode'].removesuffix('.hdf5')/'head.png').convert('RGB').resize(SIZE,Image.Resampling.BICUBIC);t=time.monotonic();raw,details=backend.complete(PROMPT,[img]);error=None
        try:
            parsed=parse_box(raw,details['image_grid_thw']);box=[v/SIZE[i%2]*1000 for i,v in enumerate(parsed['bbox_2d'])] if parsed else None
            score=iou(box,c['gold']) if box and c['gold'] else None;correct_unknown=box is None and c['gold'] is None
        except Exception as exc:parsed=box=score=None;correct_unknown=False;error=str(exc)
        r={**c,'raw':raw,'parsed':parsed,'box_1000':box,'iou':score,'correct_unknown':correct_unknown,'passed':correct_unknown or score is not None and score>=.3,'error':error,'generation':details,'seconds':time.monotonic()-t};records.append(r)
        with (out/'results.jsonl').open('a') as f:f.write(json.dumps(r,ensure_ascii=False)+'\n')
        draw=ImageDraw.Draw(img)
        if c['gold']:draw.rectangle([v/1000*SIZE[j%2] for j,v in enumerate(c['gold'])],outline='lime',width=2)
        if parsed:draw.rectangle(parsed['bbox_2d'],outline='red',width=2)
        img.save(out/(c['episode'].removesuffix('.hdf5')+'.png'))
        print('CASE',i,c['partition'],'PASS',r['passed'],'IoU',score,'ERROR',error,flush=True)
    summary={'completed':True,'development':{'count':16,'passes':sum(r['passed'] for r in records[:16])},'reserved':{'count':16,'passes':sum(r['passed'] for r in records[16:])},'physical_verifier_admitted':False}
    (out/'result.json').write_text(json.dumps(summary,indent=2)+'\n')
if __name__=='__main__':main()
