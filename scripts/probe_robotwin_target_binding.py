#!/usr/bin/env python3
"""TRAIN-only visual target-binding validation, independent of task evaluation."""
import argparse,json,time,hashlib,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from PIL import Image
from harness.robotwin_integrated_vlm import Backend
PROMPT='''Identify the ONE small movable task object to be put into the cabinet drawer. This task's object starts on the foreground working area of the table, in front of the cabinet, near either the left or the right robot gripper. The cabinet, fixed handle and robot are not the target. Items farther back near the sides of the cabinet are background clutter, not the target. This location description is the task request, not evidence that grasping or placement has succeeded.
Images are HEAD, LEFT_WRIST, RIGHT_WRIST in that order. Use wrist images only as supporting context. Identify the target in HEAD. A partly cut-off object can be identified as partial; do not invent its missing outline. If no unique visible foreground target can be identified, return unknown with null box.
Return ONLY JSON with exactly: status (identified or unknown), description (short appearance description), head_bbox_1000 (four integer xmin,ymin,xmax,ymax, each in 0..1000, normalized to HEAD image width and height, or null), visibility (full, partial or unknown), evidence (one short visible clue). No plans, actions, guessed force/contact, task success or hidden object IDs.'''
def iou(a,b):
    xa=max(a[0],b[0]);ya=max(a[1],b[1]);xb=min(a[2],b[2]);yb=min(a[3],b[3]);inter=max(0,xb-xa)*max(0,yb-ya);area=lambda x:max(0,x[2]-x[0])*max(0,x[3]-x[1]);return inter/(area(a)+area(b)-inter) if area(a)+area(b)-inter else 0

def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu-lock-owner',required=True);p.add_argument('--out',required=True);p.add_argument('--single-head-short',action='store_true');a=p.parse_args();out=Path(a.out);out.mkdir(exist_ok=False)
    base=ROOT/'outputs/validation/target_binding_train_20261006';sel=json.loads((base/'selection.json').read_text());gold=json.loads((base/'annotations.json').read_text())
    prompt=PROMPT
    if a.single_head_short:
        prompt='Locate the small movable foreground object on the table IN FRONT OF the blue cabinet, near the bottom left or bottom right of this HEAD camera image. Ignore robot parts, cabinet, and background clutter beside/behind the cabinet. Return ONLY JSON: status (identified or unknown), description (appearance), head_bbox_1000 ([xmin,ymin,xmax,ymax] normalized 0..1000 to this image, or null), visibility (full or partial or unknown), evidence (brief visible cue). A visible part is sufficient for a partial box.'
        sel['episodes']=sel['episodes'][:4]
    protocol={'prompt':prompt,'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'annotation_sha256':hashlib.sha256((base/'annotations.json').read_bytes()).hexdigest(),'episodes':sel['episodes'],'split':'TRAIN_only_first8_development_last8_reserved_check','success_criterion':'identified box IoU >=0.3 with independent visible annotation; no task-success claim','model':'existing Qwen7B NF4','no_GT_or_gold_in_model_prompt':True,'registered_before_model_load':True}
    (out/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n');backend=Backend(a.gpu_lock_owner);results=[]
    for i,name in enumerate(sel['episodes']):
        d=base/name.removesuffix('.hdf5');images=[Image.open(d/(v+'.png')).convert('RGB') for v in ('head','left','right')];t=time.monotonic();raw,details=backend.complete(prompt,images[:1] if a.single_head_short else images)
        try:
            text=raw.strip()
            if text.startswith('```json\n') and text.endswith('\n```'):text=text[8:-4]
            value=json.loads(text);box=value.get('head_bbox_1000');valid=value.get('status')=='identified' and isinstance(box,list) and len(box)==4 and all(type(x)is int and 0<=x<=1000 for x in box) and box[0]<box[2] and box[1]<box[3]
            score=iou(box,gold['annotations'][i]['head_bbox_1000']) if valid else 0
        except Exception as e:value=None;score=0
        r={'index':i,'episode':name,'partition':'development' if i<8 else 'reserved_check','raw':raw,'parsed':value,'iou':score,'box_pass':score>=.3,'seconds':time.monotonic()-t,'generation':details,'resources':backend.resources()};results.append(r)
        with (out/'results.jsonl').open('a') as f:f.write(json.dumps(r,ensure_ascii=False)+'\n')
        print('CASE',i,'IoU',round(score,3),'PASS',score>=.3,flush=True)
    summary={'completed':True,'count':len(results),'development_passes':sum(x['box_pass'] for x in results[:8]),'reserved_check_passes':sum(x['box_pass'] for x in results[8:]),'not_admission_for_holding_or_completion':True};(out/'result.json').write_text(json.dumps(summary,indent=2)+'\n');print(summary)
if __name__=='__main__':main()
