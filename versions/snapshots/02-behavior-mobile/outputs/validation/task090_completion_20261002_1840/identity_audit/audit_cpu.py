#!/usr/bin/env python3
"""Read bounded frozen TRAIN caches; no model, simulator, fitting or image-derived labels."""
import os
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
import hashlib
import json
import resource
import sys
import time
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path('/home/user/Robotic/project2')
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from scripts.build_task090_dino_localizer import cached_frame
from harness.task090_dino_localizer import Task090DinoLocalizer
started = time.monotonic()
def read(p):
    return json.loads(Path(p).read_text())
def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name, value):
    with (OUT / name).open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
prereg = read(OUT / 'preregister.json')
split_path = ROOT / 'data/behavior-1k/2026-challenge-demos/split_task_090.json'
split = read(split_path)
train = set(split['train_episode_indices'])
validation = set(split['validation_episode_indices']) if 'validation_episode_indices' in split else set(split.get('val_episode_indices', []))
assert {18000, 18001, 18002} <= train
assert not (train & validation)
artifact = ROOT / 'outputs/validation/task090_onboard_grasp_20261002/dino_localizer_v1/prototypes.npz'
localizer = Task090DinoLocalizer(artifact)
meta = localizer.meta
annotations_path = ROOT / 'configs/task090_perception_annotations.json'
annotations = read(annotations_path)
assert sha(annotations_path) == meta['source_annotation_sha256']
assert sha(artifact) == meta['artifact_sha256']
manifest_path = ROOT / 'outputs/validation/task090_preparation_20260930/data_runtime/train_observations/manifest.json'
manifest = read(manifest_path)
lookup = {(x['episode'], x['frame']): x for x in manifest['samples'] if x['camera'] == 'observation.rgb.zed_link_camera_0'}
row, col = np.divmod(np.arange(196), 14)
x, y = (col + .5) * 720 / 14, (row + .5) * 720 / 14
def mask(box):
    a,b,c,d = box
    return (x >= a) & (x < c) & (y >= b) & (y < d)
def patch(i, pos, neg):
    return dict(index=int(i), center_xy=[float(x[i]), float(y[i])], positive=float(pos[i]), negative=float(neg[i]), margin=float(pos[i]-neg[i]))
def summary(pos, neg):
    top = np.argsort(-pos, kind='stable')[:5]
    margin = pos-neg
    return dict(positive_peak=float(pos.max()), margin_peak=float(margin.max()), positive_margin_patch_count=int((margin > 0).sum()),
                original_top5=[patch(i,pos,neg) for i in top], margin_top5=[patch(i,pos,neg) for i in np.argsort(-margin,kind='stable')[:5]],
                original_top5_surviving_sign_veto=int((margin[top] > 0).sum()),
                original_top5_margin_min=float(margin[top].min()), original_top5_margin_mean=float(margin[top].mean()))
fit = cached_frame(18000,480)[:,5:]
fit /= np.maximum(np.linalg.norm(fit,axis=-1,keepdims=True),1e-8)
reconstruction={}
for name in ('positive','negative'):
    keep=mask(meta[name+'_roi']);p=fit[:,keep].mean(1);p/=np.linalg.norm(p,axis=-1,keepdims=True)
    reconstruction[name]={'patch_count':int(keep.sum()),'max_abs_difference':float(np.abs(p-getattr(localizer,name)).max())}
# Original fitting/old held-out checks and new preregistered checks kept separate.
samples=[dict(episode=18000,frame=480,role='prototype_fit_resubstitution',rois=[dict(kind='positive_fit',bbox_xyxy=meta['positive_roi']),dict(kind='negative_fit',bbox_xyxy=meta['negative_roi'])]),
         dict(episode=18001,frame=870,role='previously_used_prototype_holdout',rois=[dict(kind='previous_assistant_bin_bbox',bbox_xyxy=[485,548,720,720]),dict(kind='previous_assistant_cabinet_bbox',bbox_xyxy=[628,10,720,137]),dict(kind='negative_appliance',bbox_xyxy=[515,420,695,531])])]
samples += [{**v, 'role':'preregistered_train_check'} for v in prereg['check_samples']]
rows=[]
for s in samples:
    ep,frame=s['episode'],s['frame'];assert ep in train
    rgb=Path(lookup[ep,frame]['rgb']);assert Image.open(rgb).size==(720,720)
    pos,neg=localizer.scores(cached_frame(ep,frame));delta=pos-neg
    r={**s,'rgb':str(rgb),'rgb_sha256':sha(rgb),'scores':summary(pos,neg),'roi_statistics':[]}
    for roi in s['rois']:
        keep=mask(roi['bbox_xyxy']);ii=np.flatnonzero(keep)
        st={**roi,'patch_count':len(ii)}
        if len(ii):
            st.update(positive_mean=float(pos[ii].mean()),positive_min=float(pos[ii].min()),positive_max=float(pos[ii].max()),
                      negative_mean=float(neg[ii].mean()), margin_mean=float(delta[ii].mean()),margin_min=float(delta[ii].min()),margin_max=float(delta[ii].max()),
                      positive_margin_fraction=float(np.mean(delta[ii]>0)),
                      top5_positive_centers_in_roi=int(keep[np.argsort(-pos,kind='stable')[:5]].sum()),
                      top5_margin_centers_in_roi=int(keep[np.argsort(-delta,kind='stable')[:5]].sum()))
        r['roi_statistics'].append(st)
    rows.append(r)
save('labeled_score_comparison.json',rows)
# Bounded already-extracted TRAIN views. These rows have no identity/physical labels.
dense_path=ROOT/'outputs/validation/task090_onboard_grasp_20261002/dense_train_v1/frames.jsonl'
dense=[];skipped=[]
for text in dense_path.read_text().splitlines():
    s=json.loads(text);ep,frame=s['episode'],s['frame_index']
    assert ep in train and s['split']=='train' and s['physical_success_label'] is None
    try: features=cached_frame(ep,frame)
    except ValueError as err:
        if str(err)!='requested frame is not in the frozen cache':raise
        skipped.append([ep,frame]);continue
    pos,neg=localizer.scores(features);v=summary(pos,neg)
    dense.append(dict(episode=ep,frame=frame,identity_label=None,physical_success_label=None,**v))
save('unlabeled_dense_scores.json',dense)
def quant(values):
    return {str(q):float(np.quantile(values,q)) for q in (0,.25,.5,.75,1)}
aggregate={}
for ep in (18000,18001):
    rr=[r for r in dense if r['episode']==ep]
    aggregate[str(ep)]={'frames':len(rr),'all_original_top5_survive_zero_veto':sum(r['original_top5_surviving_sign_veto']==5 for r in rr),
                        'at_least_one_original_top5_survives_zero_veto':sum(r['original_top5_surviving_sign_veto']>0 for r in rr),
                        'positive_peak_quantiles':quant([r['positive_peak'] for r in rr]),'margin_peak_quantiles':quant([r['margin_peak'] for r in rr])}
result={'status':'cpu_scores_not_identity_admission','preregister_sha256':sha(OUT/'preregister.json'),
        'split_path':str(split_path),'split_sha256':sha(split_path),'train_count':len(train),'validation_count':len(validation),
        'fit_episode':18000,'old_check_episode':18001,'additional_check_episode':18002,
        'train_membership_verified':True,'train_validation_disjoint':not bool(train&validation),
        'prototype_reconstruction':reconstruction,'prototype_sha256':sha(artifact),'annotations_sha256':sha(annotations_path),
        'source_manifest_sha256':sha(manifest_path),'localizer_source_sha256':sha(ROOT/'harness/task090_dino_localizer.py'),
        'new_check_episode_not_in_color_or_dino_fit':18002 not in {a['episode'] for a in annotations['annotations']+annotations['negative_rois']} and meta['fit_episode']!=18002,
        'old_check_not_independent_of_all_color_calibration':any(a['episode']==18001 and a['frame']==870 for a in annotations['annotations']),
        'cache_encoder_config':read(ROOT/'data/dino_cache_task090/cache_config.json'),
        'dense_diagnostics':aggregate,'dense_skipped_uncached_frames':skipped,
        'physical_labels_available':0,'presence_admitted':False,'threshold_fitted':False,'gpu_used':False,
        'limitations':['Only appearance checks; no calibrated object presence or instance identity',
                       'Coarse bbox-center membership is not segmentation',
                       'Partial/clipped positive check has only two patch centers and uncertain boundaries',
                       'TRAIN18002 checks are excluded from this prototype/color fit, not report/test or WAM-training holdout',
                       'Cached payload integrity not rehashed in full; metadata links and artifact hash checked'],
        'resources':{'wall_seconds':time.monotonic()-started,'max_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'cpu_affinity_count':len(os.sched_getaffinity(0))}}
save('audit.json',result)
print(json.dumps({'output':str(OUT),'labeled':[{k:r[k] for k in ['episode','frame','role','roi_statistics']}|{'positive_peak':r['scores']['positive_peak'],'margin_peak':r['scores']['margin_peak'],'sign_survivors':r['scores']['original_top5_surviving_sign_veto']} for r in rows], 'aggregate':aggregate,'resources':result['resources']},ensure_ascii=False,indent=2))
