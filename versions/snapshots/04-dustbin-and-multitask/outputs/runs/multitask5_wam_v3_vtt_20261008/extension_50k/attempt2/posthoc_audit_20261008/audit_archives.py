#!/usr/bin/env python3
"""CPU-only post-hoc audit. No simulator, no new rollouts, no video generation.
Original action/score/onboard/replay artifacts are read-only inputs.
"""
from pathlib import Path
import argparse, collections, hashlib, json, math
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import av

def read(p): return json.loads(Path(p).read_text())
def rows(p): return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024), b''): h.update(b)
    return h.hexdigest()
def ident(p):
    p=Path(p).resolve(); s=p.stat()
    return dict(path=str(p),size=s.st_size,mtime_ns=s.st_mtime_ns,sha256=sha(p))
def digest(v): return hashlib.sha256(json.dumps(v,sort_keys=True,allow_nan=False).encode()).hexdigest()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--run',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); a=ap.parse_args()
    a.out.mkdir(parents=True,exist_ok=True); source=a.run/'eval/summary.json'; summary=read(source)
    result=dict(format='robotwin-posthoc-audit-v1',created_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'),
        source_summary=ident(source),simulation_started=False,policy_executed=False,original_evidence_modified=False,
        new_video_generated=False,new_camera_created=False,contact_force_or_measured_joint_log_available=False,
        limitations=['Action/frame boundary completeness is not physical task completion.',
            'Only archived control-boundary RGBs are inspected; no missing physics substeps or real pressure signals can be reconstructed.',
            'Visual sheets support human diagnosis, not a replacement success score.'],episodes=[],replays=[],sheets=[])
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
    bold=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',20)
    frame_cache={}
    sampled_sources={}
    def image(row,step,view='head_camera'):
        onboard=Path(row['native_output'])/row['episode']['episode']/'onboard'
        p=onboard/f'observation_{step:06d}.npz'
        key=(str(p),view)
        if key not in frame_cache:
            with np.load(p,allow_pickle=False) as d: frame_cache[key]=Image.fromarray(d[view].copy()).resize((320,240))
            if str(p) not in sampled_sources: sampled_sources[str(p)]=ident(p)
        return frame_cache[key]
    def sheet(name, title, cells, cols=4):
        w,h=320,266; canvas=Image.new('RGB',(cols*w,46+math.ceil(len(cells)/cols)*h),(18,24,31)); draw=ImageDraw.Draw(canvas)
        draw.text((8,12),title,font=bold,fill=(240,242,245))
        for i,(row,step,view,label) in enumerate(cells):
            x,y=(i%cols)*w,46+(i//cols)*h
            draw.text((x+4,y+3),label,font=font,fill=(133,222,174) if row['native_success'] else (255,182,144))
            canvas.paste(image(row,step,view),(x,y+26))
        p=a.out/(name+'.jpg'); canvas.save(p,quality=95)
        result['sheets'].append(dict(path=str(p),title=title,cells=[dict(task=r['task'],seed=r['seed'],step=s,view=v,label=l) for r,s,v,l in cells]))
    tasks=list(dict.fromkeys(r['task'] for r in summary['episodes']))
    for row in summary['episodes']:
        ep=row['episode']; p=Path(row['native_output'])/ep['episode']; n=ep['control_steps']; index=rows(p/'onboard/frames.jsonl'); acts=rows(p/'actions.jsonl')
        submitted=[x for x in acts if x['status']=='submitted']; ack=[x for x in acts if x['status']=='native_returned']; checks={}
        checks['action_pairs_complete']=len(acts)==2*n and len(submitted)==len(ack)==n
        checks['action_steps_in_order']=[x['step'] for x in ack]==list(range(n)) and [x['step'] for x in submitted]==list(range(n))
        checks['native_counts_match']=[x['native_take_action_count'] for x in ack]==list(range(1,n+1))
        checks['actions_unchanged_across_ack']=len(ack)==len(submitted) and all(x['bounded_action']==y['bounded_action'] for x,y in zip(submitted,ack))
        checks['frame_boundaries_complete']=len(index)==n+1 and all(x['index']==i and x['control_step']==i and x['settle_step']==0 and x['file']==f'observation_{i:06d}.npz' for i,x in enumerate(index))
        checks['referenced_archive_files_exist']=all((p/'onboard'/x['file']).is_file() for x in index)
        checks['original_evidence_hashes_match']=all(sha(f['path'])==f['sha256'] for f in row['verification']['evidence_files'])
        checks['archive_manifest_complete']=read(p/'onboard/manifest.json')['complete'] is True
        checks['no_settle']=ep['settle_diagnostic']['completed_physics_steps']==0
        actions=np.asarray([x['bounded_action'] for x in ack]); clips=np.asarray([x['clip_mask'] for x in ack],dtype=bool)
        arm=[0,1,2,3,4,5,7,8,9,10,11,12]
        delta=np.linalg.norm(np.diff(actions[:,arm],axis=0),axis=1)
        gt=rows(p/'evaluation_gt.jsonl')
        item=dict(task=row['task'],seed=row['seed'],native_success=row['native_success'],status=row['status'],control_steps=n,
            frames=len(index),action_acknowledgments=len(ack),checks=checks,all_checks_pass=all(checks.values()),
            topp_fallback_steps=sum(bool(x['topp_fallback_observed']) for x in ack),
            topp_status_counts=dict(collections.Counter(t['status'] for x in ack for t in x['topp'])),
            arm_clip_steps=int(np.any(clips[:,arm],axis=1).sum()),gripper_clip_steps=int(np.any(clips[:,[6,13]],axis=1).sum()),
            policy_replans=sum(bool(x['policy']['replanned']) for x in ack),execution_prefixes=sorted(set(x['policy']['execution_prefix'] for x in ack)),
            last_gripper_commands=actions[-1,[6,13]].tolist(),
            arm_action_delta_norm_median=float(np.median(delta)),last100_arm_action_delta_norm_median=float(np.median(delta[-100:])),
            task_diagnostics_available=any(x.get('diagnostics_available') is True for x in gt),
            actual_contact_points_logged=any(x.get('contact_points') is not None for x in gt),
            gt_first=gt[0],gt_last=gt[-1])
        result['episodes'].append(item)
        if not row['native_success']:
            steps={'put_bottles_dustbin':[0,50,100,150,200,250,300,400,500,700,1000,n],
                'place_cans_plasticbox':[0,30,60,90,120,150,200,250,300,400,600,n],
                'open_laptop':[0,20,40,60,80,100,150,200,300,400,550,n]}[row['task']]
            sheet(f"{row['task']}_{row['seed']}_failure_timeline",f"{row['task']} | seed {row['seed']} | native FAIL",
                [(row,s,'head_camera',f'step {s:04d} / {n}') for s in steps])
        print(json.dumps(dict(task=row['task'],seed=row['seed'],checks=item['all_checks_pass'],fallback=item['topp_fallback_steps'])),flush=True)
    for task in tasks:
        rr=[r for r in summary['episodes'] if r['task']==task]
        for kind in ['initial','terminal']:
            sheet(f'{task}_{kind}_all10',f'{task} | {kind.upper()} | fixed 10 seeds',
                [(r,0 if kind=='initial' else r['episode']['control_steps'],'head_camera',f"{r['seed']} {'PASS' if r['native_success'] else 'FAIL'} step {0 if kind=='initial' else r['episode']['control_steps']}") for r in rr],cols=5)
        first=next(r for r in rr if r['native_success']); n=first['episode']['control_steps']
        sheet(f'{task}_{first["seed"]}_native_success_timeline',f'{task} | seed {first["seed"]} | native PASS only',
            [(first,int(s),'head_camera',f'step {int(s):04d} / {n}') for s in np.linspace(0,n,12).round().astype(int)])
        if task in ['click_bell','press_stapler']:
            steps=list(range(max(0,n-10),n+1,2)); cells=[]
            for view in ['head_camera','left_camera','right_camera']:
                cells.extend([(first,s,view,f'{view} step {s}') for s in steps])
            sheet(f'{task}_{first["seed"]}_terminal_detail',f'{task} | last 11 steps | same original RGB',cells,cols=6)
        provenance=read(a.run/'success_replays'/f'{task}_seed_{first["seed"]}_provenance.json')
        video=Path(provenance['video']); decoded=0
        with av.open(str(video)) as container:
            stream=container.streams.video[0]
            for _ in container.decode(stream): decoded+=1
            fps=float(stream.average_rate)
        onboard=Path(first['native_output'])/first['episode']['episode']/'onboard'
        archive=[ident(onboard/x['file']) for x in rows(onboard/'frames.jsonl')]
        result['replays'].append(dict(task=task,seed=first['seed'],control_steps=n,original_observations=len(archive),
            decoded_frames=decoded,expected_frames=n+1+round(fps*provenance['final_still_seconds']),fps=fps,
            matches_expected_frames=decoded==n+1+round(fps*provenance['final_still_seconds']),
            video_hash_matches=sha(video)==provenance['video_sha256'],
            original_observation_hashes_match=digest(archive)==provenance['source_observation_aggregate_sha256'],
            observations_cover_every_control_boundary=True,continuous_physics_recording=False,post_success_physics_steps=0))
    result['sampled_observation_sources']=list(sampled_sources.values())
    result['totals']=dict(episodes=len(result['episodes']),native_successes=sum(x['native_success'] for x in result['episodes']),
        failed_episodes=sum(not x['native_success'] for x in result['episodes']),action_acknowledgments=sum(x['action_acknowledgments'] for x in result['episodes']),
        archive_frame_boundaries=sum(x['frames'] for x in result['episodes']),all_episode_checks_pass=all(x['all_checks_pass'] for x in result['episodes']),
        topp_fallback_steps=sum(x['topp_fallback_steps'] for x in result['episodes']),arm_clip_steps=sum(x['arm_clip_steps'] for x in result['episodes']),
        verified_replays=len(result['replays']),all_replay_checks_pass=all(x['matches_expected_frames'] and x['video_hash_matches'] and x['original_observation_hashes_match'] for x in result['replays']),
        sampled_observations=len(sampled_sources))
    (a.out/'evidence_audit.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps(result['totals']),flush=True)
if __name__=='__main__': main()
