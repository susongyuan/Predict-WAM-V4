#!/usr/bin/env python3
"""Independent stdlib-only evidence audit. Never imports project code or torch."""
from pathlib import Path
import collections, datetime, hashlib, json, math, pickletools, struct, subprocess, zipfile
ROOT = Path('/home/user/Robotic/project2')
OUT = ROOT/'outputs/validation/task090_takeover_20261002_1319/final50k_audit'
TRAIN = ROOT/'outputs/training/task090_v3_50k_20260930_s050000'
BATCH = ROOT/'outputs/validation/task090_remaining_stages_20261002/triplet_050000'
NOW = lambda: datetime.datetime.now().astimezone().isoformat()
def read(p): return json.loads(Path(p).read_text())
def save(name, data):
    with (OUT/name).open('x') as f: json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
def sha(p):
    p=Path(p); a=p.stat(); h=hashlib.sha256()
    with p.open('rb') as f:
        while b:=f.read(8*1024*1024): h.update(b)
    z=p.stat()
    return {'path':str(p),'sha256':h.hexdigest(),'size':a.st_size,'mtime_ns':a.st_mtime_ns,'stable':(a.st_size,a.st_mtime_ns,a.st_ino)==(z.st_size,z.st_mtime_ns,z.st_ino)}
def lines(p): return [(i,json.loads(s)) for i,s in enumerate(Path(p).read_text().splitlines(),1) if s.strip()]
def symbolic(data):
    # Interpret only data-stack operations. GLOBAL/REDUCE/BUILD are inert records:
    # no imports, callables, unpickling hooks, tensors, or tensor payload loads.
    stack=[]; memo={}; mark=object()
    def marked():
        i=len(stack)-1
        while stack[i] is not mark: i-=1
        vals=stack[i+1:]; del stack[i:]; return vals
    for op,arg,pos in pickletools.genops(data):
        n=op.name
        if n in ('PROTO','FRAME'): continue
        if n=='STOP': return stack[-1]
        if n=='MARK': stack.append(mark)
        elif n in ('BINUNICODE','SHORT_BINUNICODE','UNICODE','BININT','BININT1','BININT2','LONG1','LONG4','BINFLOAT','SHORT_BINBYTES','BINBYTES'): stack.append(arg)
        elif n=='NONE': stack.append(None)
        elif n=='NEWTRUE': stack.append(True)
        elif n=='NEWFALSE': stack.append(False)
        elif n=='EMPTY_DICT': stack.append({})
        elif n=='EMPTY_LIST': stack.append([])
        elif n=='EMPTY_TUPLE': stack.append(())
        elif n in ('BINPUT','LONG_BINPUT'): memo[arg]=stack[-1]
        elif n in ('BINGET','LONG_BINGET'): stack.append(memo[arg])
        elif n=='MEMOIZE': memo[len(memo)]=stack[-1]
        elif n=='TUPLE': stack.append(tuple(marked()))
        elif n in ('TUPLE1','TUPLE2','TUPLE3'):
            count=int(n[-1]); vals=tuple(stack[-count:]); del stack[-count:]; stack.append(vals)
        elif n=='SETITEMS':
            vals=marked(); stack[-1].update(zip(vals[::2],vals[1::2]))
        elif n=='SETITEM':
            val=stack.pop(); key=stack.pop(); stack[-1][key]=val
        elif n=='APPENDS':
            vals=marked(); stack[-1].extend(vals)
        elif n=='APPEND':
            val=stack.pop(); stack[-1].append(val)
        elif n=='GLOBAL': stack.append({'global':arg})
        elif n=='REDUCE':
            args=stack.pop(); fn=stack.pop()
            stack.append({} if fn=={'global':'collections OrderedDict'} else {'reduce':fn,'args':args})
        elif n=='BINPERSID': stack.append({'persistent':stack.pop()})
        elif n=='BUILD':
            state=stack.pop(); stack[-1]['inert_build_state']=state
        else: raise ValueError(f'Unsupported inert opcode {n} at {pos}')
    raise ValueError('Missing pickle STOP')
started=NOW(); result={'started_at':started,'scope':'final 50k training and only this batch seed0/1/2; training chain metadata only, no historical evaluation re-audit','prohibitions_observed':['no simulation launched or awaited','no torch import/load','no GPU access','no shared source/progress/config changes','no threshold changes or dev feedback'],'limitations':['No full feature payload hashing (673322391552 bytes); historical checkpoint SHA values are ancestry sidecar comparisons, not recomputed payload hashes.','Exit code 1 root cause not established: process logs end at shutdown without traceback.','This is assisted/prepositioned development manipulation, not pure-WAM autonomy, full-task success, or an official score.']}
hashes=read(OUT/'checkpoint_hashes.json'); result['checkpoint_hashes_file']=str(OUT/'checkpoint_hashes.json')
meta=[]
for h in hashes['checkpoints']:
    p=Path(h['path'])
    with zipfile.ZipFile(p) as z:
        pn=next(n for n in z.namelist() if n.endswith('/data.pkl')); raw=z.read(pn); ck=symbolic(raw)
        row={'path':str(p),'data_pickle_sha256':hashlib.sha256(raw).hexdigest(),'method':'nonexecuting pickletools stack parser; tensor data not loaded', **{k:ck.get(k) for k in ('step','model_only','optimizer_state_saved','resumable','resume_format','manifest_sha256','training_contract_sha256','resume_source','resume_source_sha256')}}
        row['manifest_hash_matches']=ck.get('manifest_sha256')==sha(TRAIN/'manifest.json')['sha256']
        state=ck.get('resume_state')
        if state:
            opt=state['optimizer_state_dict']; counts=[]
            for v in opt['state'].values():
                tensor=v['step']; storage=tensor['args'][0]['persistent']; offset=tensor['args'][1]
                assert storage[1]['global']=='torch FloatStorage' and tensor['args'][2]==()
                scalar=z.read(pn.rsplit('/',1)[0]+'/data/'+storage[2]); counts.append(struct.unpack_from('<f',scalar,offset*4)[0])
            row['resume_state']={k:state.get(k) for k in ('step','safe_optimizer_boundary','device_type','cursor','best','discarded_microbatches_replayed')}
            row['resume_state'].update(scheduler=state['scheduler_state_dict'],rng_keys=list(state['rng']),optimizer_parameters=len(opt['state']),optimizer_step_counts=dict(collections.Counter(counts)),optimizer_param_groups=opt['param_groups'])
        meta.append(row)
result['checkpoint_metadata']=meta
chain=[]
for mp in sorted((ROOT/'outputs/training').glob('task090_v3_50k_20260930_s*/manifest.json')):
    m=read(mp); summary=read(mp.parent/'summary.json'); init=read(mp.parent/'initialization.json')
    row={'manifest':sha(mp),'start':m['steps_start_at'],'target':m['requested_updates'],'schedule_steps':m['schedule_steps'],'fresh_optimizer':m['fresh_optimizer'],'completed_updates':summary['completed_updates'],'initialization':init,'contract_sha256':m['training_contract_sha256'],'resume_source':m.get('resume_source'),'resume_source_sha256':m.get('resume_source_sha256')}
    if m.get('resume_source'):
        side=read(m['resume_source']+'.sha256.json'); row['parent_sidecar']=side
        row['parent_sha_matches']=side['sha256']==m['resume_source_sha256']; row['parent_step_matches']=side['step']==row['start']; row['parent_has_training_state']=side['model_only'] is False
    chain.append(row)
result['training_chain']=chain
m=read(TRAIN/'manifest.json'); logs=lines(TRAIN/'log.jsonl'); trainrows=[x for _,x in logs if x.get('event')=='train']
result['training_final']={'status':read(TRAIN/'status.json'),'initialization':read(TRAIN/'initialization.json'),'summary':{k:v for k,v in read(TRAIN/'summary.json').items() if not isinstance(v,(dict,list))},'logged_training_events':len(trainrows),'logged_steps':[x['step'] for x in trainrows],'monotonic_logged_steps':all(a['step']<b['step'] for a,b in zip(trainrows,trainrows[1:])),'train_episodes':len(m['train_episodes']),'validation_episodes':len(m['validation_episodes']),'splits_disjoint':not(set(m['train_episodes'])&set(m['validation_episodes'])),'cache_assurance':m['cache_assurance'],'train_launch':read(ROOT/'outputs/validation/task090_remaining_stages_20261002/train_050000_launch.json')}
# Recompute source ledger hashes, including all 2233 training manifest entries.
cache={h['path']:h for h in hashes['checkpoints']}; ledger=[]
def check_sources(label, sources):
    for path,expected in sources.items():
        if path not in cache:
            try: cache[path]=sha(path)
            except OSError as e: cache[path]={'path':path,'error':str(e)}
        actual=cache[path]; ledger.append({'scope':label,'expected':expected,'actual':actual,'matches':expected==actual.get('sha256')})
check_sources('training50k',m['source_sha256'])
runs=[]; snapshots=[]
for run in sorted((ROOT/'outputs/behavior/closed_loop').glob('task090_wam_s050000_dev10_seed*')):
    rm=read(run/'manifest.json'); seed=rm['seed']; assert seed in (0,1,2)
    if not rm.get('finished_utc'):
        runs.append({'seed':seed,'status':'unfinished_not_scored','run':str(run)}); continue
    check_sources(f'seed{seed}',rm['artifact_sha256'])
    scorep=run/'eval/json/composting_waste_311_0.json'; score=read(scorep)
    tracep=next(run.glob('policy_trace*.jsonl')); tr=lines(tracep); tr=[(n,x) for n,x in tr if x.get('episode')==score['policy_episode']]
    posep=run/'eval/object_poses/composting_waste_311_0.jsonl'; ps=lines(posep); header=ps[0][1]; poses=[(n,x) for n,x in ps if type(x.get('step')) is int]; bystep={x['step']:(n,x) for n,x in poses}
    identity_match=all(header['episode_identity'][k]==score[k] for k in ('task','instance_id','rollout_id','policy_episode'))
    startn,start=next((n,x) for n,x in tr if x['event']=='orch_contract_start' and x['key']=='pick up from(trash_can,floors)')
    endn,end=next((n,x) for n,x in tr if x['event']=='orch_contract_end' and x['index']==start['index'])
    selected=[(n,x) for n,x in poses if start['step']<x['step']<=end['step']]
    target=next(k for k,v in header['entities'].items() if v['category']=='trash_can'); baseline=header['entities'][target]['aabb_at_start_world'][0][2]
    streak=0; longest=0; first=None; qualified=[]; hand_count=0; maxlift=None; actions=[]; action_errors=[]
    for n,x in selected:
        hand=x['manipulation']['assisted_grasp']; held=False
        for v in hand.values():
            if v.get('object_instance')==target and v.get('constraint_active') is True and v.get('release_pending') is False:
                hand_count+=1; lift=v['aabb_min_z']-baseline; maxlift=lift if maxlift is None else max(maxlift,lift)
                held |= lift>=0.03 and math.isfinite(lift)
        streak=streak+1 if held else 0; longest=max(longest,streak)
        if held: qualified.append(x['step'])
        if streak==15 and first is None: first=x['step']-14
        a=x['onboard_trace']; actions.append({'line':n,'observation_step':x['step'],'action_step':a['action_step'],'action':a['action']})
        if a['action_step']!=x['step']-1 or len(a['action'])!=23 or not all(math.isfinite(v) for v in a['action']): action_errors.append(x['step'])
    chunks=[(n,x) for n,x in tr if x['event']=='chunk' and start['step']<=x['step']<end['step']]
    matches=[]
    for n,x in chunks:
        pose=bystep[x['step']+1][1]; a=pose['onboard_trace']['action']; match=([round(v,4) for v in a[:3]]==x['base_cmd'] and [round(a[i],3) for i in (14,22)]==x['grip'])
        matches.append({'trace_line':n,'action_step':x['step'],'base_and_gripper_match_logged_rounding':match,'checkpoint_step':x['model_checkpoint_step']})
    measurements=[(n,x) for n,x in tr if x['event']=='onboard_measurement']
    expected_steps=list(range(start['step']+1,end['step']+1))
    goal0=poses[0][1]['goal_literals']; goalend=poses[-1][1]['goal_literals']; denominator=sum(not x for x in goal0[0]); numerator=sum((not a) and b for a,b in zip(goal0[0],goalend[0])); q=numerator/denominator if denominator else 0.0
    row={'seed':seed,'run':str(run),'ended':True,'started_utc':rm['started_utc'],'finished_utc':rm['finished_utc'],'evaluator_returncode':rm['evaluator_returncode'],'launcher_returncode':rm['launcher_returncode'],'evaluation_complete':rm['evaluation_complete'],'score':score,'q_score_independent':{'reset':goal0,'final':goalend,'numerator':numerator,'denominator':denominator,'q':q,'matches':q==score['q_score']['final']},'pose_end':ps[-1][1],'all_pose_steps_contiguous':list(bystep)==list(range(score['steps']+1)),'episode_identity_matches':identity_match,'official_score_valid':rm['official_score_valid'],'artifact_changes_reported':rm['artifact_changes'],'checkpoint_sha_matches':rm['checkpoint_sha256']==hashes['checkpoints'][0]['sha256'],'first_grasp':{'rule_source':str(ROOT/'scripts/summarize_task090_checkpoints.py')+':10-95','lift_threshold_m':0.03,'required_consecutive_steps':15,'success':first is not None,'first_qualified_observation':first,'longest_held_lift_steps':longest,'qualified_steps':qualified,'active_target_hand_observations':hand_count,'maximum_active_target_lift_m':maxlift,'baseline_bottom_z_m':baseline,'first_action_step':start['step'],'last_action_step':end['step']-1,'observations_contiguous':[x['step'] for _,x in selected]==expected_steps,'valid_observations':len(selected)},'wam':{'chunks':len(chunks),'actions':len(actions),'action_errors':action_errors,'first_chunk':{'line':chunks[0][0],'value':chunks[0][1]},'first_emitted_action':actions[0],'chunk_start_matches':matches,'contract_end':{'line':endn,'value':end},'all_checkpoint_steps_50000':all(x['model_checkpoint_step']==50000 for _,x in chunks)},'onboard':{'records':len(measurements),'contiguous_steps':[x['step'] for _,x in measurements]==expected_steps,'verified_true':sum(x['measurement'].get('verified') is True for _,x in measurements),'admitted_true':sum(x['measurement'].get('admitted') is True for _,x in measurements),'diagnostic_only_all':all(x.get('diagnostic_only') is True for _,x in measurements),'rigid_status_counts':dict(collections.Counter(x['measurement'].get('rigid_tracking',{}).get('status') for _,x in measurements))}}
    # Preserve the actually submitted 23D actions, not only policy proposal summaries.
    save(f'seed{seed}_emitted_actions.json',{'source':str(posep),'semantics':header['onboard_trace'],'actions':actions})
    evidence=[run/'manifest.json',scorep,tracep,posep,run/'policy_server.log',run/'process.log',run/'resource_summary.json',BATCH/f'seed{seed}_launch.json',BATCH/f'seed{seed}_unit.log']
    row['evidence_fingerprints']=[sha(p) for p in evidence]
    row['image_evidence']=[sha(p) for p in sorted((run/'observations').rglob('*')) if p.is_file()]
    text=(run/'process.log').read_text().splitlines(); row['process_end_excerpt']=text[-9:];row['traceback_present']=any('Traceback (most recent call last)' in x for x in text)
    snapshots.append({'seed':seed,'manifest':rm,'score':score,'pose_header':header,'pose_end':ps[-1][1],'policy_server_log':(run/'policy_server.log').read_text(),'resource_summary':read(run/'resource_summary.json')})
    runs.append(row)
result['runs']=runs; result['ended_seeds']=[x['seed'] for x in runs if x.get('ended')];result['unfinished_seeds']=[x['seed'] for x in runs if not x.get('ended')]
result['source_checks']={'total_comparisons':len(ledger),'unique_files':len(cache),'mismatches':[x for x in ledger if not x['matches']],'details_file':str(OUT/'source_fingerprints.json')}
result['source_maps_equal_between_seeds']=all(snapshots[0]['manifest']['artifact_sha256']==s['manifest']['artifact_sha256'] for s in snapshots)
result['batch_state_snapshot']=read(BATCH/'state.json')
result['finished_at']=NOW()
save('source_fingerprints.json',{'method':'serial 8MiB SHA256; all declared source ledger entries checked against current files; this is not authentication of historical execution','checks':ledger})
save('raw_snapshots.json',snapshots)
save('audit.json',result)
print(json.dumps({'saved':str(OUT/'audit.json'),'finished_at':result['finished_at'],'ended':result['ended_seeds'],'unfinished':result['unfinished_seeds'],'source_mismatches':len(result['source_checks']['mismatches']),'runs':[{k:x[k] for k in ('seed','first_grasp','evaluator_returncode','onboard')} for x in runs]},ensure_ascii=False,indent=2))
