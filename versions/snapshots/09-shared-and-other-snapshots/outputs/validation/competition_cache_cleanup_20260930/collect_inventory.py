#!/usr/bin/env python3
"""Bounded read-only storage inventory; writes only requested JSON receipts, never deletes."""
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import json, os, re, stat, subprocess

ROOT = Path('/home/user/Robotic/project2')
OUT = ROOT / 'outputs/validation/competition_cache_cleanup_20260930'
GIB = 2**30

def J(p):
    return json.loads(Path(p).read_text())

def stamp(p):
    p = Path(p); s = p.lstat()
    return {'path':str(p), 'logical_bytes':s.st_size, 'allocated_bytes':s.st_blocks*512,
            'device':s.st_dev, 'inode':s.st_ino, 'nlink':s.st_nlink,
            'mtime_ns':s.st_mtime_ns, 'ctime_ns':s.st_ctime_ns,
            'symlink':p.is_symlink(), 'realpath':str(p.resolve())}

def group(rows):
    a=sum(x['allocated_bytes'] for x in rows); l=sum(x['logical_bytes'] for x in rows)
    return {'files':len(rows),'allocated_bytes':a,'logical_bytes':l,'allocated_gib':round(a/GIB,6),'logical_gib':round(l/GIB,6)}

# GNU du -x -B1 --max-depth=2 and --apparent-size observed in this session.
measurements = {
'data/skill_vtt_task013':(40640512,40602552,'derived_skill_conditions',13),
'data/lowdim_cache_task000':(145354752,144573655,'derived_lowdim_actions_states',0),
'data/lowdim_cache_task001':(354983936,354110759,'derived_lowdim_actions_states',1),
'data/lowdim_cache':(1292939264,1292130422,'derived_lowdim_actions_states',13),
'data/skill_vtt_task000':(3514368,3492715,'derived_skill_conditions',0),
'data/skill_vtt_task001':(10326016,10299763,'derived_skill_conditions',1),
'data/dino_cache_task000':(540777058304,540767128330,'derived_dino_features_and_pool',0),
'data/frame_cache':(73135153152,73121320234,'shared_packed_RGB_JPEG',None),
'data/behavior-1k':(21010964480,21006724051,'original_demo_dataset_and_metadata',None),
'data/behavior-1k/2026-challenge-demos/videos':(19047219200,19046930647,'original_RGB_video',None),
'data/behavior-1k/2026-challenge-demos/data':(1943396352,1943352100,'original_parquet',None),
'data/behavior-assets-download':(33251278848,33251242366,'downloaded_simulation_asset_archives_NOT_frame_cache',None),
'data/omnigibson':(38709579776,38422050258,'installed_simulation_assets_and_task_instances_NOT_cache',None),
'models/dinov3-vitl16-pretrain-lvd1689m':(1212645376,1212602384,'pretrained_vision_encoder',None),
'models/dinov2-with-registers-large':(1217601536,1217562435,'pretrained_vision_encoder',None),
'.conda-envs':(59753766912,58539342948,'tool_environments_DO_NOT_CLEAN',None),
'.python-tools':(193159168,188758575,'tool_environment_DO_NOT_CLEAN',None),
 'third_party':(202248192,195765316,'source_and_environment_dependencies_DO_NOT_CLEAN',None),
}
training_sizes = {
'stage1_skillproto':(7631437824,7631412483),'stage2_contractproto_20260922':(2543525888,2543513787),
'stage2_contractproto_evalenc_20260922':(7630749696,7630715567),'stage3_contract_posonly_20260922':(7630749696,7630715940),
'stage4_contract_delta_20260922':(7630827520,7630793976),'stage5_manipulation_pilot_20260923':(7630614528,7630586232),
'stage6_multiview_manip_20260924':(7630835712,7630804010),'stage7_prophist_manip_20260924':(5087141888,5087114485),
'stage7b_prophistdiff_manip_20260924':(7630786560,7630754276),'stage8_motionw_manip_20260925':(7630786560,7630754097),
 'task0_manip_v1_20260927':(7630831616,7630800789),'task0_manip_v2_20260928':(10174349312,10174313695),
 'task0_v3_30k_20260929_s005000':(6219776000,6219699659),'task0_v3_30k_20260929_s010000':(6219792384,6219713085),
 'task0_v3_30k_20260929_s015000':(6219792384,6219713852),'task0_v3_30k_20260929_s020000':(6219796480,6219717693),
 'task0_v3_30k_20260929_s025000':(6219792384,6219713671),'task0_v3_30k_20260929_s030000':(6219792384,6219712725),
 'task0_v3_30k_20260929_stages':(327680,263166),'task0_v3_30k_20260929_dryplan_stages':(8192,5535),
 'task0_v3_pilot_20260928_a1':(4442398720,4442349234),'task0_v3_pilot_20260928_a2':(4442394624,4442347123),
 'task0_v3_pilot_20260928_a3':(5331382272,5331309691),'task0_v3_tiny_20260928_a1':(12288,5088),
 'task0_v3_tiny_20260928_a2':(5369286656,5369225573),'task1_manip_mv_20260926':(7630893056,7630859175),
 'task1_manip_v2_20260926':(7630987264,7630952625),'task1_manip_v3_20260926':(7630987264,7630951893),
}
for n,(a,l) in training_sizes.items():
    task=0 if n.startswith('task0') else 1 if n.startswith('task1') else 13
    measurements['outputs/training/'+n]=(a,l,'learned_checkpoints_and_training_metadata',task)

directory_inventory=[]
for rel,(a,l,kind,task) in measurements.items():
    directory_inventory.append({'path':str(ROOT/rel),'kind':kind,'behavior_task_id':task,'allocated_bytes':a,'logical_bytes':l,
                                'allocated_gib':round(a/GIB,6),'logical_gib':round(l/GIB,6),'root_stat':stamp(ROOT/rel)})

# One stat-only traversal of candidate caches and training; never follow symlinks or cross devices.
cache_roots=[p for p in (ROOT/'data').iterdir() if 'cache' in p.name or p.name.startswith('skill_vtt')]
scan_roots=cache_roots+[ROOT/'outputs/training']
links=[]; cross_devices=[]; errors=[]; files=[]
for base in scan_roots:
    base_dev=base.lstat().st_dev
    for current,dirs,names in os.walk(base,followlinks=False,onerror=lambda e:errors.append(str(e))):
        for name in list(dirs):
            p=Path(current)/name
            if p.is_symlink():
                links.append({**stamp(p),'escape_project':not p.resolve().is_relative_to(ROOT.resolve())});dirs.remove(name)
            elif p.lstat().st_dev!=base_dev:
                cross_devices.append(str(p));dirs.remove(name)
        for name in names:
            p=Path(current)/name
            if p.is_symlink(): links.append({**stamp(p),'escape_project':not p.resolve().is_relative_to(ROOT.resolve())})
            else: files.append(stamp(p))
root_escape=[str(p) for p in scan_roots if p.is_symlink() or not p.resolve().is_relative_to(ROOT.resolve())]

frames=defaultdict(list); dino=defaultdict(list); checkpoints=[]
for f in files:
    p=Path(f['path'])
    if p.is_relative_to(ROOT/'data/frame_cache'):
        m=re.fullmatch(r'episode_(\d+)\.(.+)',p.name)
        if m:
            ep=int(m[1]);task=0 if ep<200 else 1 if 200<=ep<400 else 13 if 2600<=ep<2800 else 'unknown'
            frames[(task,m[2])].append(f)
    elif p.is_relative_to(ROOT/'data/dino_cache_task000'):
        m=re.fullmatch(r'episode_(\d+)\.(.+)',p.name)
        if m:dino[m[2]].append(f)
    if p.is_relative_to(ROOT/'outputs/training') and p.suffix=='.pt':checkpoints.append(f)

# Inspect only small source metadata; preserve historical hashes without rehashing payloads.
source_rows=[]; missing=[]; incomplete=[]; raw_video_set=set()
for meta in sorted((ROOT/'data/dino_cache_task000').glob('*/episode_*.meta.json')):
    dm=J(meta);ep=int(dm['episode']);view=meta.parent.name
    fm=ROOT/f'data/frame_cache/{view}/episode_{ep}.meta.json'
    if not fm.is_file():missing.append(str(fm));continue
    sm=J(fm); raw=ROOT/'data/behavior-1k/2026-challenge-demos'/sm['video_path'];raw_video_set.add(str(raw))
    deps=[fm.with_name(f'episode_{ep}.jpgpack'),fm.with_name(f'episode_{ep}.idx.npy'),raw]
    missing.extend(str(x) for x in deps if not x.is_file())
    if not dm.get('complete') or not sm.get('complete'):incomplete.append(str(meta))
    source_rows.append({'dino_meta_path':str(meta),'dino_meta':dm,'source_RGB_meta_path':str(fm),'source_RGB_meta':sm})

raw_by_task=[]
for kind,base in [('raw_video',ROOT/'data/behavior-1k/2026-challenge-demos/videos'),('raw_parquet',ROOT/'data/behavior-1k/2026-challenge-demos/data')]:
    for task in [0,1,13]:
        paths=list(base.glob(f'*/chunk-{task:03d}/*.mp4')) if kind=='raw_video' else list(base.glob(f'chunk-{task:03d}/*.parquet'))
        raw_by_task.append({'kind':kind,'behavior_task_id':task,**group([stamp(p) for p in paths]),'paths':[str(p) for p in paths]})

# Anchored executable patterns, excluding this metadata-only Python process.
patterns=[r'^([^[:space:]]*/)?(python[0-9.]*|isaac-sim|kit|omnigibson|torchrun|accelerate)([[:space:]]|$)',
          r'^([^[:space:]]*/)?(bash|sh)[[:space:]]+([^[:space:]]*/)?(run_task[013]_closed_loop|run_wam_v3_unit|run_train_unit|run_s1_[^[:space:]]*)[.]sh([[:space:]]|$)']
process_checks=[];project_processes=[]
for pat in patterns:
    x=subprocess.run(['pgrep','-af',pat],capture_output=True,text=True)
    rows=[]
    for line in x.stdout.splitlines():
        pid=int(line.split()[0])
        if pid==os.getpid():continue
        rows.append(line)
        if '/project2/' in line or any(t in line for t in ['train_wam','policy_server.py','omnigibson.eval','cache_dino','run_task','run_wam','run_train','replay_demo_actions']):project_processes.append(line)
    process_checks.append({'anchored_pattern':pat,'returncode':x.returncode,'matches_excluding_inventory_self':rows})
units=subprocess.run(['systemctl','--user','list-units','--all','--no-pager','--no-legend','project2*','p2-*','wam*','behavior-run-*','task0*','task1*','task13*'],capture_output=True,text=True)
lockrows=[]
proc_locks=Path('/proc/locks').read_text().splitlines()
for p in [Path('/tmp/project2_gpu.lock'),*sorted((ROOT/'outputs/training').glob('*/*.lock'))]:
    if p.exists():
        s=p.stat();key=f'{os.major(s.st_dev):02x}:{os.minor(s.st_dev):02x}:{s.st_ino}'
        lockrows.append({**stamp(p),'holders_from_proc_locks':[l for l in proc_locks if key in l]})
    else:lockrows.append({'path':str(p),'exists':False,'holders_from_proc_locks':[]})

# Primary per-task evidence is partial, never an offline-best success claim.
task0_result=ROOT/'outputs/behavior/closed_loop/task0_v3_30k_20260929_s025000_dev10_k1'
task1_result=ROOT/'outputs/behavior/closed_loop/task1_cl_dev10_prim_v7'
task13_result=ROOT/'outputs/validation/demo_replay_feasibility_20260924/handoff_ep2787_N849_stage8final'
retention_specs=[
('outputs/training/task0_v3_30k_20260929_s025000/checkpoint_step_025000.pt','task0_primary_partial','25k run: partial pick progress per current progress; direct evaluator confirms success=false, Q=0, 3225 steps. Not a statistically demonstrated best.'),
('outputs/training/task1_manip_v2_20260926/checkpoint_final.pt','task1_primary_reference_partial','Exact checkpoint in current Task1 closed-loop/handoff manifests; primitive progress is NOT WAM success; v3 training init depends on this.'),
('outputs/training/stage8_motionw_manip_20260925/checkpoint_final.pt','task13_primary_diagnostic_partial','Train-instance demo-prefix handoff opened lid; not autonomous full-task/dev/official success.'),
('outputs/training/task0_manip_v2_20260928/checkpoint_es_best.pt','required_existing_v3_initialization','Pinned DEFAULT_INIT in scripts/train_wam_v3.py and run_wam_v3_unit.sh; keep until replacement init is approved.'),
('outputs/training/task0_v3_30k_20260929_s030000/checkpoint_last.pt','full_resume_and_candidate_v3_initialization','Last includes optimizer/scheduler/RNG unlike model-only checkpoint files. Do not resume old run; preserve until new-task init decided.'),
('outputs/training/task0_v3_30k_20260929_s030000/checkpoint_best.pt','resume_companion','Existing restore-best logic may require saved best path; keep companion of retained full resume.'),
('outputs/training/task0_v3_30k_20260929_s005000/checkpoint_step_005000.pt','live_symlink_target_hold','stages/checkpoint_best_task.pt points here. best_task.json explicitly says no successful checkpoint; do not treat name as success.'),
('outputs/training/task1_manip_v3_20260926/checkpoint_final.pt','new_task_init_candidate_hold','Retain while garbage-task data/init compatibility unresolved; not validated physical best.'),
('outputs/training/stage3_contract_posonly_20260922/checkpoint_best.pt','legacy_navigation_reference_hold','Matched exec32 navigation partial-result manifest; optional after lead confirms no dependency. Not lid or full task success.')]
retained=[]
for rel,role,reason in retention_specs:
    p=ROOT/rel
    retained.append({**stamp(p),'role':role,'reason':reason,'historical_sha_receipt':str(p)+'.sha256.json' if Path(str(p)+'.sha256.json').exists() else None})
keep_set={x['path'] for x in retained}
checkpoint_candidates=[x for x in checkpoints if x['path'] not in keep_set]

partial_evidence=[{'behavior_task_id':0,'run_path':str(task0_result),'result':J(task0_result/'eval/json/turning_on_radio_311_0.json'),'manifest_checkpoint':J(task0_result/'manifest.json')['checkpoint'],'status':'partial_not_success'},
 {'behavior_task_id':1,'run_path':str(task1_result),'status':'primitive_partial_not_WAM_success','episodes':J(task1_result/'task1_audit.json')['episodes'],'timeline':J(task1_result/'task1_audit.json')['timeline'],'manifest_checkpoint':J(task1_result/'manifest.json')['checkpoint']},
 {'behavior_task_id':13,'run_path':str(task13_result),'status':'train_handoff_lid_success_only_full_task_unproven','verdict':J(task13_result/'analysis/summary.json')['verdict'],'j_trunk':J(task13_result/'analysis/summary.json')['j_trunk'],'manifest_checkpoint':J(task13_result/'manifest.json')['wam_checkpoint']}]

mandatory_paths=[
'data/behavior-1k/2026-challenge-demos','data/omnigibson','data/behavior-assets-download','models',
'.conda-envs','.python-tools','third_party','configs','harness','scripts','docs/progress_archive',
'data/lowdim_cache','data/lowdim_cache_task000','data/lowdim_cache_task001','data/skill_vtt_task000','data/skill_vtt_task001','data/skill_vtt_task013',
'data/dino_cache_task000/cache_config.json','data/dino_cache_task000/sha256_ledger_v1.json','data/dino_cache_task000/sha256_ledger_micro16_v1.json','data/dino_cache_task000/sha256_ledger_micro16_30k_v1.json',
'outputs/training/task0_v3_30k_20260929_stages',
'outputs/validation/task0_takeover_20260930_morning/cache_verify',
'outputs/validation/task1_handoff_20260926/prim_b5_pickplace113_ep389_N2341_oracle',
'outputs/validation/task1_handoff_20260926/handoff_b_pickcan113_ep389_N2341_v2final',
'outputs/behavior/closed_loop/task1_cl_dev10_wam_v8',
'outputs/behavior/closed_loop/s1_dev10_posonly_best5500_exec32_720p_20260922']
mandatory_paths.extend(str(p.relative_to(ROOT)) for p in [task0_result,task1_result,task13_result])
videos=[]
for rel in mandatory_paths:
    p=ROOT/rel
    if p.is_dir() and rel.startswith('outputs/'):
        for v in p.glob('eval/videos/*.mp4'):videos.append(stamp(v))

candidate_payload=dino['feat.npy']+dino['pool.npy']
unsafe=[x['path'] for x in candidate_payload if x['symlink'] or x['nlink']!=1 or not Path(x['realpath']).is_relative_to(ROOT.resolve())]
preconditions={'project_training_or_simulation_processes_none':not project_processes,
 'related_user_systemd_units_none':units.returncode==0 and not units.stdout.strip(),
 'project_lock_holders_none':not any(x['holders_from_proc_locks'] for x in lockrows),
 'all_600_DINO_and_RGB_metadata_complete':len(source_rows)==600 and not incomplete,
 'RGB_packs_indices_and_raw_videos_exist':not missing,
 'candidate_paths_no_symlinks_or_hardlinks':not unsafe,
 'scan_no_escape_or_mounts_or_errors':not root_escape and not cross_devices and not errors and not any(x['escape_project'] for x in links)}

inventory={'schema':'competition-cache-inventory-v1','recorded_at':datetime.now().astimezone().isoformat(),
 'project_root':str(ROOT),'mode':'read_only_inventory_no_deletions','goal_source':str(ROOT/'progress.md'),
 'prior_competition_pivot_directory_present':(ROOT/'outputs/validation/competition_pivot_20260930').exists(),
 'method':{'directory_sizes':'GNU du -x -B1 and --apparent-size; directory rows overlap and must not be summed blindly',
 'payload_sizes':'lstat only; no model load, no NPY/MP4/JPG/parquet data read, no hashing',
 'historical_hashes':'Copied from existing JSON metadata only, NOT freshly verified',
 'bounded_source_checks':'600 DINO and 600 RGB small JSONs plus retained result/config metadata; no full tests or GPU'},
 'totals':{'data_allocated_bytes':708731797504,'data_logical_bytes':708413679201,'training_allocated_bytes':166282924032,'training_logical_bytes':166281633571},
 'directories':directory_inventory,
 'RGB_cache_by_task_and_suffix':[{'behavior_task_id':t,'suffix':s,**group(v)} for (t,s),v in sorted(frames.items(),key=lambda x:str(x[0]))],
 'DINO_by_suffix':{s:group(v) for s,v in dino.items()},'raw_data_by_task':raw_by_task,
 'DINO_config':J(ROOT/'data/dino_cache_task000/cache_config.json'),
 'DINO_source_binding':{'metadata_pairs':len(source_rows),'missing_sources':missing,'incomplete_metadata':incomplete,'raw_videos':[stamp(p) for p in sorted(raw_video_set)],'source_metadata_snapshot':source_rows},
 'checkpoints':checkpoints,'partial_physical_evidence':partial_evidence,'retained_video_files':videos,
 'active_state':{'project_training_or_simulation_pids':project_processes or 'none','process_checks':process_checks,'user_units':{'returncode':units.returncode,'stdout':units.stdout,'stderr':units.stderr},'locks':lockrows},
 'path_safety':{'scan_roots':[str(p) for p in scan_roots],'symlinks':links,'root_escape':root_escape,'cross_device_entries':cross_devices,'errors':errors,'scope':'Full metadata-only walk of candidate cache roots and training only; environments/installed assets were measured with du but not audited as deletion candidates.'},
 'unknowns':['Competition task number is not BEHAVIOR task id; target dataset and new initialization have not been finalized.','No payload integrity rehash or reconstruction run performed.','No claim of a global best checkpoint across all old runs; selected representatives are evidence-bound partial/diagnostic outcomes.']}

proposal={'schema':'competition-cache-proposed-actions-v1','recorded_at':inventory['recorded_at'],'inventory_path':str(OUT/'inventory.json'),
 'execution_authority':'Proposal only. No deletions performed by this subagent; parent session executes after refreshing progress and checking dependencies.',
 'safe_first_batch':{'name':'Task0 regenerable DINO activations and pooled features only','status':'ready_for_parent_dependency_confirmation' if all(preconditions.values()) else 'blocked_by_precondition',
  'preconditions_observed':preconditions,'exact_files':candidate_payload,'estimated_release':group(candidate_payload),
  'preserve_in_place':['cache_config.json and all sha256 ledgers','all episode_*.meta.json and episode_*.frames.npy','all source RGB packs/indices/meta, raw videos, lowdim and skill annotations, frozen encoder and configs'],
  'rebuild_sources':{'immediate_RGB_root':str(ROOT/'data/frame_cache'),'episode_ids':[0,199],'views':inventory['DINO_config']['views'],'native_RGB_paths':[str(p) for p in sorted(raw_video_set)],'encoder':str(ROOT/'models/dinov3-vitl16-pretrain-lvd1689m'),'encoder_revision':inventory['DINO_config']['encoder_revision'],'config':str(ROOT/'configs/lilawam_task000_multiview_manip_v2_train.yaml'),'script':str(ROOT/'scripts/cache_dino_features.py'),'selection':'200 Task0 episodes, even stride2 frames in non-move-to segments; 434997 images, layers -12/-8/-4, 201 tokens, 1024 channels, bf16 uint16; pool float32 CLS+patch mean'},
  'why_regenerable':'Frozen eval encoder outputs from preserved deterministic RGB frame packs, not original images or learned weights. No unique learned state resides in these NPY payloads.',
  'important_dependency_limit':'Existing Task0 V3 trainer/configs require BOTH feat.npy and pool.npy, and V3 stats/source ledgers bind them. Old Task0 training must remain paused. New garbage-task training must bind its own matching cache before launch; do not reuse the Task0 cache path by accident.',
  'rebuild_trap':'cache_dino_features.py skips any episode with complete=true in meta even if feat/pool files were deleted. Rebuild into a fresh explicit --out directory, or archive/remove the matching completion markers deliberately before rebuilding the old path. Rebinding/regenerating cache-ledger stat receipts is then required; copying old stat receipts is invalid.',
  'before_parent_deletes':['Update progress before action.','Recheck exact files against saved device/inode/size/mtime/ctime; do not follow symlinks, expand wildcards, or delete parent directories.','Recheck anchored active processes and lock holders immediately; prevent new jobs from starting.','Record payload removal and its impact on paused Task0 configs; retain all provenance and this inventory.']},
 'deferred_cache_actions':[{'paths':[str(ROOT/'data/frame_cache'),str(ROOT/'data/lowdim_cache_task001'),str(ROOT/'data/skill_vtt_task001')],'decision':'KEEP_NOW','reason':'frame_cache is shared by Task0 ep0-199, Task1 ep200-399, Task13 ep2600-2799. Task1 garbage demos may be reused for competition transfer; task-id equivalence not established. Never delete the whole shared frame_cache.'},
 {'paths':[str(ROOT/'data/lowdim_cache'),str(ROOT/'data/lowdim_cache_task000'),str(ROOT/'data/skill_vtt_task013'),str(ROOT/'data/skill_vtt_task000')],'decision':'KEEP_NOW','reason':'Small relative to DINO; needed source actions, phase labels/contract conditions and retained diagnostic reproducibility. Defer until new-task init/dependency selection.'}],
 'checkpoint_retention':retained,
 'checkpoint_pruning_candidates':{'decision':'DEFER_UNTIL_INIT_AND_RETENTION_SET_CONFIRMED','files':checkpoint_candidates,'estimated_release_if_confirmed':group(checkpoint_candidates),'reason':'These are learned checkpoints, NOT regenerable caches or proven duplicates. No deletion until parent confirms which init/resume/representative results survive. Full training dirs must not be removed: preserve all config, manifest, summary, logs, validation plans/scores, SHA sidecars. Newly chosen initialization can promote any candidate to KEEP.'},
 'must_keep_paths':[{'path':str(ROOT/p),'exists':(ROOT/p).exists()} for p in mandatory_paths],
 'must_keep_file_classes':['All non-checkpoint training metadata, configuration, manifests, split/normalization statistics, scoring results and historical SHA sidecars.','All DINO and RGB metadata and index arrays while payload cleanup is performed.','Retained result videos, viewer sidecars, score JSONs and short failure evidence; do not delete failed outcomes to manufacture success.','Task13 handoff success video and analysis are a train-instance diagnostic, not a dev/full-task success.','No model/environment/asset-archive cleanup under this proposal.'],
 'physical_best_boundary':{'task0':'No successful checkpoint. best_task symlink points at 5k despite all Q0; 25k is a partial representative only.','task1':'No proven autonomous WAM full-task success in inspected evidence. Primitive/oracle handoffs are separately labeled.','task13':'Stage8 final has actual post-demo-prefix physical lid opening (max 1.7887rad), but no independent autonomous full-task success claim.'},
 'new_competition_initialization_selected':False,'gpu_started':False,'files_deleted':[]}

OUT.mkdir(parents=True,exist_ok=True)
for name,data in [('inventory.json',inventory),('proposed_actions.json',proposal)]:
    path=OUT/name
    if path.exists():raise RuntimeError(f'Refuse to overwrite existing report: {path}')
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'inventory':str(OUT/'inventory.json'),'proposed_actions':str(OUT/'proposed_actions.json'),'safe_first_batch':proposal['safe_first_batch']['estimated_release'],'preconditions':preconditions,'frame_task_sizes':inventory['RGB_cache_by_task_and_suffix'],'checkpoint_pruning_deferred':proposal['checkpoint_pruning_candidates']['estimated_release_if_confirmed'],'links':links,'active_project_pids':inventory['active_state']['project_training_or_simulation_pids']},ensure_ascii=False))
