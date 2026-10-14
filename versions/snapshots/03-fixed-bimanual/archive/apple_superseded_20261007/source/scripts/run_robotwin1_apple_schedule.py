#!/usr/bin/env python3
"""One-shot continuation: collection -> native interface -> cache/train30k -> paired5+5.
The existing collection unit owns the GPU until it exits; this supervisor holds no
GPU lock while waiting. No retries/resume/source substitutions. Every phase is logged.
"""
from pathlib import Path
import argparse,datetime,hashlib,json,os,signal,subprocess,sys,time,traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from harness.robotwin_policy_worker import atomic_json
from scripts.run_robotwin_checkpoint_queue import unit_state
from scripts.gpu_lock import owner
TASK='pick_apple_messy';COLLECT=ROOT/'outputs/runs/pick_apple_messy/collection_100_20261007'
DATA=ROOT/'data/benchmarks/robotwin1/pick_apple_messy_100_rgb_v1'
CONFIG=ROOT/'configs/fixed_bimanual/robotwin1_apple_100_wam.yaml'
TRAIN=ROOT/'outputs/runs/pick_apple_messy/wam_30k_20261007'
EVAL=ROOT/'outputs/runs/pick_apple_messy/pure_wam_25k_30k_5each_20261007'
TRAIN_UNIT='p2-apple-wam30k-20261007';COLLECT_UNIT='p2-apple-collect100-20261007'
FIRST=ROOT/'outputs/verification/robotwin_fruit_20261007/native_firstframe_910000'
SEEDS=[910001,910002,910003,910004,910005]
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def read(path):
    try:return json.loads(Path(path).read_text())
    except (OSError,ValueError):return {}
def finalize_data(collection):
    import yaml
    from harness.robotwin_dataset import make_split,RoboTwinEpisode
    assert collection['status']=='complete' and len(collection['accepted'])==100
    DATA.mkdir(parents=True,exist_ok=False);(DATA/'data').mkdir()
    episodes=[]
    for index,row in enumerate(collection['accepted']):
        src=Path(row['hdf5']);assert row['episode_id']==index and sha(src)==row['sha256']
        assert row['first_frame_visibility']['head_camera']['target_pixels']>=32
        ep=RoboTwinEpisode(src);assert len(ep)==row['frames'] and ep.state.shape[1]==14
        dest=DATA/'data'/f'episode_{index:07d}.hdf5';os.link(src,dest)
        episodes.append({'source':str(src),'source_sha256':row['sha256'],'path':str(dest),'sha256':row['sha256'],'frames':row['frames'],'source_episode_id':row['seed']})
    # All attempted collection seeds are excluded from physical evaluation, not just accepted seeds.
    (COLLECT/'seed.txt').write_text('\n'.join(str(r['seed']) for r in collection['attempts'])+'\n')
    source={'status':'complete','task':TASK,'backend':'robotwin1','repo':'RoboTwin-Platform/RoboTwin','revision':'65a6f5bb74794ba193c6bf17d8eae3b74fc2df70',
      'dataset_origin':'locally_collected_official_expert_not_downloaded_RGB_demos','official_pointcloud_archive_separate':'data/benchmarks/robotwin_fruit/pick_apple_messy_early_raw/SOURCE.json',
      'collection_status_sha256':sha(COLLECT/'status.json'),'asset_download_receipt':read(ROOT/'third_party/RoboTwin1/DOWNLOAD_PROVENANCE.json'),
      'compatibility_patches':read(ROOT/'third_party/RoboTwin1/LOCAL_PATCHES.json'),'expert_planner_adaptation':'Official INSTALLATION section3.2 collision-flag removal confined to local demonstration planner; never imported by pure-WAM evaluator.',
      'nominal_frequency':250/15,'measured_clock_available':False,'state_semantics':'controller_joint_targets_and_gripper_command_not_measured_qpos','gripper_units':'physical_meters_0_to_0.045',
      'alignment':'state=command[t], action=command[t+1], RGB=observation[t]; omit last observation','images':'native RGB rendered serially, one JPEG quality95 encoding; no pointcloud-rendered RGB','task_selection_basis':'User asked to continue from the grasp-apple candidate; explicit original nickname mapping not independently confirmed.',
      'episodes':episodes,'first_frame_gate':'Every accepted demonstration has >=32 native apple segmentation pixels in head-camera frame0; one independent first RGB manually inspected by Codex.',
      'heldout_policy':'Fixed80/10/10 split; heldout excluded from feature cache/statistics/training/checkpoint choice.'}
    atomic_json(DATA/'SOURCE.json',source)
    split=make_split(DATA/'data',seed=20261004,counts=(80,10,10));atomic_json(DATA/'split_manifest.json',split)
    cfg=yaml.safe_load((ROOT/'configs/fixed_bimanual/robotwin_cabinet_500_wam.yaml').read_text())
    cfg['dataset'].update(task_name=TASK,backend='robotwin1',task_config='pick_apple_messy',split_manifest=str((DATA/'split_manifest.json').relative_to(ROOT)),split_sha256=sha(DATA/'split_manifest.json'),feature_cache='data/robotwin1_apple_features',nominal_frequency=250/15,gripper_units='physical_meters_0_to_0.045')
    with CONFIG.open('x') as f:yaml.safe_dump(cfg,f,sort_keys=False)
    return cfg

def write_report(out,state):
    q=read(EVAL/'queue_status.json');counts=q.get('counts',{})
    lines=['# 抓苹果 WAM 实验自动状态', '',f"更新时间：{now()}",f"状态：{state['status']}；阶段：{state.get('stage')}",
      '任务：RoboTwin1 pick_apple_messy；RGB演示为本地运行官方专家采集，不是已下载的官方RGB包。',
      '纯WAM，原生14D物理夹爪单位；无VTT/VLM/恢复，GT仅评分。','', '| 权重 | 有效测试 | 原生成功 | 原生失败 | 无效初始化 | 中断/未完成 |', '|---|---:|---:|---:|---:|---:|']
    for step in ('25000','30000'):
        c=counts.get(step,{})
        lines.append(f"| {step} | {c.get('scored_episodes',0)}/5 | {c.get('native_successes','未测试')} | {c.get('native_failures','未测试')} | {c.get('invalid_initializations',0)} | {c.get('incomplete_attempts',0)} |")
    if state.get('error'):lines+=['',state['error']]
    lines+=['','未凑齐5条有效结果时成功率不可计算；初始化无效/中断不当策略失败。',f'训练目录：`{TRAIN.relative_to(ROOT)}/train`',f'测试证据：`{EVAL.relative_to(ROOT)}`']
    text='\n'.join(lines)+'\n';(out/'report.md').write_text(text)
    # User asked to keep results in the previously consolidated experiment document.
    import fcntl
    with (ROOT/'docs/archive/legacy_mobile_experiments_20261004.md').open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX);f.write('\n\n## 2026-10-07 抓苹果任务自动实验回执\n\n'+text);f.flush();fcntl.flock(f,fcntl.LOCK_UN)

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--execute',action='store_true');a=p.parse_args();out=Path(a.out).resolve()
    if not a.execute:
        print(json.dumps({'collection':str(COLLECT),'data':str(DATA),'config':str(CONFIG),'training':str(TRAIN),'eval':str(EVAL),'train_steps':30000,'test_seeds':SEEDS,'gpu_overlap':False},indent=2));return 0
    out.mkdir(parents=True,exist_ok=False)
    state={'status':'running','stage':'waiting_for_collection','task':TASK,'started_at':now(),'train_steps':30000,'test_seeds':SEEDS,'training_unit':TRAIN_UNIT,'collection_unit':COLLECT_UNIT}
    def stage(name,**extra):state.update(stage=name,**extra);atomic_json(out/'status.json',state)
    def run(name,command,timeout=None):
        stage(name,command=command)
        with (out/(name+'.log')).open('x') as f:
            child=subprocess.Popen(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            start=time.monotonic()
            try:
                while child.poll() is None:
                    if timeout and time.monotonic()-start>timeout:raise TimeoutError(f'{name} exceeded {timeout}s')
                    time.sleep(.5)
            except BaseException:
                if child.poll() is None:
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=90)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                raise
        if child.returncode:raise RuntimeError(f'{name} failed rc={child.returncode}; see {name}.log')
    def stopped(sig,frame):raise InterruptedError(f'supervisor received signal {sig}; no automatic resubmission')
    for s in (signal.SIGTERM,signal.SIGINT):signal.signal(s,stopped)
    code=1
    try:
        deadline=time.monotonic()+28800
        while True:
            collection=read(COLLECT/'status.json');unit=unit_state(COLLECT_UNIT)
            stage('waiting_for_collection',accepted=len(collection.get('accepted',[])),collection_status=collection.get('status'))
            if collection.get('status') in ('failed','stopped') or unit.get('ActiveState')=='failed':raise RuntimeError('collection failed: '+str(collection.get('error',unit)))
            if (collection.get('status')=='complete' and unit.get('SubState')=='exited' and unit.get('MainPID')=='0' and unit.get('Result')=='success' and owner() is None):break
            if time.monotonic()>deadline:raise TimeoutError('collection/GPU release wait expired')
            time.sleep(5)
        stage('sealing_dataset');cfg=finalize_data(collection)
        native=out/'native_interface';lock='p2-apple-interface-20261007'
        command=['/usr/bin/python3',str(ROOT/'scripts/gpu_lock.py'),'run','--owner',lock,'--max-wait','120','--','timeout','-k','20','180',str(ROOT/'.venvs/robotwin-sim/bin/python'),'-u','-B',str(ROOT/'scripts/eval_robotwin1_wam.py'),'--native-smoke-only','--run-dir',str(native),'--seeds','910000','--gpu-lock-owner',lock]
        run('native_interface',command,timeout=340)
        check=read(native/'result.json');assert check.get('status')=='native_interface_smoke_complete',check
        first=read(FIRST/'result.json');assert first.get('first_frame_target_visible') is True and first.get('native_initial_success') is False
        admission={'schema_version':1,'task':TASK,'source_split_sha256':cfg['dataset']['split_sha256'],'rgb_real_image_visible':True,'native_preflight_passed':True,'interface_accepted':True,'operator_confirmed':True,'operator':'Codex assistant: direct RGB visual inspection plus native instance/segmentation and qpos smoke evidence','confirmed_at':now(),'note':'Head RGB native seed910000 was visually inspected and contains a complete red apple; all100 expert first frames meet native head target-pixel gate. RGB/native14 controller interface passed one hold action, not policy success. Operator is assistant, not a claimed human confirmation.','evidence':{'first_frame_rgb':{'path':str(FIRST/'head_camera.png'),'sha256':sha(FIRST/'head_camera.png')},'native_preflight':{'path':str(native/'result.json'),'sha256':sha(native/'result.json')}}}
        atomic_json(out/'admission.json',admission)
        audit={'complete':True,'selection_seeds':[910000],'note':'New native1 branch only preflight/visual/hold seed910000. All expert source attempts excluded independently via seed.txt. Neither test candidate nor heldout rollout has been used for selection.'};atomic_json(out/'selection_seeds.json',audit)
        traincmd=[str(ROOT/'.venvs/wam-fixed/bin/python'),'-B',str(ROOT/'scripts/run_robotwin_training_pipeline.py'),'--config',str(CONFIG),'--admission-json',str(out/'admission.json'),'--out',str(TRAIN),'--unit',TRAIN_UNIT,'--execute']
        evalcmd=[str(ROOT/'.venvs/wam-fixed/bin/python'),'-u','-B',str(ROOT/'scripts/run_robotwin_posttrain_eval_pipeline.py'),'--training-config',str(CONFIG),'--training-run-dir',str(TRAIN/'train'),'--training-unit',TRAIN_UNIT,'--run-dir',str(EVAL),'--selection-seeds-file',str(out/'selection_seeds.json'),'--seeds',*[str(s) for s in SEEDS],'--execute']
        run('validate_posttrain_plan',evalcmd[:-1],timeout=60)
        run('submit_cache_smoke_train',traincmd,timeout=60)
        run('wait_train_then_pure_eval',evalcmd,timeout=172800)
        counts=read(EVAL/'queue_status.json').get('counts',{})
        if not all(counts.get(step,{}).get('scored_episodes')==5 for step in ('25000','30000')):raise RuntimeError('test queue returned without 5 scored episodes at each checkpoint')
        state.update(status='complete',stage='tests_complete',counts=counts);code=0
    except BaseException as e:state.update(status='stopped' if isinstance(e,InterruptedError) else 'failed',error=f'{type(e).__name__}: {e}',traceback=traceback.format_exc())
    finally:
        state['updated_at']=now();atomic_json(out/'status.json',state);write_report(out,state)
    return code
if __name__=='__main__':raise SystemExit(main())
