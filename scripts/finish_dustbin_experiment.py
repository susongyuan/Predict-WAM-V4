#!/usr/bin/env python3
"""Await already-submitted training, run pre-registered pure tests, write user report."""
from pathlib import Path
import datetime,json,subprocess,sys,fcntl,signal,os,traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from harness.robotwin_policy_worker import atomic_json
OUT=ROOT/'outputs/pipelines/dustbin_20261007'
EVAL=ROOT/'outputs/runs/put_bottles_dustbin/pure_25k_30k_5each_20261007'
TRAIN=ROOT/'outputs/runs/put_bottles_dustbin/wam_30k_20261007/train'
def read(p):
 try:return json.loads(Path(p).read_text())
 except (OSError,ValueError):return {}
def stamp():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def main():
 claim=OUT/'evaluation_supervisor_claim.json'
 with claim.open('x') as f:json.dump({'pid':os.getpid(),'started_at':stamp()},f)
 cmd=[str(ROOT/'.venvs/wam-fixed/bin/python'),'-u','-B',str(ROOT/'scripts/run_robotwin_posttrain_eval_pipeline.py'),'--training-config',str(ROOT/'configs/fixed_bimanual/robotwin_dustbin_50_wam.yaml'),'--training-run-dir',str(TRAIN),'--training-unit','p2-dustbin-wam30k-20261007','--run-dir',str(EVAL),'--selection-seeds-file',str(OUT/'selection_seeds.json'),'--seeds','920001','920002','920003','920004','920005','--camera-readback','serial','--execute']
 state={'status':'running','stage':'waiting_for_training_then_pure_eval','started_at':stamp(),'command':cmd};atomic_json(OUT/'status.json',state);rc=1
 try:
  with (OUT/'posttrain_supervisor.log').open('x') as log:
   rc=subprocess.call(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
  q=read(EVAL/'queue_status.json');counts=q.get('counts',{})
  complete=rc==0 and all(counts.get(k,{}).get('scored_episodes')==5 for k in ('25000','30000'))
  state.update(status='complete' if complete else 'incomplete_or_failed',counts=counts,returncode=rc,training_status=read(TRAIN/'status.json'),evaluation_status=read(EVAL/'pipeline_status.json'))
 except BaseException as e:state.update(status='failed',error=repr(e),traceback=traceback.format_exc())
 finally:
  state['updated_at']=stamp();atomic_json(OUT/'status.json',state)
  lines=['# 双臂Aloha瓶子入垃圾桶：30k WAM实验',f"\n更新：{stamp()}；状态：{state['status']}",
    '\n任务：RoboTwin2 `put_bottles_dustbin`，官方clean50示教，40/5/5划分。双臂机体；三个刚体瓶投入敞口垃圾桶。**不包含开盖或真实柔性物体。**',
    '\n纯WAM：三RGB+原生14D commanded state、零条件、chunk32/exec8、K1/Euler10；无VTT/VLM/反馈恢复/专家动作。GT只评分。',
    '\n|检查点|有效完成|原生成功|原生失败|初始化无效|其他未完成|成功率|','|---|---:|---:|---:|---:|---:|---:|']
  for step in ('25000','30000'):
   c=state.get('counts',{}).get(step,{});rate=c.get('native_success_rate');lines.append(f"|{step}|{c.get('scored_episodes',0)}/5|{c.get('native_successes','未测试')}|{c.get('native_failures','未测试')}|{c.get('invalid_initializations',0)}|{c.get('incomplete_attempts',0)}|{rate if rate is not None else '不可计算'}|")
  lines+=['\n两检查点使用相同预登记920001–920005种子，每条独立进程、保存onboard和第三视角录像。每条最多1700控制步；5100秒episode/5400秒total/5460秒systemd界限，防止用旧700步预算误截断。',
   '\n原生成功仅检查三个瓶子原点在垃圾桶的指定xy/z范围；不证明严格完整容纳、释放或持久稳定。无效/中断不计策略失败。',
   f'\n训练证据：`{TRAIN.relative_to(ROOT)}`；测试逐次结果和录像：`{EVAL.relative_to(ROOT)}`。']
  if not state.get('counts'):lines+=['\n目前无有效测试成绩；不得将空结果写成0/5。']
  report='\n'.join(lines)+'\n';(OUT/'report.md').write_text(report)
  with (ROOT/'docs/archive/legacy_mobile_experiments_20261004.md').open('a') as f:
   fcntl.flock(f,fcntl.LOCK_EX);f.write('\n\n## 2026-10-07瓶子入垃圾桶实验自动回执\n\n'+report);f.flush();fcntl.flock(f,fcntl.LOCK_UN)
 return 0 if state['status']=='complete' else 1
if __name__=='__main__':raise SystemExit(main())
