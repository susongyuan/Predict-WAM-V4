#!/usr/bin/env python3
"""Read-only CPU log/FK audit. Run with .conda-envs/lilawam/bin/python -B <this file>.
Only writes physical_5k_diagnostic.{json,md} alongside this script; no torch/video/inference.
"""
import os,sys,time,json,hashlib,collections
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
import numpy as np
from harness.r1pro_kinematics import R1ProKinematics
OUT=Path(__file__).parent
RUN=ROOT/'outputs/behavior/closed_loop/task0_v3_30k_20260929_s005000_dev10_k1'
TRAIN=ROOT/'outputs/training/task0_v3_30k_20260929_s005000'
T=RUN/'policy_trace_20260929_041112.jsonl'
started=time.time()
def load(p):return json.loads(p.read_text())
def ref(p,line):return f'{p.relative_to(ROOT)}:{line}'
def fieldref(p,field):
 lines=p.read_text().splitlines();return ref(p,next((i for i,l in enumerate(lines,1) if '"'+field+'"' in l),1))+' / '+field
rows=[]
for i,l in enumerate(T.open(),1):
 x=json.loads(l);x['_line']=i;rows.append(x)
ctx=[x for x in rows if x['event']=='v3_context']; chunks=[x for x in rows if x['event']=='chunk']; cm={x['step']:x for x in ctx}
def stats(a):
 a=np.asarray(a,float);return {k:round(float(v),6) for k,v in zip(['min','p50','mean','p95','max'],[min(a),np.median(a),np.mean(a),np.percentile(a,95),max(a)])} if len(a) else None
def ratio(n,d):return {'n':int(n),'denominator':int(d),'fraction':round(n/d,6) if d else None}
def history(group,key):
 ns=[x[key] for x in group]; n=len(ns)
 return {'frames_or_chunks':n,'proprio_lags_nonpadding':{str(l):ratio(sum(v>l for v in ns),n) for l in [0,4,8,16,32,64]},'memory_lags_nonpadding':{str(l):ratio(sum(v>l for v in ns),n) for l in [8,32]},'all_memory_lags_nonpadding':ratio(sum(v>32 for v in ns),n),'all_proprio_lags_nonpadding':ratio(sum(v>64 for v in ns),n),'memory_lag_slots_nonpadding':ratio(sum(v>l for v in ns for l in [8,32]),n*2),'start_slot_older_than_current':ratio(sum(v>1 for v in ns),n)}
boundaries=[]
for x in ctx:
 if x.get('reset_reason'):
  e=[{'event':a['event'],'line':a['_line'],'key':a.get('key')} for a in rows if a.get('step')==x['step'] and a['event'].startswith('orch_')]
  boundaries.append({'step':x['step'],'trace_line':x['_line'],'reason':x['reset_reason'],'matching_boundary_events':e})
errors=[]
for j,x in enumerate(ctx):
 if x['step']!=j: errors.append(['step_gap',j,x['step']])
 expected=1 if x.get('reset_reason') else (ctx[j-1]['context_steps']+1 if j else 1)
 if x['context_steps']!=expected:errors.append(['context_increment',x['step'],expected,x['context_steps']])
for x in chunks:
 if x['v3_context_steps']!=cm[x['step']]['context_steps'] or x['proprio_frames']!=x['v3_context_steps'] or x['proprio_padded_lags']!=sum(l>=x['proprio_frames'] for l in x['proprio_lags']):errors.append(['chunk_history_mismatch',x['step']])
manifest=load(RUN/'manifest.json')
paths=['harness/policy_server.py','harness/wam_v3.py','harness/r1pro_kinematics.py','harness/task0_predicates.py','configs/r1pro_eef_offsets.json','data/omnigibson/omnigibson-robot-assets/models/r1pro/urdf/r1pro.urdf']
provenance=[]
for name in paths:
 p=ROOT/name;h=hashlib.sha256(p.read_bytes()).hexdigest();expected=manifest['artifact_sha256'].get(str(p));provenance.append({'path':name,'sha256_now':h,'sha256_run':expected,'matches_run':h==expected if expected else None})
kin=R1ProKinematics(); attempts=[]
for ai,(lo,hi) in enumerate([(518,1279),(1340,1893),(1964,2414)],1):
 cc=[x for x in chunks if lo<=x['step']<hi]; details=[]; transitions=[];prev=None
 for x in cc:
  s=np.array(x['state_raw']);q=np.array(x['q_cmd_end']);actual=s[42:45];pred=kin.fk('right',np.r_[q[:4],q[11:18]])[0];meas_fk=kin.fk('right',np.r_[s[53:57],s[28:35]])[0]
  status='close' if x['grip'][1]<0 else 'open_or_zero'
  if status!=prev:transitions.append({'step':x['step'],'offset':x['step']-lo,'sampled_first_action':x['grip'][1],'to':status,'trace_line':x['_line']});prev=status
  det=[z for z in rows if z['event']=='perception' and z['step']<=x['step'] and any(v['label']=='radio' for v in z['detections'])]
  d=det[-1] if det else None;target=next((v['xyz_base'] for v in d['detections'] if v['label']=='radio'),None) if d else None
  rec={'step':x['step'],'trace_line':x['_line'],'predicted_terminal_right_eef_m':pred.tolist(),'observed_current_right_eef_m':actual.tolist(),'terminal_minus_current_eef_m':float(np.linalg.norm(pred-actual)),'fk_measured_vs_proprio_m':float(np.linalg.norm(meas_fk-actual)),'right_finger_sum_m':float(s[49:51].sum()),'right_first_action':x['grip'][1], 'phase_argmax':max(x['phase_probabilities'],key=x['phase_probabilities'].get),'right_gate':x['view_gate']['observation.rgb.right_realsense_link_camera_0'],'distance_head_demo_proxy_m':float(np.exp(x['predicted_log_dist'][0])-.02)}
  if target is not None:
   rec.update(perceived_radio_point_m=target,perception_step=d['step'],perception_trace_line=d['_line'],perception_age_steps=x['step']-d['step'],predicted_terminal_to_perceived_point_m=float(np.linalg.norm(pred-target)),observed_current_to_perceived_point_m=float(np.linalg.norm(actual-target)))
  details.append(rec)
 def first(cond):return next((v for v in details if cond(v)),None)
 attempts.append({'attempt':ai,'step_range_half_open':[lo,hi],'chunks':len(cc),'trace_lines':[cc[0]['_line'],cc[-1]['_line']],'history':history(cc,'v3_context_steps'),'phase_argmax_counts':dict(collections.Counter(d['phase_argmax'] for d in details)),'gate_by_view':{v:stats([x['view_gate'][v] for x in cc]) for v in cc[0]['view_gate']},'terminal_minus_current_eef_m':stats([d['terminal_minus_current_eef_m'] for d in details]),'fk_measured_vs_proprio_m':stats([d['fk_measured_vs_proprio_m'] for d in details]),'predicted_terminal_to_perceived_point_m':stats([d['predicted_terminal_to_perceived_point_m'] for d in details if 'perceived_radio_point_m' in d]),'observed_current_to_perceived_point_m':stats([d['observed_current_to_perceived_point_m'] for d in details if 'perceived_radio_point_m' in d]),'perception_age_steps':stats([d['perception_age_steps'] for d in details if 'perceived_radio_point_m' in d]),'same_step_perception_only':{'n':sum(d.get('perception_age_steps')==0 for d in details),'observed_distance_m':stats([d['observed_current_to_perceived_point_m'] for d in details if d.get('perception_age_steps')==0]),'predicted_distance_m':stats([d['predicted_terminal_to_perceived_point_m'] for d in details if d.get('perception_age_steps')==0])},'sampled_gripper_transitions':transitions,'first_sampled_close':first(lambda d:d['right_first_action']<0),'first_sampled_empty_closed':first(lambda d:d['right_first_action']<0 and d['right_finger_sum_m']<.003),'last_chunk':details[-1],'chunk_details':details})
vals=[]
for n in ['validation_000000.json','validation_005000.json']:
 p=TRAIN/n;d=load(p);r=d['report'];vals.append({'source':str(p.relative_to(ROOT)),'step':d['step'],'split':r['split'],'plan_sha256':r['plan_sha256'],'overall':{k:r[k] for k in ['fine_eef_m','fine_hold_eef_m','fine_samples','action_mae','phase_accuracy','loss_total']},'pick':{k:r['by_skill']['pick up from'][k] for k in ['fine_eef_m','fine_hold_eef_m','fine_samples','action_mae','hold_action_mae','phase_accuracy','samples']},'refs':[fieldref(p,'fine_eef_m'),fieldref(p,'pick up from')]})
refs={'trace':str(T.relative_to(ROOT)),'context_record_and_sample':['harness/policy_server.py:825-879','harness/policy_server.py:974-990','harness/policy_server.py:1160-1162'],'history_implementation':['harness/wam_v3.py:54-61','harness/wam_v3.py:601-641'],'memory_model':['harness/wam_v3.py:348-362'],'chunk_semantics':['harness/policy_server.py:1273-1299'],'fk':['harness/r1pro_kinematics.py:6-9','harness/r1pro_kinematics.py:108-127'],'thresholds':['harness/task0_predicates.py:4-9','harness/task0_predicates.py:24-31','harness/task0_predicates.py:76-98'],'train_proxy':['harness/wam_v3.py:129-183','harness/wam_v3.py:569'],'condition':fieldref(RUN/'condition_bank.json','arms')}
limitations=['非padding比例是依据逐帧context_steps实测重建的历史可用率，不等于信息有效率或模型因果利用率。日志没有pool差分、memory投影/attention张量或消融；不能证明非零memory改变了动作。sample调用链传入pool，gate/phase为运行时输出，不以配置声明替代证据。','q_cmd_end是32步chunk末端（index31），部署仅执行index0..7后重规划；不能用它计算8步执行跟踪误差，也不能证明终点将实际执行。state_raw为采样前观测。','grip仅保存每个chunk首动作；夹爪变号是8步级稀疏采样，不是完整逐步切换计数。真实首次切换可在前一采样后至本采样之间，其他短暂切换不可排除。','perceived point为policy当时检测到的radio抓取候选点，使用最近过去观测，非GT/非示教目标；两者均标为base frame但末端参考点可能不同，不是接触/物体可抓性证明。JSON给出观测时差及仅同step感知的子集；旧观测未做底盘运动补偿，不能把全样本距离当同刻真实目标误差。','predicted_log_dist还原exp(log_dist)-0.02仅是弱示教事件距离proxy，不是预测右手XYZ或实测物体距离；phase contact也非实测接触。','FK末端偏置和夹爪EMPTY_MAX=.003/HOLD_RANGE等来自TRAIN标定；只用于事后度量/复核既有判据，未给policy任何审计真值，未硬套示教终点。','当前wam_v3.py是否与该run完全一致由provenance标注；manifest未覆盖时只能将实现作为当前解释，不能升级为历史运行源码证明。']
report={'run':str(RUN.relative_to(ROOT)),'scope':{'read_only_cpu':True,'no_gpu_no_inference_no_video':True,'outputs_only':[str(OUT.relative_to(ROOT)/n) for n in ['analysis.py','physical_5k_diagnostic.json','physical_5k_diagnostic.md']],'initial_audit_started_local':'2026-09-29T04:52:42+08:00','generated_local':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'analysis_seconds':round(time.time()-started,3)},'context':{'rows':len(ctx),'step_range':[ctx[0]['step'],ctx[-1]['step']],'resets':boundaries,'continuity_or_history_errors':errors,'all_steps':history(ctx,'context_steps'),'policy_chunks':history(chunks,'v3_context_steps'),'all_steps_include_navigation_and_hold':True,'encode_ms':stats([x['encode_ms'] for x in ctx]),'total_vision_ms':stats([x['total_vision_ms'] for x in ctx]),'total_vision_over_30hz_budget':ratio(sum(x['total_vision_ms']>1000/30 for x in ctx),len(ctx)),'encode_sum_seconds':sum(x['encode_ms'] for x in ctx)/1000,'hold_encode_sum_seconds':sum(x['encode_ms'] for x in ctx if x['step']>=2414)/1000,'fresh_pooled_true':sum(x['fresh_pooled'] for x in ctx)},'chunks':{'n':len(chunks),'field_counts':dict(collections.Counter(k for x in chunks for k in x if k!='_line')),'missing_for_exact_tracking':['full executed actions indices 0..7','q_cmd index7','per-step gripper commands','per-step measured joint/eef states','memory tensors/activation deltas','explicit predicted xyz target']},'attempts':attempts,'existing_validation_not_recomputed':vals,'source_provenance':provenance,'limitations':limitations,'references':refs}
md=['# 5k physical diagnostic（只读 CPU）','',f'对象 `{RUN.relative_to(ROOT)}`。既有 Q0/3225步与三次空抓审计不重做。诊断起点 04:52:42；产出 '+report['scope']['generated_local']+'。仅解析日志/既有验证、CPU FK；无训练、推理、GPU、视频解码或policy真值输入。','','## 1. Context / lags / memory','',f'全量 {len(ctx)} 条 step 0..3224，连续性/计数/每chunk padding交叉核验异常 {len(errors)}；{len(boundaries)} 次 reset 全有同step orchestrator边界。不是每步reset，但“仅skill/retry/HOLD”不完全准确：另有三次navigation settle模式入口。','', '|reset step|trace行|同step边界|','|---|---|---|']
for b in boundaries:md.append(f"|{b['step']}|{b['trace_line']}|{', '.join(e['event'] for e in b['matching_boundary_events'])}|")
for title,g in [('全控制帧（含导航/HOLD）',report['context']['all_steps']),('实际送WAM采样的chunk',report['context']['policy_chunks'])]:
 md+=['',f"**{title} n={g['frames_or_chunks']}**：memory lag8/32非padding分别 "+' / '.join(f"{v['n']}/{v['denominator']}={v['fraction']:.2%}" for v in g['memory_lags_nonpadding'].values())+f"；全部proprio lag成熟 {g['all_proprio_lags_nonpadding']['fraction']:.2%}。",'proprio各lag非padding：'+', '.join(f"L{k}={v['fraction']:.2%}" for k,v in g['proprio_lags_nonpadding'].items())+'。']
c=report['context'];md+=['',f"encode(ms) p50/p95/max={c['encode_ms']['p50']}/{c['encode_ms']['p95']}/{c['encode_ms']['max']}；total vision={c['total_vision_ms']['p50']}/{c['total_vision_ms']['p95']}/{c['total_vision_ms']['max']}，超33.333ms {c['total_vision_over_30hz_budget']['n']}/{len(ctx)}。encode累计 {c['encode_sum_seconds']:.1f}s，HOLD段 {c['hold_encode_sum_seconds']:.1f}s。是既有run耗时，不是本诊断用GPU。",'',limitations[0],'', 'start slot另存不可变合同起点；配置lag0是当前帧，额外start不体现在proprio_lags列表。有效期计算：context_steps-1 >= lag。来源：'+ '; '.join(refs['history_implementation']+refs['context_record_and_sample'])+'。','','## 2. 三次抓取：可证的量与不可证的量','',f"{len(chunks)}条chunk均保存phase_probabilities、view_gate、predicted_log_dist、grip、q_meas(18)、q_cmd_end(18)、state_raw(61)。字段覆盖计数与逐chunk值在JSON。",'','|attempt / steps|chunks|phase argmax|右腕gate均值|FK终点−当前实测 p50/max (m)|当前/预测至最近感知点 p50 (m；可陈旧)|首个采样close / empty-close|','|---|---:|---|---:|---|---|---|']
for a in attempts:
 close=a['first_sampled_close'];empty=a['first_sampled_empty_closed'];rg=next(v for k,v in a['gate_by_view'].items() if 'right' in k)
 md.append(f"|{a['attempt']} / {a['step_range_half_open']}|{a['chunks']}|{a['phase_argmax_counts']}|{rg['mean']:.3f}|{a['terminal_minus_current_eef_m']['p50']:.3f}/{a['terminal_minus_current_eef_m']['max']:.3f}|{a['observed_current_to_perceived_point_m']['p50']:.3f}/{a['predicted_terminal_to_perceived_point_m']['p50']:.3f}|{close['step'] if close else None}/{empty['step'] if empty else None}|")
for a in attempts:
 md+=['',f"attempt {a['attempt']}：trace:{a['trace_lines'][0]}–{a['trace_lines'][1]}；采样夹爪状态序列 "+', '.join(f"{t['step']}→{t['to']}({t['sampled_first_action']})" for t in a['sampled_gripper_transitions'])+f"；FK(实测q)对proprio误差 p50/max={a['fk_measured_vs_proprio_m']['p50']:.6f}/{a['fk_measured_vs_proprio_m']['max']:.6f}m；感知点age max={a['perception_age_steps']['max']}步。"]
md+=['']+['- '+x for x in limitations[1:6]]+['','字段语义：'+ '; '.join(refs['chunk_semantics']+refs['fk']+refs['thresholds']+refs['train_proxy'])+'。','','## 3. 欠拟合 vs 部署条件/历史缺口：只列证据与30k待验假设','', '|既有固定验证（不是闭环）|0 step|5k|','|---|---:|---:|']
for scope,k in [('overall','fine_eef_m'),('overall','fine_hold_eef_m'),('pick','fine_eef_m'),('pick','fine_hold_eef_m'),('pick','action_mae'),('pick','hold_action_mae'),('pick','phase_accuracy')]:md.append(f"|{scope}.{k}|{vals[0][scope][k]:.6f}|{vals[1][scope][k]:.6f}|")
md+=['','- **训练侧证据**：两次既有验证split=validation且plan_sha256相同；总体与pick显著改善，但5k精细EEF仅略优于hold，pick action MAE仍差于hold。支持“尚有学习/闭环裕量不足”假设，不足以认定训练欠拟合，更不支持单次失败改架构。验证为示教参考动作，不是部署应硬追的真值目标。','- **部署/历史证据**：合同开始确有短暂padding、三次重试各自清空历史；除此大部分采样有完整历史。只能排除每步误reset，不能证明memory在闭环有因果贡献。三次重试同episode、非独立样本。','- **条件证据**：chunk.arm=prototype、condition_bank frozen/source=train-contract-vtt-v1；TRAIN合同prototype，而非此测试episode的示教向量。部署gate/phase与感知点几何为观测证据，phase训练标签为弱事件代理；不能把far直接判作感知失败或“条件失配已确诊”。','- **30k后待验（不在本次执行）**：在相同固定验证计划下比较fine EEF/hold、pick MAE和phase；在可比闭环条件下核对checkpoint SHA/seed/实例/denoise10/K1/exec8，再看三次初始姿态和抓取时机是否改善。若离线改善而闭环仍差，才增强条件/初始状态分布缺口假设；若离线也弱，才增强训练不足假设。memory因果性仍须另行授权的受控验证，不能由本日志补证。','', '**结论**：无逐步context误重置；存在合同/settle/HOLD正常重置与开头padding。现有字段足以量化预测32步终点与当前实测末端/感知候选点的差、8步粒度夹爪时机，却不足证明实际执行目标跟踪误差或memory因果有效。保持trainer主线，不因本run改模型。','','## 来源/可复跑','',f"trace路径：`{T.relative_to(ROOT)}`（上文trace行均指该文件）。"]
for v in vals:md.append('- 验证：'+'；'.join(v['refs']))
md+=['- 条件：'+refs['condition']+'（单行JSON，字段 arms.*[].source / frozen）。','- 源码/URDF/offset当前SHA及run记录比对在JSON source_provenance；'+limitations[-1],f'- 复跑：`PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= .conda-envs/lilawam/bin/python -B {Path(__file__).relative_to(ROOT)}`。只覆盖同目录两个诊断文件。']
(OUT/'physical_5k_diagnostic.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
(OUT/'physical_5k_diagnostic.md').write_text('\n'.join(md)+'\n')
print(json.dumps({'seconds':time.time()-started,'outputs':[str(OUT/'physical_5k_diagnostic.json'),str(OUT/'physical_5k_diagnostic.md')],'errors':errors,'attempt_summary':[{k:a[k] for k in ['attempt','chunks','phase_argmax_counts','terminal_minus_current_eef_m']} for a in attempts]},ensure_ascii=False))
