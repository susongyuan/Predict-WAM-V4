"""Offline paired development report; evaluator truth never returns to policy."""
import argparse
import collections
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from harness.robotwin_policy_worker import atomic_json

def summarize(manifest_path):
    path=Path(manifest_path).resolve();manifest=json.loads(path.read_text());rows=[]
    for job in manifest['jobs']:
        folder=Path(job['run_dir']);result_path=folder/'result.json'
        row={'group':job['group'],'seed':job['seed'],'run_dir':str(folder),'status':'not_started',
             'valid':False,'native_success':None,'failure_stage':None}
        if result_path.exists():
            result=json.loads(result_path.read_text());episodes=result.get('episodes',[])
            row['status']=result.get('status','unknown')
            if episodes:
                e=episodes[0];row.update(episode_status=e.get('status'),control_steps=e.get('control_steps'),
                    middle_system=e.get('middle_system'),failure_phase=e.get('failure_phase'),error=e.get('error'))
                row['valid']=result.get('status')=='completed' and result.get('scored_episodes')==1
                row['native_success']=bool(result['native_successes']) if row['valid'] else None
                gt=folder/e['episode']/'evaluation_gt.jsonl'
                if gt.exists():
                    traces=[json.loads(line) for line in gt.read_text().splitlines()]
                    valid=[v for v in traces if v.get('object_pose_world') and v.get('settle_step',0)==0]
                    if valid:
                        start=valid[0]['object_pose_world'][2]
                        dz=[v['object_pose_world'][2]-start for v in valid]
                        joints=[max(v['cabinet_joint_qpos']) for v in valid if v.get('cabinet_joint_qpos')]
                        row.update(max_object_lift_m=max(dz),final_object_height_delta_m=dz[-1],
                                   max_drawer_open_m=max(joints) if joints else None)
                        if row['valid'] and not row['native_success']:
                            row['failure_stage']=('never_lifted_above_7mm' if max(dz)<=.007 else
                                'lifted_then_fell_below_table' if dz[-1]<-.1 else 'lifted_but_goal_not_achieved')
        rows.append(row)
    groups={}
    for group in sorted(set(r['group'] for r in rows)):
        rr=[r for r in rows if r['group']==group];valid=[r for r in rr if r['valid']]
        groups[group]={'planned':len(rr),'valid':len(valid),'native_successes':sum(r['native_success'] for r in valid),
                       'success_rate':sum(r['native_success'] for r in valid)/len(valid) if valid else None,
                       'incomplete':sum(r['status']!='not_started' and not r['valid'] for r in rr),
                       'not_started':sum(r['status']=='not_started' for r in rr),
                       'failure_stages':dict(collections.Counter(r['failure_stage'] for r in valid if not r['native_success']))}
    report={'scope':'fixed development seeds, matched once per group; not heldout generalization',
            'groups':groups,'rows':rows,'physical_task_verifier_admitted':False,
            'online_task_success_memory_enabled':False,
            'memory_tested':'TRAIN demonstration condition retrieval + scoped controller-postcondition ledger',
            'causal_limits':'middle_vtt vs vtt_only combines feedback supervision and condition retrieval; does not isolate each module',
            'scoring':'native benchmark, independent evaluator only; 120 settle physics steps ~0.48 seconds, not5 seconds'}
    atomic_json(path.parent/'report.json',report)
    text=['# 中下层配对开发测试','',report['scope'],'',
          '| 组 | 完成有效数 | 原生成功 | 未完成/基础设施 | 未启动 |', '|---|---:|---:|---:|---:|']
    for group,g in groups.items():text.append(f"| {group} | {g['valid']}/{g['planned']} | {g['native_successes']} | {g['incomplete']} | {g['not_started']} |")
    text.extend(['','物理抓持/完整容纳验证仍未准入；VTT候选不是成功。未完成和基础设施错误不记为策略失败。',
                 '记忆测试包括TRAIN演示条件检索与控制器后条件账本；任务成功经验在线晋升未启用。',
                 'middle_vtt与vtt_only比较包含反馈监督和条件检索的组合变化，不能分别归因。',''])
    (path.parent/'report.md').write_text('\n'.join(text))
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);a=p.parse_args()
    print(json.dumps(summarize(a.manifest)['groups']))
