#!/usr/bin/env python3
"""Derive the v41 hard-prefix-commitment evaluation stack from the audited v37 files.

Every literal substitution must match exactly once and every regex exactly once;
the resulting diffs against the v37 sources are the complete change set. Frozen
v36/v37 files are never modified. Re-running refuses to overwrite.
"""
from pathlib import Path
import hashlib
import json
import re
import subprocess
ROOT = Path(__file__).resolve().parents[1]


def patch(src, dst, subs, regex_subs=()):
    text = (ROOT/src).read_text()
    for old, new in subs:
        n = text.count(old)
        if n != 1:
            raise SystemExit(f'{src}: literal count {n} != 1: {old[:90]!r}')
        text = text.replace(old, new)
    for pattern, new in regex_subs:
        text, n = re.subn(pattern, lambda _m: new, text, flags=re.S)
        if n != 1:
            raise SystemExit(f'{src}: regex count {n} != 1: {pattern[:90]!r}')
    out = ROOT/dst
    if out.exists():
        raise SystemExit(f'{dst} exists; refuse to overwrite')
    out.write_text(text)
    leftovers = [l.strip() for l in text.splitlines() if 'v37' in l and 'evidence_v37' not in l]
    return dict(sha256=hashlib.sha256(text.encode()).hexdigest(), source=src, remaining_v37_lines=leftovers)


REPLAY_DIAG = (
"            diagnostic=reply['prefix_commitment']\n"
"            e.require(diagnostic['commitment']=='hard_replacement_inpainting_normalized_space'\n"
"                and diagnostic['physical_inference_delay_actions']==0 and diagnostic['execution_horizon']==8\n"
"                and diagnostic['model_weights_updated'] is False\n"
"                and diagnostic['stage_progress_or_grasp_oracle'] is False,'changed commitment contract')\n"
"            e.require(diagnostic['previous_query_step']==(None if i==0 else i-8)\n"
"                and diagnostic['committed_actions']==(0 if i==0 else 24),'wrong commitment physical time alignment')\n"
"            if i==0:\n"
"                e.require(diagnostic['previous_unexecuted_native_chunk'] is None\n"
"                    and diagnostic['denoising_trace']==[] and diagnostic['fresh_rows']==[0,32],'first query received a fictitious prefix')\n"
"            else:\n"
"                e.require(np.array_equal(diagnostic['previous_unexecuted_native_chunk'],previous_chunk[8:]),\n"
"                    'prior commitment shifted, replayed, or substituted')\n"
"                e.require(np.array_equal(raw[:24],previous_chunk[8:]),'committed prefix is not bit-exact in the new chunk')\n"
"                e.require(diagnostic['fresh_rows']==[24,32] and len(diagnostic['denoising_trace'])==10 and all(t['tau']==k/10\n"
"                    and e.finite(t['flow_rms']) and e.finite(t['suffix_flow_rms'])\n"
"                    for k,t in enumerate(diagnostic['denoising_trace'])),'commitment denoising evidence missing')\n")

TEST_SAMPLE = (
"    def sample(features,inputs,seed,step,event):\n"
"        previous=raw if step else None\n"
"        fresh=raw-np.r_[np.ones(7)*.1,0,np.ones(7)*.1,0]\n"
"        chunk=raw if step==0 else np.vstack([previous[8:],fresh[24:]])\n"
"        policy.commit_sampler.last_diagnostic=dict(commitment='hard_replacement_inpainting_normalized_space',\n"
"            physical_inference_delay_actions=0,execution_horizon=8,model_weights_updated=False,stage_progress_or_grasp_oracle=False,\n"
"            previous_query_step=None if step==0 else step-8,committed_actions=0 if step==0 else 24,\n"
"            fresh_rows=[0,32] if step==0 else [24,32],\n"
"            previous_unexecuted_native_chunk=None if step==0 else previous[8:].tolist(),\n"
"            denoising_trace=[] if step==0 else [dict(tau=i/10,flow_rms=0.,suffix_flow_rms=0.,pinned_prefix_flow_rms=0.) for i in range(10)],\n"
"            suffix_boundary=None if step==0 else dict(position_jump_rad=0.,within_prefix_step_rad=0.,within_suffix_step_rad=0.))\n"
"        calls.append(step);return chunk\n")

PREFLIGHT = (
"    preflight=e.read_json(BASE/'verification/dagger80k_extra20_20261011_v24/prefix_commit_preflight_v41.json')\n"
"    e.require(preflight['cpu']['status']=='passed' and preflight['gpu']['status']=='passed'\n"
"              and preflight['gpu']['checkpoint_sha256']==metadata['sha256'], 'Actual frozen-model commitment preflight missing')\n"
"    e.require(preflight['source_sha256']==e.sha256_file(ROOT/'harness/zhuhai_prefix_commit_v41.py'), 'Commitment changed after preflight')\n"
"    e.require(str(SEEDS[0]) in preflight['gpu']['seeds'] and preflight['gpu']['seeds'][str(SEEDS[0])]['first_chunk_exact_original'] is True,\n"
"              'Preflight did not cover the preregistered seed')\n")

PAIRED = (
"    paired_rows=[row for row in paired['episodes'] if row['seed'] in SEEDS]\n"
"    e.require(len(paired_rows)==len(SEEDS),'Paired original20 row missing for preregistered seed')\n"
"    plan['paired_original20']=dict(path=str(paired_path),sha256=e.sha256_file(paired_path),\n"
"        paired_seeds=list(SEEDS),scores=[row['observed_base_score'] for row in paired_rows],\n"
"        full_successes=sum(row['full_task_success'] for row in paired_rows))\n")

SOURCES = (
"    for name in ('scripts/eval_precision_continuity1_v41.py','scripts/eval_zhuhai_precision_continuity_v41.py',\n"
"                 'scripts/precision_continuity_evidence_v37.py','scripts/precision_continuity_physics_abort_v41.py',\n"
"                 'scripts/precision_continuity_replay_v41.py','scripts/test_precision_continuity_v41.py',\n"
"                 'harness/zhuhai_chunk_continuity_v41.py','harness/zhuhai_observer_video_v41.py','harness/zhuhai_prefix_commit_v41.py',\n"
"                 'scripts/check_prefix_commit_v41.py','scripts/progress1_log_v41.py'):\n")

COMMIT_SETTINGS = (
"    settings['prefix_commitment'] = dict(sources=['https://arxiv.org/html/2506.07339v1','https://sjtu-deng-lab.github.io/Streaming-WAM/'],\n"
"        implementation_sha256=hashlib.sha256((Path(__file__).parent/'zhuhai_prefix_commit_v41.py').read_bytes()).hexdigest(),\n"
"        synchronous=True, physical_delay_actions=0, committed_old_indices=[8,32], committed_new_indices=[0,24], fresh_indices=[24,32],\n"
"        method='hard replacement inpainting of the previous unexecuted actions along the rectified-flow interpolant in normalized action space; fresh suffix conditioned through the model input',\n"
"        denoising_steps=10, checkpoint_modified=False, task_progress_oracle=False, soft_guidance=False)\n")

results = {}
results['harness/zhuhai_chunk_continuity_v41.py'] = patch('harness/zhuhai_chunk_continuity_v37.py', 'harness/zhuhai_chunk_continuity_v41.py', [
    ('"""Independent arm-continuity overlay; frozen evaluators never import this.', '"""Independent arm-continuity overlay with v41 hard prefix commitment; frozen evaluators never import this.'),
    ("VERSION = 'precision-v2-rtc-soft-overlap-c1-fine250-v37'", "VERSION = 'precision-v2-hard-prefix-commit-c1-fine250-v41'"),
    ("no_prefix_training=True, full_asynchronous_RTC=False,soft_overlap_guidance=True, no_task_state_intervention=True))", "no_prefix_training=True, full_asynchronous_RTC=False,hard_prefix_commitment=True, no_task_state_intervention=True))"),
    ("extra = dict(overlap_guidance=self.overlap_sampler.last_diagnostic,raw_denormalized_chunk=raw.tolist(),", "extra = dict(prefix_commitment=self.commit_sampler.last_diagnostic,raw_denormalized_chunk=raw.tolist(),"),
    ("execution_reason='RTC_soft_overlap_generation_then_original_spline_C1_and_official_fine250_braking',", "execution_reason='hard_prefix_commit_generation_then_original_spline_C1_and_official_fine250_braking',"),
], [(r"    settings\['overlap_guidance'\] = dict\(.*?task_progress_oracle=False\)\n", COMMIT_SETTINGS)])

results['harness/zhuhai_observer_video_v41.py'] = patch('harness/zhuhai_observer_video_v37.py', 'harness/zhuhai_observer_video_v41.py', [
    ('SEED=932106', "SEED=int(os.environ.get('P2_CONTINUITY_THIRD_VIEW_SEED','932106'))"),
    ("f'80k RTC continuity v37 | seed{SEED} | {now:.1f}s | step{self.frames}'", "f'80k hard-prefix-commit v41 | seed{SEED} | {now:.1f}s | step{self.frames}'"),
    ("schema='requested-third-person-video-v37'", "schema='requested-third-person-video-v41'"),
])

results['scripts/eval_zhuhai_precision_continuity_v41.py'] = patch('scripts/eval_zhuhai_precision_continuity_v37.py', 'scripts/eval_zhuhai_precision_continuity_v41.py', [
    ('from harness import zhuhai_chunk_continuity_v37 as braking', 'from harness import zhuhai_chunk_continuity_v41 as braking'),
    ('from scripts.progress1_log import log', 'from scripts.progress1_log_v41 import log'),
    ("    return os.environ.get('P2_CONTINUITY_THIRD_VIEW_SEED')=='932106' and seed==932106", "    requested=os.environ.get('P2_CONTINUITY_THIRD_VIEW_SEED')\n    return requested is not None and str(seed)==requested"),
    ("changes=['generation-time RTC soft-overlap VJP guidance with synchronous delay0',", "changes=['generation-time hard prefix commitment: previous unexecuted actions8:32 pinned as new0:24 by replacement inpainting, fresh24:32, synchronous delay0',"),
    ('        from harness.zhuhai_overlap_guidance_v37 import OverlapSampler\n        sample = OverlapSampler(wrapper, condition, lambda: original.verify_owner(args.gpu_lock_owner))', '        from harness.zhuhai_prefix_commit_v41 import PrefixCommitSampler\n        sample = PrefixCommitSampler(wrapper, condition, lambda: original.verify_owner(args.gpu_lock_owner))'),
    ('        policy.overlap_sampler = sample', '        policy.commit_sampler = sample'),
    ("system_mode='precision_wam_v2_rtc_soft_overlap_c1_fine250'", "system_mode='precision_wam_v2_hard_prefix_commit_c1_fine250'"),
    ('    from harness import zhuhai_observer_video_v37\n    zhuhai_observer_video_v37.install()', '    from harness import zhuhai_observer_video_v41\n    zhuhai_observer_video_v41.install()'),
])

results['scripts/precision_continuity_replay_v41.py'] = patch('scripts/precision_continuity_replay_v37.py', 'scripts/precision_continuity_replay_v41.py', [
    ('"""Independent v37 audit of v36 targets; accepts only the archived exact float32 cast."""', '"""Independent v41 audit of committed-generation targets; accepts only the archived exact float32 cast."""'),
    ('from harness import zhuhai_chunk_continuity_v37 as c', 'from harness import zhuhai_chunk_continuity_v41 as c'),
    ('def video_evidence(directory,raw):', 'def video_evidence(directory,raw,recorded_seed):'),
    ("    if raw['seed']!=932106:", "    if raw['seed']!=recorded_seed:"),
    ("    e.require((raw.get('observer_video_receipt',receipt)==receipt) and receipt['seed']==932106", "    e.require((raw.get('observer_video_receipt',receipt)==receipt) and receipt['seed']==recorded_seed"),
], [(r"            diagnostic=reply\['overlap_guidance'\]\n.*?'guidance denoising evidence missing'\)\n", REPLAY_DIAG)])

results['scripts/precision_continuity_physics_abort_v41.py'] = patch('scripts/precision_continuity_physics_abort_v37.py', 'scripts/precision_continuity_physics_abort_v41.py', [
    ('from scripts import eval_zhuhai_precision_continuity_v37 as evaluator', 'from scripts import eval_zhuhai_precision_continuity_v41 as evaluator'),
    ('from harness import zhuhai_chunk_continuity_v37 as braking', 'from harness import zhuhai_chunk_continuity_v41 as braking'),
])

results['scripts/eval_precision_continuity1_v41.py'] = patch('scripts/eval_precision_continuity1_v37.py', 'scripts/eval_precision_continuity1_v41.py', [
    ('"""One preregistered diagnostic scene with C1/fine250 execution; first case observer video explicitly requested."""', '"""One preregistered diagnostic scene with v41 hard prefix commitment and C1/fine250 execution; first case observer video explicitly requested."""'),
    ('from scripts import eval_zhuhai_precision_continuity_v37 as runtime', 'from scripts import eval_zhuhai_precision_continuity_v41 as runtime'),
    ('from scripts import precision_continuity_physics_abort_v37 as contact', 'from scripts import precision_continuity_physics_abort_v41 as contact'),
    ('from scripts.progress1_log import log', 'from scripts.progress1_log_v41 import log'),
    ('from harness import zhuhai_chunk_continuity_v37 as braking', 'from harness import zhuhai_chunk_continuity_v41 as braking'),
    ('SCHEMA = "precision-80k-rtc-continuity-single-diagnostic-v37"', 'SCHEMA = "precision-80k-hard-prefix-commit-single-diagnostic-v41"'),
    ("    from scripts.precision_continuity_replay_v37 import replay,video_evidence\n    replayed=replay(path.parent,changed['action_postprocessing'])\n    files.update(video_evidence(path.parent,raw))",
     "    from scripts.precision_continuity_replay_v41 import replay,video_evidence\n    replayed=replay(path.parent,changed['action_postprocessing'])\n    files.update(video_evidence(path.parent,raw,SEEDS[0]))"),
    ('gpu_lock_owner="zhuhai-rtc-single-20261012-v37",', 'gpu_lock_owner="zhuhai-prefix-commit-single-20261012-v41",'),
    ('systemd_unit="p2-zhuhai-rtc-single-20261012-v37.service",', 'systemd_unit="p2-zhuhai-prefix-commit-single-20261012-v41.service",'),
    ("    plan['recording']=dict(requested=True,recorded_seeds=[932106],selection='first preregistered case',", "    plan['recording']=dict(requested=True,recorded_seeds=[SEEDS[0]],selection='first preregistered case',"),
    ('    return dict(schema="80k-rtc-single-scored-trajectory-analysis-v37", time=now(),', '    return dict(schema="80k-hard-prefix-commit-single-scored-trajectory-analysis-v41", time=now(),'),
    ('scope="One source development diagnostic; fixed weights, soft-overlap generation plus C1/fine250; task-phase semantic failures remain possible, no heldout or success-rate claim")', 'scope="One source development diagnostic; fixed weights, hard prefix commitment generation plus C1/fine250; task-phase semantic failures remain possible, no heldout or success-rate claim")'),
    ('record_video=True, third_view=True, recorded_seeds=[932106], randomized=False, original_five=plan["original_five"],', 'record_video=True, third_view=True, recorded_seeds=[SEEDS[0]], randomized=False, original_five=plan["original_five"],'),
    ('recorded_seeds=[932106], source_count=len(plan["extension_sources"]))))', 'recorded_seeds=[SEEDS[0]], source_count=len(plan["extension_sources"]))))'),
], [
    (r"    preflight=e\.read_json\(BASE/'verification/dagger80k_extra20_20261011_v24/overlap_guidance_preflight_v37\.json'\)\n.*?'Guidance changed after preflight'\)\n", PREFLIGHT),
    (r'interpretation="One preregistered seed932106 [^"]*",', 'interpretation=f"One preregistered seed{SEEDS[0]} with unchanged80k and generation-time hard prefix commitment plus C1/fine250; paired development diagnostic, no heldout or success-rate claim",'),
    (r'user_authorization="[^"]*",', 'user_authorization="用户2026-10-12：先诊断改进整臂跨块抖动与无效晃动/闭合拖延，实质改进后一例完整真实第三视角；不恢复十测、不加训采集；旧有效失败不重试",'),
    (r"    plan\['paired_original20'\]=dict\(.*?paired\['episodes'\]\[:1\]\)\)\n", PAIRED),
    (r"    for name in \('scripts/eval_precision_continuity1_v37\.py'.*?'scripts/check_overlap_guidance_v37\.py'\):\n", SOURCES),
    (r"    plan\['child_env'\]\['P2_CONTINUITY_THIRD_VIEW_SEED'\]='932106' *\n", "    plan['child_env']['P2_CONTINUITY_THIRD_VIEW_SEED']=str(SEEDS[0])\n"),
    (r'log\("80k RTC新单例932106：[^"]*", "root"\)', 'log(f"80k硬前缀承诺新单例{SEEDS[0]}：预注册生成侧硬前缀承诺+C1/250Hz原官方制动，原严格评分/2400步/源分布，真实第三视角；旧有效失败不重试、不续十测/训练/采集。", "root")'),
    (r'log\(f"80kRTC单例\{len\(rows\)\}/1 seed[^"]*", "root"\)', 'log(f"80k硬承诺单例{len(rows)}/1 seed{item[\'seed\']}：{row[\'status\']}，得分{row[\'observed_base_score\']}，整任务成功{row[\'full_task_success\']}。", "root")'),
])

results['scripts/test_precision_continuity_v41.py'] = patch('scripts/test_precision_continuity_v37.py', 'scripts/test_precision_continuity_v41.py', [
    ('from harness import zhuhai_chunk_continuity_v37 as c', 'from harness import zhuhai_chunk_continuity_v41 as c'),
    ('    policy.overlap_sampler=SimpleNamespace(last_diagnostic=dict(cpu_mock=True))', '    policy.commit_sampler=SimpleNamespace(last_diagnostic=dict(cpu_mock=True))'),
    ('        assert np.array_equal(targets[:,c.GRIPPER],np.repeat(np.clip(raw[step%8,c.GRIPPER],0,1)[None],25,axis=0))', '        assert np.array_equal(targets[:,c.GRIPPER],np.repeat(np.clip((raw if step<8 else raw[8:])[step%8,c.GRIPPER],0,1)[None],25,axis=0))'),
    ('    from scripts.precision_continuity_replay_v37 import replay', '    from scripts.precision_continuity_replay_v41 import replay'),
    ('    from harness import zhuhai_observer_video_v37 as video', '    from harness import zhuhai_observer_video_v41 as video'),
    ("    out=ROOT/'outputs/zhuhai2026_tasks12/verification/dagger80k_extra20_20261011_v24/continuity_cpu_checks_v37.json'", "    out=ROOT/'outputs/zhuhai2026_tasks12/verification/dagger80k_extra20_20261011_v24/continuity_cpu_checks_v41.json'"),
    ("        frozen_sources_edited=False,success_rate_claim=False)", "        frozen_sources_edited=False,success_rate_claim=False,hard_prefix_commitment_contract=True)"),
], [(r"    def sample\(features,inputs,seed,step,event\):\n.*?calls\.append\(step\);return raw if step==0 else raw-np\.r_\[np\.ones\(7\)\*\.1,0,np\.ones\(7\)\*\.1,0\]\n", TEST_SAMPLE)])

for dst, info in results.items():
    diff = subprocess.run(['diff', str(ROOT/info['source']), str(ROOT/dst)], capture_output=True, text=True).stdout
    info['diff_lines'] = len(diff.splitlines())
receipt = dict(schema='derive-continuity-v41-from-v37', files=results,
               derivation_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
out = ROOT/'outputs/zhuhai2026_tasks12/verification/dagger80k_extra20_20261011_v24/derive_continuity_v41_receipt.json'
if out.exists():
    raise SystemExit('receipt exists')
out.write_text(json.dumps(receipt, ensure_ascii=False, indent=1)+'\n')
print(json.dumps(receipt, ensure_ascii=False, indent=1))
