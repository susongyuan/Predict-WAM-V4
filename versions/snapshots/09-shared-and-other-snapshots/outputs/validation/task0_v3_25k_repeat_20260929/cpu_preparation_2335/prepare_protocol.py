#!/usr/bin/env python3
"""Generate the preregistration and transport hashes; this script never starts a process."""
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parents[2]
FROZEN = BASE / 'frozen_runtime'
RUNNER = BASE / 'runner'
ORIGINAL = ROOT / 'outputs/behavior/closed_loop/task0_v3_30k_20260929_s025000_dev10_k1'
m = json.loads((ORIGINAL / 'manifest.json').read_text())
config = json.loads((FROZEN / 'configs/behavior_environment.json').read_text())
protocol = {
    'schema': 1, 'version': '25krepeat_v1_20260929', 'status': 'prepared_not_launched',
    'user_scope': 'Show the 25k checkpoint and evaluate ten times. User wants a general system and future VLM-driven failure recovery, not task-specific action tuning.',
    'design': 'Ten same-instance stochastic development repeats: reuse the one completed original seed0, add seeds1–9 exactly once. Not cross-instance/cross-task generalization, not an ablation, no outcome-based stopping.',
    'new_seeds': list(range(1, 10)), 'total_evaluations': 10,
    'reused_seed0': {'seed': 0, 'run_dir': str(ORIGINAL), 'success': False, 'q_score': 0.0,
                     'steps': 3225, 'rerun_allowed': False, 'integrity_validation': 'Revalidate original manifest, evaluator result, RNG handshake, traces, resource receipt, video full CPU decode and all 52 original artifact hashes.'},
    'checkpoint': {'path': m['checkpoint'], 'sha256': m['checkpoint_sha256'], 'step': 25000,
                   'selection': 'Fixed explicit checkpoint_step_025000.pt; never best/last/final, never model polling.'},
    'task': {'id': 0, 'name': 'turning_on_radio', 'profile': 'development', 'public_instance_index': 10,
             'resolved_instance_id': 311, 'num_rollouts_per_process': 1, 'same_instance_repeats': True},
    'runtime': {'samples': 1, 'exec_horizon': 8, 'num_inference_steps': 10, 'policy_device': 'cuda',
                'policy_cuda_fraction': 0.30, 'head_rgbd': [720, 720], 'wrist_rgb': [224, 224],
                'viewer_video': [1280, 720], 'video_framing': 'auto', 'max_steps_override': None,
                'episode_budget': 'Original evaluator 1.5 x mean demo length, limit 3224; its strict > timeout check produced 3225 executed steps in seed0. No override.',
                'wall_timeout_seconds': 3600, 'orchestrator_min_budget_steps': json.loads(m['orchestrate'])['min_budget_steps']},
    'orchestrate': m['orchestrate'],
    'recovery': {'vlm': 'none', 'program': 'frozen original Task0 program',
                 'decision': 'Original rule/proprioception postconditions and bounded reapproach/retry (max_attempts=3).',
                 'execution': 'Docking controller for navigation, WAM for manipulation; not Task1 IK primitives.',
                 'user_future_intent': 'VLM-diagnosed recovery remains a later general-system phase; this unchanged baseline is not a VLM recovery evaluation.'},
    'success_definition': 'Only evaluator eval/json/turning_on_radio_311_0.json success and q_score.final determine physical outcome. Program completion, press candidates, VTT closure, holding an object, and internal gates never count as success.',
    'official_score_valid': False, 'simulator': 'Isaac Sim 6.1 port; not official Isaac 5.1 challenge results; PhysX/RTX are not fully deterministic despite RNG seeds.',
    'resource_policy': config['resource_policy'],
    'resource_constraints': 'Original baseline config retained byte-for-byte, never relaxed here. Original short-utilization-spike exception remains as recorded; no new resource authorization is inferred. Simulator and training never overlap. One /tmp/project2_gpu.lock held for the serial chain; no steal/auto-clear. Original cgroups and guards remain in unchanged launcher.',
    'isolation': {'frozen_root': str(FROZEN), 'source_inventory': str(RUNNER / 'freeze_inventory.json'),
                   'original_manifest_sha_is_authority': True,
                   'recovered_original_files': ['harness/policy_server.py', 'harness/task0_predicates.py'],
                   'recovered_from': str(ROOT / 'outputs/validation/task0_post30k_holdguard_20260929/before'),
                   'no_root_source_edits': True, 'no_training_imports': True,
                   'transport_changes_only': 'Local interpreter wrappers explicitly chdir, pin task/camera env, disable ambient PYTHONPATH and user-site influence, then bootstrap a strict verified local finder. Baseline harness/scripts/config bytes unchanged. Both systemd child interpreters go through wrappers.',
                   'shared': 'data/models/third_party and installed conda environment contents reused via symlinks; not globally chmoded, not exhaustive snapshots. All baseline-bound shared assets, including full checkpoint/DINO weights, rehashed before/after each run. Original recording utility additionally bound by video sidecar SHA.'},
    'evidence_and_idempotency': {'run_name': 'task0_v3_25krepeat_v1_20260929_s025000_dev10_seed{1..9}_k1',
                                'outputs': str(FROZEN / 'outputs/behavior/closed_loop'), 'state': str(RUNNER / 'state'),
                                'pending': 'Atomic exclusive pending and process receipts written before launch. Pending without verified complete is never automatically relaunched.',
                                'complete': 'Atomic exclusive receipt only after exact episode identity/seeds, source hashes, resource result, trace coverage, no masked policy exception, full video decode, and child-unit shutdown. Complete receipts are reverified then skipped.',
                                'infrastructure_error': 'Stop whole chain; keep failed run, pending/error receipts, logs and GPU lock. Manual inspection required; never retry/replace a failed infrastructure run silently.',
                                'terminal_reporting': 'JSON events per seed and chain to stdout/journal; physical failures retained and chain continues. Does not poll models.'},
    'prelaunch_requirements': ['CPU tests and dry-run pass.', 'Main workflow reviews actual command before invoking --execute; this preparation starts no GPU/simulator/systemd unit.',
                              'No trainer/simulator process; GPU lock free and original admission guards pass at launch.',
                              'Shared third-party/data/environment edits must be paused during all nine new episodes; source drift fails closed.'],
    'launch_draft': ['systemd-run', '--user', '--collect', '--unit=task0-25krepeat-v1-20260929',
                     '--working-directory=' + str(FROZEN), '--property=MemoryMax=4G', '--property=MemorySwapMax=0',
                     '/usr/bin/python3', '-B', str(RUNNER / 'repeat.py'), '--execute'],
    'cpu_dry_run': ['/usr/bin/python3', '-B', str(RUNNER / 'repeat.py'), '--dry-run']
}
path = BASE / 'protocol.json'
if path.exists():
    raise SystemExit('Protocol already exists; refusing automatic replacement')
path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + '\n')
# The controller and interpreter transports are new infrastructure, not mislabelled original baseline code.
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
files = [path, RUNNER / 'bootstrap.py', RUNNER / 'repeat.py', RUNNER / 'check_closed_loop_video.py', RUNNER / 'freeze_inventory.json']
files += [FROZEN / '.conda-envs' / name / 'bin/python' for name in ('lilawam', 'behavior6')]
(RUNNER / 'runner_integrity.json').write_text(json.dumps({str(p): sha(p) for p in files}, indent=2) + '\n')
print(json.dumps({'protocol': str(path), 'transport_files_bound': len(files), 'launched': False}))
