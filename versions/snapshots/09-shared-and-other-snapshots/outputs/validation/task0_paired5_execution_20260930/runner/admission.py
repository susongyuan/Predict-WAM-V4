"""Receipt-bound resource overlay, shared by both images. No environment-only authorization."""
import hashlib
import json
import os
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
PROJECT = Path('/home/user/Robotic/project2')
AUTH = PROJECT / 'outputs/validation/task0_gpu_peak_authorization_20260930/authorization.json'
AUTH_SHA = '9bd06b2fbdda711378c8e31b2b7605da3fd3ed0453e6ebf2c797925ec5c69cad'
USER_TEXT = '允许gpu超过85%的限制，允许达到峰值，我觉得测试和训练应该可以并行吧'
GPU_LOCK = Path('/tmp/project2_gpu.lock')

def require(ok, message):
    if not ok: raise RuntimeError(message)

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def authorize():
    require(AUTH.resolve() == AUTH and AUTH.is_file(), 'Authorization receipt missing or redirected')
    require(sha(AUTH) == AUTH_SHA, 'Current-user authorization receipt changed; no env override accepted')
    row = json.loads(AUTH.read_text())
    require(row['source'] == 'actual_user_message_in_current_session' and row['user_text'] == USER_TEXT,
            'No matched current-user GPU consent')
    require(row['gpu_peak_above_85_percent_allowed'] is True and row['gpu_utilization_percent_allowed'] == 100,
            'GPU peak execution not authorized')
    require(row['cpu_percent_ceiling_unchanged'] == row['ram_percent_ceiling_unchanged'] == 85,
            'CPU/RAM authorization changed')
    require(row['operational_total_vram_abort_mib'] == 15300, 'VRAM operational guard changed')
    return row

def resource_policy(old):
    return {**old, 'gpu_execution_enabled':True, 'gpu_total_vram_limit_mib':15300,
            'requested_total_limit_percent':100, 'admission_total_percent':85,
            # Legacy capacity helper multiplies total VRAM by this field. CPU/RAM have separate limits below.
            'monitor_exit_percent':100, 'cpu_exit_percent':85, 'ram_exit_percent':85,
            'gpu_utilization_exit_percent':100, 'poll_seconds':0.25,
            'resource_adapter':'paired5_receipt_bound_gpu_peak_v1', 'authorization_sha256':AUTH_SHA,
            'monitor_is_polling_not_instantaneous_quota':True}

def violation(snapshot):
    if snapshot['memory_used_percent'] >= 85: return 'global RAM threshold reached'
    if snapshot['cpu_used_percent'] >= 85: return 'global CPU threshold reached'
    if not snapshot['gpu']: return 'GPU monitoring unavailable'
    gpu = snapshot['gpu'][0]
    if gpu['vram_used_mib'] >= min(15300, gpu['vram_total_mib']): return 'global VRAM threshold reached'
    if not 0 <= gpu['utilization_percent'] <= 100: return 'Invalid GPU utilization telemetry'
    return None

def live_ticket(variant, run_name):
    authorize()
    import re
    match = re.fullmatch(rf'task0_paired5_{variant}_20260930_s025000_dev10_seed([0-4])_k1', run_name)
    require(match is not None, 'Unregistered paired5 run')
    seed = int(match[1])
    require(not (variant == 'original' and seed == 0), 'Original seed0 is reuse-only')
    state = BASE / 'state' / variant / f'seed{seed}'
    require(state.resolve() == state, 'State path redirected')
    pending = json.loads((state / 'pending.json').read_text())
    require(not (state / 'complete.json').exists() and not (state / 'infrastructure_error.json').exists(), 'Terminal ticket cannot launch')
    require(pending['authorization_sha256'] == AUTH_SHA and pending['variant'] == variant and pending['seed'] == seed,
            'Invalid execution ticket')
    pid = pending['controller_pid']
    args = (Path('/proc') / str(pid) / 'cmdline').read_bytes().split(b'\0')
    require(str(BASE / 'runner/paired5.py').encode() in args and b'--execute' in args, 'No live explicit controller')
    require(GPU_LOCK.is_file() and not GPU_LOCK.is_symlink() and GPU_LOCK.read_text().strip() == pending['lock_owner'],
            'GPU lease is absent or belongs to another job')
    return pending

def install_overlay():
    authorize()
    import behavior_preflight as preflight
    import behavior_resources as resources
    preflight.CONFIG['resource_policy'] = resource_policy(preflight.CONFIG['resource_policy'])
    preflight.CONFIG['authorization'] = {**preflight.CONFIG.get('authorization', {}),
          'source':str(AUTH), 'sha256':AUTH_SHA, 'short_gpu_utilization_spikes_allowed':True}
    resources.violation = violation
