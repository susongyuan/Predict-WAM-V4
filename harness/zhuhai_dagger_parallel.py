"""Resource envelopes for independently scheduled DAgger simulator processes."""
import subprocess
from pathlib import Path

GIB=1024**3


class ResourcePressure(RuntimeError):
    """Retain this attempt as partial/failed data; do not count as expert failure."""


def memory():
    rows=dict((k.rstrip(':'),int(v.split()[0])*1024) for k,v in
        (line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines()))
    return dict(available=rows['MemAvailable'],total=rows['MemTotal'],
        swap_fraction=0. if not rows['SwapTotal'] else 1-rows['SwapFree']/rows['SwapTotal'])


def snapshot():
    row=memory()
    value=subprocess.run(['nvidia-smi','--query-gpu=memory.free,utilization.gpu',
        '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=10).stdout.strip().split(',')
    if len(value)!=2:raise ValueError('Expected one GPU resource row')
    row.update(gpu_free_bytes=int(value[0])*1024**2,gpu_utilization_percent=int(value[1]))
    return row


def can_expand(row, active_count, *, maximum=4):
    floor=max(2*GIB,.10*row['total'])
    # Observed Isaac RSS ~6GiB. Reserve seven for a cold environment IN ADDITION
    # to the existing runtime floor, not just a pre-launch free-RAM threshold.
    return (active_count<maximum and row['available']>=7*GIB+floor
        and row['gpu_free_bytes']>=6*GIB and row['swap_fraction']<(.90 if active_count>=3 else .95))


def check_runtime_resources(slot):
    row=memory();floor=max(2*GIB,.10*row['total'])
    if row['available']<floor or row['swap_fraction']>=.95:
        raise ResourcePressure('RAM/swap absolute reserve reached; retained without quota')
    if slot==3 and row['swap_fraction']>=.90:
        raise ResourcePressure('Fourth-worker 90% swap exit; other participants may continue')
    return row
