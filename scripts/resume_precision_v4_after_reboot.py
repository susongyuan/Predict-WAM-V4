#!/usr/bin/env python3
"""Explicit post-reboot continuation; preserve interrupted evidence, never retry."""
from pathlib import Path
import argparse
import fcntl
import json
import os
import signal
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import continue_precision_v4_r2 as s


def execute(run):
    import torch
    from scripts import gpu_lock
    from scripts import train_zhuhai_wam_precision_v4 as trainer
    torch.set_num_threads(4)
    run.mkdir(exist_ok=False)
    with (run / '.writer.lock').open('x') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            s.require(gpu_lock.owner() is None, 'GPU owner remains; do not overlap jobs')
            s.require(not s.EVALUATION.exists(), 'Preserve any existing physical evaluation')
            s.require(not (s.TRAINING / 'failure_guard.json').exists(), 'Training failure requires separate diagnosis')
            old = s.VERIFY / 'r2_continuation_supervisor'
            guard = s.read_json(old / 'failure_guard.json')
            s.require(guard['error'] == 'KeyboardInterrupt: Signal 15; no automatic restart',
                      'Unexpected prior interruption; inspect before continuing')
            short = s.read_json(old / 'short_admission_audit.json')
            s.require(short['status'] == 'passed', 'Short-run admission did not pass')
            contract = s.read_json(s.TRAINING / 'manifest.json')['contract']
            s.require(contract['v4_training']['schedule'] == s.SCHEDULE, 'Schedule changed')
            sources = dict(contract['source_files'])
            sources[str(Path(__file__).resolve())] = s.digest_file(Path(__file__).resolve())
            sources[str(Path(s.__file__).resolve())] = s.digest_file(s.__file__)
            s.verify_sources(sources)
            full = trainer._load_resume_path(s.TRAINING / 'checkpoint_last.pt', contract)
            boundary = full['updates']
            s.require(boundary == 2500, 'Expected the committed 2500-update reboot boundary')
            for tensor in full['model_state_dict'].values():
                s.require(bool(torch.isfinite(tensor).all()), 'Nonfinite committed model')
            counts = s.audit_optimizer(full['optimizer_state_dict'], boundary=boundary)
            s.require(full.get('rng') and full.get('torch_rng') is not None and full.get('cuda_rng'),
                      'Complete RNG state required')
            s.require(not torch.cuda.is_initialized(), 'Admission must remain CPU-only')
            record = dict(status='passed', updates=boundary, step=80000 + boundary,
                full_state=s.receipt(s.TRAINING / 'checkpoint_last.pt'), optimizer=counts,
                previous_guard=str(old / 'failure_guard.json'), previous_guard_preserved=True,
                short_admission_sha256=s.digest_file(old / 'short_admission_audit.json'),
                boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                driver=Path('/proc/driver/nvidia/version').read_text(), explicit_user_continue=True)
            temporary = s.TRAINING / 'checkpoint_last.pt.tmp'
            if temporary.exists():
                preserved = s.TRAINING / 'checkpoint_interrupted_reboot_20261012.pt'
                s.require(not preserved.exists(), 'Interrupted artifact already preserved')
                temporary.rename(preserved)
                record['preserved_interrupted_checkpoint'] = dict(path=str(preserved), size=preserved.stat().st_size)
            del full
            s.write_once(run / 'resume_admission.json', record)
            train, evaluation = s.commands()
            s.write_once(run / 'plan.json', dict(training_command=train, evaluation_command=evaluation,
                frozen_sources=sources, explicit_reboot_continuation=True, automatic_retries=0))
            s.latest(run, 'training')
            s.journal('机器21:13重启中断V4；用户再次明确继续。原guard/tmp证据保留，82500完整断点准入通过，精确恢复2500更新并继续原50k日程。')
            s.run_child(train, 'training', run, sources)
            status = s.check_no_training_failure(s.TRAINING)
            s.require(status.get('status') == 'completed' and status.get('updates') == 50000
                and status.get('step') == 130000 and gpu_lock.owner() is None, 'Formal training incomplete')
            s.latest(run, 'evaluating')
            s.run_child(evaluation, 'evaluation', run, sources)
            summary = s.read_json(s.EVALUATION / 'summary.json')
            s.require(summary.get('status') == 'completed' and len(summary.get('episodes', [])) == 10
                and gpu_lock.owner() is None, 'Final physical evaluation incomplete')
            s.write_once(run / 'completed.json', dict(status='completed', training_status=status,
                evaluation_summary=str(s.EVALUATION / 'summary.json'),
                evaluation_summary_sha256=s.digest_file(s.EVALUATION / 'summary.json'), physical_retries=0))
            s.latest(run, 'completed')
            return 0
        except BaseException as exc:
            error = f'{type(exc).__name__}: {exc}'
            s.latest(run, 'failed', dict(error=error))
            s.write_once(run / 'failure_guard.json', dict(error=error, preserved=True, automatic_retry=False))
            s.journal('V4重启后接续停止：' + error + '；保留证据，不自动重试。')
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    run = s.VERIFY / 'r2_post_reboot_supervisor_20261012'
    if not args.execute:
        print(json.dumps(dict(out=str(run), commands=s.commands(), committed_updates=2500)))
        return 0
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Signal {signum}; preserve this attempt')
    signal.signal(signal.SIGTERM, interrupted)
    return execute(run)


if __name__ == '__main__':
    raise SystemExit(main())
