#!/usr/bin/env python3
"""Queue original fixed V4 tests after Future16 training, without video exports."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness import zhuhai_precision_v4_future16 as future
from scripts import eval_precision_v4_final5 as queue

_binding, _plan = queue.validate_binding, queue.make_plan


def validate_binding(raw, *args, **kwargs):
    result = _binding(raw, *args, **kwargs)
    if (result.get('future_prediction') != future.FUTURE_PREDICTION
            or result.get('future_prediction_horizons') != [16, 32]
            or result.get('execution_prefix') != 8):
        raise queue.e.EvidenceError('Require Future16/32 weights and unchanged exec8')
    return result


def make_plan(args):
    plan = _plan(args)
    plan.update(future_prediction=future.FUTURE_PREDICTION, export_success_videos=False)
    for relative in future.SOURCES:
        path = ROOT / relative
        plan['frozen_sources'][str(path)] = queue.e.sha256_file(path)
    return plan


def main():
    future.install()
    queue.v4 = future
    queue.RUNTIME = ROOT / 'scripts/eval_zhuhai_wam_precision_v4_future16.py'
    queue.validate_binding = validate_binding
    queue.e.validate_binding = validate_binding
    queue.make_plan = make_plan
    return queue.main()


if __name__ == '__main__':
    raise SystemExit(main())
