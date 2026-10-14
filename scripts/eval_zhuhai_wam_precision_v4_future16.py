#!/usr/bin/env python3
"""Strict Future16 V4 weights, original exec8 and original simulator scoring."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness import zhuhai_precision_v4_future16 as future
from scripts import eval_zhuhai_wam_precision_v4 as runtime

_serve = runtime.serve


def serve(channel, policy):
    policy.runtime.update(future_prediction=future.FUTURE_PREDICTION,
                          future_prediction_horizons=[16, 32], execution_prefix=8)
    return _serve(channel, policy)


def main():
    future.install()
    runtime.v4 = future
    runtime.__file__ = __file__
    runtime.serve = serve
    return runtime.main()


if __name__ == '__main__':
    raise SystemExit(main())
