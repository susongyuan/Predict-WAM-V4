#!/usr/bin/env python3
"""Train the existing V4 schedule with Future horizons +16/+32."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness import zhuhai_precision_v4_future16 as future
from scripts import train_zhuhai_wam_precision_v4 as trainer

_diagnostic = trainer.diagnostic


def diagnostic(*args, **kwargs):
    result = _diagnostic(*args, **kwargs)
    for split in result.values():
        split['means']['future16'] = sum(row['future16'] for row in split['rows']) / split['count']
    return result


def main():
    future.install()
    trainer.__file__ = __file__
    trainer.diagnostic = diagnostic
    return trainer.main()


if __name__ == '__main__':
    raise SystemExit(main())
