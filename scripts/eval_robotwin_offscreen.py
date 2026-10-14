#!/usr/bin/env python3
"""Offscreen process entry; delegates to the unchanged native evaluator.

Remove desktop session handles before loading SAPIEN. Do not override exit codes,
skip cleanup, intercept renderer errors or alter observations/actions/scoring.
"""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for name in ('DISPLAY', 'WAYLAND_DISPLAY'):
    os.environ.pop(name, None)
from scripts.eval_robotwin_wam import main

if __name__ == '__main__':
    raise SystemExit(main())
