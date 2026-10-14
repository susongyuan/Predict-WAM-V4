#!/usr/bin/env python3
"""Offscreen evaluator; recording is off unless explicitly requested.

Delegates policy, physics, observation, action bounds and scoring to the native
evaluator. Without --record-video, no third-view camera or ffmpeg is created.
"""
import argparse
import functools
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
for name in ('DISPLAY','WAYLAND_DISPLAY'):os.environ.pop(name,None)
from scripts import eval_robotwin_wam as native
from harness.robotwin_policy_worker import atomic_json


class DisabledRecording:
    def __init__(self,directory,episode,seed):
        self.directory=Path(directory);self.episode=episode;self.seed=seed;self.closed=False

    def capture(self,env,camera,control_step,settle_step=0):
        # Deliberately does not access env, render, allocate a camera or write RGB.
        return None

    def close(self,complete=False):
        if self.closed:return
        self.closed=True
        atomic_json(self.directory/'recording.json',dict(enabled=False,status='disabled_by_user',
                    episode=self.episode,seed=self.seed,video_files_created=0,
                    third_view_camera_created=False))


def disabled_factory(env,directory,episode,seed,fps,ffmpeg=None):
    return DisabledRecording(directory,episode,seed),None


def main(argv=None):
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--record-video',action='store_true')
    preference,remaining=parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    original_run=native.run_evaluation;original_plan=native.plan_evaluation
    def plan(args):
        result=original_plan(args)
        result['recording']=dict(enabled=preference.record_video,
            default='off; explicit user request required',third_view_policy_access=False)
        return result
    native.plan_evaluation=plan
    if not preference.record_video:
        native.run_evaluation=functools.partial(original_run,video_factory=disabled_factory)
    try:
        return native.main(remaining)
    finally:
        native.run_evaluation=original_run;native.plan_evaluation=original_plan


if __name__=='__main__':raise SystemExit(main())
