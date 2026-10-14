#!/usr/bin/env python3
"""Explicit TRAIN-scene diagnostic; never reports held-out generalization."""
from pathlib import Path
import argparse, json, sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from scripts import eval_zhuhai_precision_official_braking as frozen


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--scenario-plan', required=True)
    pre.add_argument('--scenario-index', required=True, type=int)
    known, remainder = pre.parse_known_args()
    plan = json.loads(Path(known.scenario_plan).read_text())
    scenario = plan['scenarios'][known.scenario_index]
    assert scenario['split'] == 'train' and scenario['purpose'] == 'train_scene_diagnostic'
    frozen.configure()
    from harness import isaac_zhuhai_runtime as runtime
    Parent = runtime.IsaacZhuhaiEnv
    class DiagnosticEnv(Parent):
        def setup(self, task, seed, randomized=False, mapping='body_text', source_seed=None):
            if task != 'zhuhai_task2' or seed != scenario['seed']:
                raise ValueError('Diagnostic scene differs from predeclared TRAIN row')
            super().setup(task, seed, randomized=scenario['randomized'], mapping=mapping,
                          source_seed=scenario['source_parent'])
            frozen.base.atomic_json(Path(self.output_dir)/'train_scene_binding.json', dict(
                scenario=scenario, scene_manifest=self.manifest,
                scenario_plan_sha256=frozen.base.identity(known.scenario_plan)['sha256'],
                scope='TRAIN initial scene replay; autonomous actions, no expert targets read by policy'))
    runtime.IsaacZhuhaiEnv = DiagnosticEnv
    sys.argv = [sys.argv[0], *remainder]
    return frozen.base.main()


if __name__ == '__main__': raise SystemExit(main())
