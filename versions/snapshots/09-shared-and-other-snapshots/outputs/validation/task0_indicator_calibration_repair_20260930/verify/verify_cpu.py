#!/usr/bin/env python3
"""Independent CPU-only verification. Writes evidence only beside this file."""
import ast
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import random
import resource
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path('/home/user/Robotic/project2')
BASE = ROOT / 'outputs/validation/task0_indicator_calibration_repair_20260930'
HERE = Path(__file__).resolve().parent
SOURCE = ROOT / 'scripts/fit_task0_indicator.py'
BEFORE = BASE / 'source_before/scripts/fit_task0_indicator.py'
METADATA = ROOT / 'data/behavior-1k/2026-challenge-demos/skill_segments_task000.json'
FIT_OUTPUT = ROOT / 'outputs/validation/task0_indicator_20260928/fit.json'
PARAMS = {'min_area': 12, 'ring_green': 0.35, 'ring_red': 0.6}
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ.pop('TASK0_INDICATOR_TEST_SOURCE', None)
allowed = os.sched_getaffinity(0)
os.sched_setaffinity(0, {min(allowed)})
# Each process has at most one CPU and 3 GiB virtual address space.
resource.setrlimit(resource.RLIMIT_AS, (3 * 1024 ** 3, 3 * 1024 ** 3))
sys.path.insert(0, str(ROOT))

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def dump(name, obj):
    with (HERE / name).open('x') as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2)
        handle.write('\n')

def rows(frames, ep=0, truth='off', prediction='on'):
    return [{'ep': ep, 'frame': f, 'truth': truth,
             'g': [(50, 1.0)] if prediction == 'on' else [],
             'r': [(50, 1.0)] if prediction == 'off' else []} for f in frames]

def worker():
    import cv2
    import numpy as np
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    from harness import task0_indicator as ti
    new = load('verify_new', SOURCE)
    old = load('verify_old', BEFORE)
    old.DEMO = METADATA.parent
    result = {'admitted': False, 'scope': 'CPU metadata and synthetic fixtures only'}
    document = json.loads(METADATA.read_text())
    expected = {}
    for ep, record in document['episodes'].items():
        presses = [s for s in record['segments'] if s['skill'] == 'press']
        assert len(presses) == 1
        expected[int(ep)] = (presses[0]['start'], presses[0]['end'])
    actual = new.press_segments(METADATA)
    assert actual == expected and len(actual) == 200
    result['real_metadata'] = {'episodes': len(actual), 'all_ranges_exact_match': True,
                               'sha256': sha(METADATA)}
    try:
        old.press_segments()
        raise AssertionError('Original unexpectedly accepted real metadata')
    except AttributeError as exc:
        result['baseline_nested_error'] = str(exc)
    result['streak_counterexamples'] = []
    for indices in ([0, 2, 4], [0, 100, 200], [0, 2, 6], [0, 0, 0], [4, 2, 0]):
        baseline = old.episode_level(rows(indices), PARAMS)
        try:
            current = new.episode_level(rows(indices), PARAMS)
        except ValueError as exc:
            current = {'error': type(exc).__name__, 'message': str(exc)}
        result['streak_counterexamples'].append({'frames': indices, 'before': baseline, 'after': current})
    # Oracle enumerates complete matching windows rather than reusing the implementation's run counter.
    rng = random.Random(73019)
    for case in range(400):
        stride = 1 + case % 2
        fixture = []
        for ep in range(rng.randrange(1, 5)):
            frame = rng.randrange(0, 5)
            for _ in range(rng.randrange(0, 18)):
                fixture.extend(rows([frame], ep=ep, truth=rng.choice(['on', 'off']),
                                    prediction=rng.choice(['on', 'off', 'unknown'])))
                frame += rng.choice([stride, stride, stride, stride + 1, stride + 4])
        # Interleave episodes while preserving each episode's original order.
        fixture.sort(key=lambda r: (r['frame'], r['ep']))
        expected = {'episodes': len({r['ep'] for r in fixture}), 'on_run_after': 0,
                    'on_run_before': 0, 'off_run_before': 0, 'off_run_after': 0}
        count = 5 if stride == 1 else 3
        for ep in {r['ep'] for r in fixture}:
            scoped = [r for r in fixture if r['ep'] == ep]
            for truth in ('on', 'off'):
                for prediction in ('on', 'off'):
                    valid = False
                    for start in range(len(scoped) - count + 1):
                        window = scoped[start:start + count]
                        temporal = [r['frame'] for r in window] == [window[0]['frame'] + j * stride for j in range(count)]
                        values = all(r['truth'] == truth and bool(r['g' if prediction == 'on' else 'r']) for r in window)
                        valid |= temporal and values
                    expected[f"{prediction}_run_{'after' if truth == 'on' else 'before'}"] += int(valid)
        assert new.episode_level(fixture, PARAMS, stride) == expected, (case, fixture)
    result['independent_window_oracle'] = {'cases': 400, 'strides': [1, 2], 'all_passed': True}
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        data = root / 'data/chunk-000'
        data.mkdir(parents=True)
        (data / 'fixture.parquet').touch()
        rewards = [0, 0, 1, 0, 0, 0, 0, 0, -1, 0, 0, 0]
        table = SimpleNamespace(column=lambda c: {'episode_index': [0] * 12,
             'frame_index': list(range(12)), 'next.reward': rewards}[c])
        with patch.object(old, 'DEMO', root), patch.object(old.pq, 'read_table', return_value=table):
            toggles = old.toggle_frames()
        (root / 'episode_0.jpgpack').touch()
        pack = SimpleNamespace(length=12, has=lambda f: True, rgb=lambda f: None)
        with patch.object(old, 'CACHE', root), patch.object(old, 'FramePack', return_value=pack), patch.object(old, 'comps', return_value=([], [])):
            collected = old.collect([0], toggles, {0: (0, 12)}, None)
        assert {x['frame']: x['truth'] for x in collected}[8] == 'on'
        result['baseline_reward_counterexample'] = {'negative_reward_frame': 8, 'old_truth': 'on', 'old_toggle': toggles}
    for name, call in [('toggle_frames', lambda: new.toggle_frames()), ('collect', lambda: new.collect([0], {0: 2}, {0: (0, 12)}, None)), ('collect_empty', lambda: new.collect([], {}, {}, None))]:
        try:
            call()
            raise AssertionError(f'{name} did not refuse')
        except new.CalibrationRefused:
            result[name] = 'CalibrationRefused'
    unrelated = np.zeros((224, 224, 3), dtype=np.uint8)
    cv2.circle(unrelated, (35, 35), 7, (30, 210, 40), -1)
    mixed = ti._synthetic((215, 25, 20))
    cv2.rectangle(mixed, (5, 5), (45, 45), (15, 15, 15), -1)
    cv2.circle(mixed, (25, 25), 7, (30, 210, 40), -1)
    result['unadmitted_detector_still_has_counterexamples'] = {
        'unrelated_green': ti.classify(unrelated)[0],
        'off_target_plus_unrelated_green': ti.classify(mixed)[0],
        'missing': ti.classify(None)[0]}
    assert list(result['unadmitted_detector_still_has_counterexamples'].values()) == ['on', 'on', 'unknown']
    for image in (unrelated, mixed, ti._synthetic(None)):
        assert old.comps(image, ti.DEFAULT) == new.comps(image, ti.DEFAULT)
    result['component_extraction_parity'] = {'fixtures': 3, 'all_equal': True}
    # Document unsupported provenance / temporal meanings without inventing admission.
    flat = [{'episode': 0, 'skill': 'press', 'start': 0, 'end': 999999, 'split': 'dev', 'verified': True}]
    result['explicit_boundaries'] = {
        'legacy_flat_has_no_length_or_split_provenance_checks': new.parse_press_segments(flat),
        'stride3_frames_0_3_proxy_count': new.episode_level(rows([0, 3]), PARAMS, 3)['on_run_before'],
        'stride5_single_frame_proxy_count': new.episode_level(rows([0]), PARAMS, 5)['on_run_before'],
        'interpretation': 'Structural/cached-count diagnostics only. Neither flat range/source provenance nor live-frame duration is established.'}
    result['prohibited_modules_loaded'] = [k for k in sys.modules if k.split('.')[0] in {'torch', 'isaacsim', 'omnigibson', 'tensorflow'}]
    assert not result['prohibited_modules_loaded']
    assert not FIT_OUTPUT.exists()
    result['fit_exists'] = False
    result['passed'] = True
    dump('independent_cases.json', result)
    print(json.dumps({'passed': True, 'metadata_episodes': 200, 'oracle_cases': 400, 'admitted': False}))

def main():
    os.environ['TMPDIR'] = str(HERE / 'tmp')
    (HERE / 'tmp').mkdir(exist_ok=True)
    before_receipt = json.loads((BASE / 'source_before_sha256.json').read_text())['sha256_before']
    author = json.loads((BASE / 'repair_evidence.json').read_text())
    tracked = [ROOT / 'progress.md', SOURCE, ROOT / 'scripts/test_fit_task0_indicator.py', METADATA,
               BEFORE, BASE / 'calibration_repair.patch']
    tracked.extend(ROOT / p for p in before_receipt if p.startswith('harness/'))
    pre = {str(p): sha(p) for p in tracked}
    runtime_match = {p: pre[str(ROOT / p)] == h for p, h in before_receipt.items() if p.startswith('harness/')}
    assert all(runtime_match.values())
    assert pre[str(BEFORE)] == before_receipt['scripts/fit_task0_indicator.py']
    assert all(sha(ROOT / p) == h for p, h in author['changed_sha256'].items())
    # Hash only copied, small frozen files; never traverse model/data payloads.
    inventory = json.loads((ROOT / 'outputs/validation/task0_v3_25k_repeat_20260929/runner/freeze_inventory.json').read_text())
    frozen = {}
    for entry in inventory['files']:
        if not entry.get('copied'):
            continue
        p = Path(entry['runtime'])
        assert p.stat().st_size < 2 * 1024 ** 2, str(p)
        frozen[str(p)] = {'sha256': sha(p), 'matches_inventory': sha(p) == entry['sha256']}
        tracked.append(p)
    assert all(x['matches_inventory'] for x in frozen.values())
    pre.update({str(p): sha(p) for p in tracked})
    commands = []
    def run(label, argv, expected, extra=None):
        env = dict(os.environ)
        if extra:
            env.update(extra)
        completed = subprocess.run(argv, cwd=ROOT, env=env, text=True, capture_output=True, timeout=40)
        with (HERE / f'{label}.log').open('x') as stream:
            stream.write('COMMAND ' + json.dumps(argv) + '\nSTDOUT\n' + completed.stdout + '\nSTDERR\n' + completed.stderr)
        commands.append({'name': label, 'argv': argv, 'expected_exit': expected, 'exit': completed.returncode, 'passed': completed.returncode == expected})
        assert completed.returncode == expected, label
    for label, python in [('system', '/usr/bin/python3'), ('project', str(ROOT / '.conda-envs/lilawam/bin/python'))]:
        run('tests_' + label, [python, '-B', '-m', 'unittest', 'discover', '-s', str(ROOT / 'scripts'), '-p', 'test_fit_task0_indicator.py', '-v'], 0)
    python = str(ROOT / '.conda-envs/lilawam/bin/python')
    run('independent_cases', [python, '-B', str(Path(__file__).resolve()), '--worker'], 0)
    selected = ['ParserTests.test_real_nested_shape', 'StreakTests.test_sparse_or_missing_frames_cannot_accumulate',
                'StreakTests.test_duplicate_or_reverse_frames_are_rejected_not_sorted',
                'RewardBoundaryTests.test_positive_then_negative_rewards_cannot_create_toggle_labels',
                'RewardBoundaryTests.test_legacy_collector_refuses_permanent_on_labels',
                'RewardBoundaryTests.test_even_empty_collection_cannot_be_mistaken_for_trusted_zero_errors']
    run('baseline_selected_expected_failure', [python, '-B', str(ROOT / 'scripts/test_fit_task0_indicator.py'), *selected, '-v'], 1,
        {'TASK0_INDICATOR_TEST_SOURCE': str(BEFORE)})
    run('cli_default', ['/usr/bin/python3', '-B', str(SOURCE)], 2)
    run('cli_metadata', ['/usr/bin/python3', '-B', str(SOURCE), '--check-segments'], 0)
    post = {str(p): sha(p) for p in tracked}
    assert pre == post
    assert not FIT_OUTPUT.exists()
    dump('verification.json', {'passed': True, 'admitted': False,
        'resource_controls': {'cpu_affinity': list(os.sched_getaffinity(0)), 'native_threads': 1, 'address_space_limit_bytes': 3 * 1024 ** 3, 'CUDA_VISIBLE_DEVICES': ''},
        'commands': commands, 'before_sha256': pre, 'after_sha256': post,
        'four_harness_files_match_author_before_receipt': runtime_match,
        'frozen_copied_files': frozen, 'fit_exists': False,
        'limits': ['No model or frame-cache payload hashing; patch and scoped SHA checks are not a global filesystem-history audit.',
                   'No GPU, simulation, training, downloading, or threshold fitting commands were executed by this verification.',
                   'No real RGB/reward reads or dev/report tuning; only real Task0 segment metadata and synthetic fixtures.',
                   'No claim of physical labels, calibrated accuracy, outcome admission, or Task0 capability.']})
    print(json.dumps({'passed': True, 'tests': [28, 28], 'frozen_copied_files_checked': len(frozen), 'admitted': False}))

if __name__ == '__main__':
    worker() if '--worker' in sys.argv else main()
