#!/usr/bin/env python3
"""Construct only this evaluation's local image. CPU/file operations; no model or sim imports."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path('/home/user/Robotic/project2')
BASE = ROOT / 'outputs/validation/task0_v3_25k_repeat_20260929'
FROZEN = BASE / 'frozen_runtime'
ORIGINAL = ROOT / 'outputs/behavior/closed_loop/task0_v3_30k_20260929_s025000_dev10_k1'
BEFORE = ROOT / 'outputs/validation/task0_post30k_holdguard_20260929/before'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def main():
    manifest = json.loads((ORIGINAL / 'manifest.json').read_text())
    if (BASE / 'runner/freeze_inventory.json').exists():
        raise RuntimeError('Image already finalized; refusing rebuild')
    FROZEN.mkdir(exist_ok=True)  # Resume only an incomplete build; existing files must match below.
    rows = []
    for original, expected in manifest['artifact_sha256'].items():
        relative = Path(original).relative_to(ROOT)
        source = BEFORE / relative if (BEFORE / relative).is_file() else Path(original)
        actual = sha(source)
        if actual != expected:
            raise RuntimeError(f'Original baseline SHA mismatch: {relative}: {actual} != {expected}')
        copy = relative.parts[0] in ('harness', 'scripts', 'configs') or str(relative).startswith('outputs/validation/')
        target = FROZEN / relative if copy else Path(original)
        if copy:
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copyfile(source, target)
            target.chmod(0o444)
            assert sha(target) == expected
        rows.append({'original': original, 'relative': str(relative), 'source': str(source),
                     'runtime': str(target), 'sha256': expected, 'baseline_bound': True, 'copied': copy})
    # Baseline did not hash the inert package marker or lock/video verifier. Keep supplemental provenance explicit.
    supplemental = ['harness/__init__.py', 'scripts/gpu_lock.py']
    for relative in supplemental:
        target = FROZEN / relative
        if not target.exists():
            shutil.copyfile(ROOT / relative, target)
        assert sha(target) == sha(ROOT / relative)
        target.chmod(0o444)
        rows.append({'original': str(ROOT / relative), 'relative': relative, 'source': str(ROOT / relative),
                     'runtime': str(target), 'sha256': sha(target), 'baseline_bound': False, 'copied': True})
    checker = BASE / 'runner/check_closed_loop_video.py'
    shutil.copyfile(ROOT / 'scripts/check_closed_loop_video.py', checker)
    # Recording code has an independent original video-sidecar fingerprint.
    recording = ROOT / 'third_party/BEHAVIOR-1K-isaac6/OmniGibson/omnigibson/eval/utils/recording_utils.py'
    metadata = json.loads((ORIGINAL / 'eval/videos/turning_on_radio_311_0.mp4.json').read_text())
    assert sha(recording) == metadata['recording_utils_sha256']
    rows.append({'original': str(recording), 'relative': str(recording.relative_to(ROOT)),
                 'source': str(recording), 'runtime': str(recording), 'sha256': sha(recording),
                 'baseline_bound': 'video_sidecar', 'copied': False})
    for directory in ('data', 'models', 'third_party'):
        (FROZEN / directory).symlink_to(ROOT / directory, target_is_directory=True)
    # Never link the entire outputs directory: all new evidence stays inside this image.
    (FROZEN / 'outputs/training').symlink_to(ROOT / 'outputs/training', target_is_directory=True)
    for name in ('lilawam', 'behavior6'):
        env = FROZEN / '.conda-envs' / name
        (env / 'bin').mkdir(parents=True)
        for child in (ROOT / '.conda-envs' / name).iterdir():
            if child.name != 'bin':
                (env / child.name).symlink_to(child, target_is_directory=child.is_dir())
        # systemd children do not inherit the launcher's environment. Explicit wrappers therefore pin cwd,
        # Python isolation and camera/task settings independently for BOTH interpreters.
        script = '#!/bin/sh\nset -eu\n'
        script += f'cd {FROZEN}\n'
        script += 'unset PYTHONPATH PYTHONHOME PYTHONSAFEPATH\n'
        script += 'export P2_TASK_ID=0 P2_HEAD_RES=720 P2_WRIST_RES=224 PYTHONDONTWRITEBYTECODE=1\n'
        script += f'exec {ROOT}/.conda-envs/{name}/bin/python -B -E -s {BASE}/runner/bootstrap.py {name} "$@"\n'
        (env / 'bin/python').write_text(script)
        (env / 'bin/python').chmod(0o555)
    inventory = {'schema': 1, 'original_root': str(ROOT), 'frozen_root': str(FROZEN),
                 'baseline_manifest': str(ORIGINAL / 'manifest.json'),
                 'baseline_manifest_sha256': sha(ORIGINAL / 'manifest.json'), 'files': rows,
                 'shared_links': {d: str(ROOT / d) for d in ('data', 'models', 'third_party', 'outputs/training')},
                 'limitation': 'Shared third-party/data/installed environment files are reused, not filesystem write-protected; every baseline-bound artifact is fully rehashed before and after each episode. Large datasets/environments are not copied or exhaustively baseline-bound.'}
    (BASE / 'runner/freeze_inventory.json').write_text(json.dumps(inventory, indent=2) + '\n')
    print(json.dumps({'baseline_files_verified': len(manifest['artifact_sha256']),
                      'copied_files': sum(r['copied'] for r in rows), 'all_hashes_match': True}))


if __name__ == '__main__':
    main()
