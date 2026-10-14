#!/usr/bin/env python3
"""Read-only CPU video evidence checker. No simulator/model imports.

Usage: python3 -B scripts/check_closed_loop_video.py RUN [--out NEW_JSON]
Default output is stdout, never RUN/summary.json. Exit 0 means all discovered
scored episodes passed; exit 2 means not_verified (including missing tools).
Verification covers decoded video evidence, NOT task success or visual framing.

Binding evidence: omnigibson/eval/eval.py:195-196,246 in the Isaac6 source:
  eval/json/{task}_{instance_id}_{rollout_id}.json
  eval/videos/{task}_{instance_id}_{rollout_id}.mp4
Historical example: loading_the_car_311_0.json -> loading_the_car_311_0.mp4.
Never glob for an arbitrary MP4; instance_id is not a development index.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import subprocess
import sys

MIN_WIDTH, MIN_HEIGHT = 1280, 720


def _identity(row):
    if not isinstance(row, dict):
        raise ValueError('episode must be an object')
    task = row.get('task')
    if not isinstance(task, str) or not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_-]*', task):
        raise ValueError('missing or unsafe task identity')
    ids = [row.get(k) for k in ('instance_id', 'rollout_id')]
    if any(type(x) is not int or x < 0 for x in ids):
        raise ValueError('missing or invalid instance_id/rollout_id')
    return task, *ids


def _run(argv, timeout):
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1',
           'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout,
                          env=env, check=False)


def _inside(path, parent):
    return path.resolve().is_relative_to(parent.resolve())


def check_run(run_dir, *, ffprobe='ffprobe', ffmpeg='ffmpeg', timeout=120, pyav_python=None):
    """Return JSON-serializable evidence; never write to the run directory."""
    if timeout <= 0:
        raise ValueError('timeout must be positive')
    root = Path(run_dir).resolve()
    report = {'schema_version': 1, 'run': str(root), 'status': 'not_verified',
              'all_verified': False, 'minimum_dimensions': [MIN_WIDTH, MIN_HEIGHT],
              'scope': 'discovered eval/json episodes only; not task success, official score, '
                       'planned-episode completeness or visual framing',
              'issues': [], 'episodes': []}
    records = []
    paths = sorted((root / 'eval/json').glob('*.json'))
    if not paths:
        report['issues'].append('no_episode_json')
    for path in paths:
        item = {'episode_json': str(path.relative_to(root)), 'status': 'not_verified',
                'binding_verified': False, 'decode_verified': False, 'issues': []}
        report['episodes'].append(item)
        try:
            if not _inside(path, root):
                raise ValueError('episode JSON resolves outside run')
            row = json.loads(path.read_text())
            identity = _identity(row)
            item['identity'] = dict(zip(('task', 'instance_id', 'rollout_id'), identity))
            records.append((path, row, identity, item))
        except (OSError, ValueError, TypeError) as exc:
            item['issues'].append('invalid_episode: ' + str(exc))
    counts = Counter(identity for _, _, identity, _ in records)
    for path, row, identity, item in records:
        stem = '_'.join(map(str, identity))
        video = root / 'eval/videos' / (stem + '.mp4')
        item['video'] = str(video.relative_to(root))
        if counts[identity] != 1:
            item['issues'].append('duplicate_episode_identity')
        if path.name != stem + '.json':
            item['issues'].append('episode_filename_identity_mismatch')
        if not _inside(video, root):
            item['issues'].append('video_resolves_outside_run')
        if item['issues']:
            continue
        item['binding_verified'] = True
        steps = row.get('steps')
        if type(steps) is not int or steps <= 0:
            item['issues'].append('missing_or_invalid_steps')
            continue
        item.update(steps=steps, expected_frames=steps + 1)
        if not video.is_file() or video.stat().st_size == 0:
            item['issues'].append('missing_or_empty_bound_video')
            continue
        sidecar = video.with_name(video.name + '.json')
        item['sidecar'] = str(sidecar.relative_to(root))
        try:
            if not _inside(sidecar, root):
                raise ValueError('video sidecar resolves outside run')
            metadata = json.loads(sidecar.read_text())
            if not isinstance(metadata, dict):
                raise ValueError('video sidecar is not an object')
            checks = {
                'status': metadata.get('status') == 'completed',
                'view': metadata.get('view') == 'viewer',
                'environment_steps': type(metadata.get('environment_steps')) is int and metadata['environment_steps'] == steps,
                'frames': type(metadata.get('frames')) is int and metadata['frames'] == steps + 1,
                'coverage': metadata.get('complete_frame_coverage') is True,
                'success': type(metadata.get('success')) is bool and metadata['success'] is row.get('success'),
            }
            for key, ok in checks.items():
                if not ok:
                    item['issues'].append('sidecar_mismatch_' + key)
        except (OSError, ValueError, TypeError) as exc:
            item['issues'].append('sidecar_not_verified: ' + str(exc))
            continue
        if pyav_python:
            # Existing PyAV links FFmpeg; decode every frame on one CPU thread.
            # Header frame counts are never substituted for a real decode.
            code = """import av,json,sys
container=av.open(sys.argv[1], options={'err_detect':'explode'})
assert len(container.streams.video)==1, 'expected one video stream'
stream=container.streams.video[0]; stream.thread_count=1
count=0; dims=set()
for frame in container.decode(stream):
    assert not frame.is_corrupt, 'corrupt decoded frame'
    count+=1; dims.add((frame.width,frame.height))
container.close()
assert len(dims)==1, 'missing or varying frame dimensions'
w,h=next(iter(dims))
print(json.dumps({'width':w,'height':h,'frames':count,'pyav_version':av.__version__}))
"""
            argv = [str(pyav_python), '-B', '-c', code, str(video)]
            item['decode_argv'] = argv
            try:
                decoded = _run(argv, timeout)
                if decoded.returncode or decoded.stderr.strip():
                    raise ValueError(f'PyAV decode rc={decoded.returncode}: {decoded.stderr.strip()[:2000]}')
                result = json.loads(decoded.stdout)
                width, height, count = (result[k] for k in ('width', 'height', 'frames'))
                if any(type(v) is not int or v <= 0 for v in (width, height, count)):
                    raise ValueError('invalid decoded dimensions/count')
                item.update(dimensions=[width,height], counted_frames=count, decode_returncode=0,
                            decoder='PyAV / CPU FFmpeg', pyav_version=result['pyav_version'], decode_verified=True)
                if width < MIN_WIDTH or height < MIN_HEIGHT:
                    item['issues'].append('dimensions_below_1280x720')
                if count != steps + 1:
                    item['issues'].append('frame_count_mismatch')
                if not item['issues']:
                    item['status'] = 'verified'
            except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
                item['issues'].append('decode_not_verified: ' + str(exc))
            continue
        probe_argv = [ffprobe, '-v', 'error', '-threads', '1', '-select_streams', 'v:0',
                      '-count_frames', '-show_entries', 'stream=width,height,nb_read_frames',
                      '-of', 'json', str(video)]
        item['probe_argv'] = probe_argv
        try:
            probe = _run(probe_argv, timeout)
            if probe.returncode != 0 or probe.stderr.strip():
                raise ValueError(f'ffprobe rc={probe.returncode}: {probe.stderr.strip()[:2000]}')
            streams = json.loads(probe.stdout).get('streams')
            if not isinstance(streams, list) or len(streams) != 1:
                raise ValueError('expected exactly one selected video stream')
            stream = streams[0]
            width, height = stream.get('width'), stream.get('height')
            if type(width) is not int or type(height) is not int or min(width, height) <= 0:
                raise ValueError('missing or invalid dimensions')
            item['dimensions'] = [width, height]
            # nb_frames is often zero/absent for fragmented MP4: require counted frames.
            counted = stream.get('nb_read_frames')
            if isinstance(counted, bool) or not re.fullmatch(r'[0-9]+', str(counted)):
                raise ValueError('missing or invalid nb_read_frames; header nb_frames is not evidence')
            item['counted_frames'] = int(counted)
            if width < MIN_WIDTH or height < MIN_HEIGHT:
                item['issues'].append('dimensions_below_1280x720')
            if int(counted) != steps + 1:
                item['issues'].append('frame_count_mismatch')
        except (OSError, ValueError, TypeError, AttributeError, subprocess.TimeoutExpired) as exc:
            item['issues'].append('probe_not_verified: ' + str(exc))
            continue
        decode_argv = [ffmpeg, '-nostdin', '-hide_banner', '-v', 'error', '-xerror',
                       '-err_detect', 'explode', '-hwaccel', 'none', '-threads', '1',
                       '-i', str(video), '-map', '0:v:0', '-an', '-sn', '-dn',
                       '-threads', '1', '-filter_threads', '1', '-filter_complex_threads', '1',
                       '-f', 'null', '-']
        item['decode_argv'] = decode_argv
        try:
            decoded = _run(decode_argv, timeout)
            item['decode_returncode'] = decoded.returncode
            if decoded.returncode != 0 or decoded.stderr.strip():
                item['issues'].append(f'decode_not_verified: rc={decoded.returncode}: '
                                      + decoded.stderr.strip()[:2000])
            else:
                item['decode_verified'] = True
        except (OSError, subprocess.TimeoutExpired) as exc:
            item['issues'].append('decode_not_verified: ' + str(exc))
        if not item['issues'] and item['decode_verified']:
            item['status'] = 'verified'
    report['all_verified'] = bool(report['episodes']) and all(
        e['status'] == 'verified' for e in report['episodes']) and not report['issues']
    if report['all_verified']:
        report['status'] = 'verified'
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--out', type=Path, help='new JSON outside the historical run; default stdout')
    parser.add_argument('--pyav-python', type=Path, help='existing Python with PyAV; full CPU decode instead of ffprobe/ffmpeg CLI')
    parser.add_argument('--ffprobe', default='ffprobe')
    parser.add_argument('--ffmpeg', default='ffmpeg')
    parser.add_argument('--timeout', type=float, default=120, help='seconds per tool invocation')
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error('--timeout must be positive')
    if args.out and (_inside(args.out, args.run_dir) or args.out.exists()):
        parser.error('--out must be a new file outside the historical run')
    report = check_run(args.run_dir, ffprobe=args.ffprobe, ffmpeg=args.ffmpeg, timeout=args.timeout, pyav_python=args.pyav_python)
    text = json.dumps(report, ensure_ascii=False, indent=2) + '\n'
    if args.out:
        with args.out.open('x') as file:
            file.write(text)
    else:
        print(text, end='')
    return 0 if report['all_verified'] else 2


if __name__ == '__main__':
    sys.exit(main())
