#!/usr/bin/env python3
"""Explicitly authorized, exclusive-owner GPU probe; NEVER acquires a GPU lock.

Main must supply systemd RuntimeMaxSec/TimeoutStopSec/KillMode=control-group and
hold the matching owner before launch. This script starts ONE persistent worker
and sends the same six saved TRAIN montages with the unchanged CPU-probe prompt.
No HDF, GT, policy, VAL/heldout/dev data or task execution. CPU unit tests import
this module without loading torch/CUDA; --help also does not touch the GPU.

  .conda-envs/lilawam/bin/python -B scripts/probe_robotwin_vlm_gpu.py \
      --run-dir NEW_DIRECTORY --gpu-lock-owner MAIN_OWNED_LOCK --dtype bf16
"""
from __future__ import annotations
import argparse
import base64
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness import robotwin_vlm_worker as vlm

PYTHON = ROOT / '.conda-envs/lilawam/bin/python'
SOURCE = ROOT / 'outputs/validation/robotwin_visible_probe_train_20261006'
CASE_IDS = tuple(f'episode_{episode:07d}_{phase}' for episode in (0, 1) for phase in ('first', 'middle', 'last'))


def atomic(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_ready(value):
    if not isinstance(value, dict) or value.get('type') != 'ready':
        raise vlm.ProtocolError('no_ready: ' + str(value)[:700])
    vlm.keys(value, ('version', 'type', 'status', 'model', 'resources', 'operations', 'capabilities',
                     'policy_condition', 'max_new_tokens', 'native_shape', 'views', 'montage_size',
                     'verify_prompt_sha256'))
    if (type(value.get('version')) is not int or value.get('version') != vlm.VERSION or value.get('status') != vlm.STATUS
            or type(value.get('max_new_tokens')) is not int or value.get('max_new_tokens') != 192
            or value.get('native_shape') != list(vlm.SHAPE) or value.get('views') != list(vlm.VIEWS)
            or value.get('montage_size') != [960, 264] or value.get('capabilities') != [vlm.CAPABILITY]
            or value.get('operations') != ['verify', 'ground']
            or value.get('policy_condition') != 'zero_single_task'
            or value.get('verify_prompt_sha256') != vlm.digest(vlm.VERIFY_PROMPT.encode())):
        raise vlm.ProtocolError('incompatible_worker_ready')
    model = value.get('model', {})
    if (model.get('id') != vlm.MODEL_ID or model.get('revision') != vlm.REVISION
            or model.get('device') != 'cuda:0' or model.get('dtype') not in ('bf16', 'fp16')
            or model.get('torch_threads') != 2 or not isinstance(value.get('resources'), dict)):
        raise vlm.ProtocolError('unexpected_model_or_resources')
    return value


def check_response(value, request):
    req = vlm.decode_request(request)
    vlm.keys(value, ('version', 'type', 'id', 'operation', 'binding', 'status', 'outcome', 'parsed',
                     'raw_text', 'errors', 'resources', 'timing'),
             ('prompt_sha256', 'montage_pixel_sha256', 'generation'))
    if (type(value.get('version')) is not int or not isinstance(value, dict) or value.get('type') != 'response' or value.get('version') != vlm.VERSION
            or value.get('status') != vlm.STATUS or value.get('id') != req['id']
            or value.get('operation') != req['operation'] or value.get('binding') != vlm.request_binding(req)):
        raise vlm.ProtocolError('response_id_operation_or_observation_binding_mismatch')
    if 'verified' in value or 'success' in value:
        raise vlm.ProtocolError('unauthorized_authority_claim')
    if type(value.get('raw_text')) is not str or len(value['raw_text']) > vlm.MAX_RAW_CHARS:
        raise vlm.ProtocolError('invalid_raw_text')
    if value.get('outcome') == 'ok':
        # Independently parse raw output; do not trust a mismatching parsed body.
        if vlm.parse_model_text(value['raw_text'], req['operation']) != value.get('parsed'):
            raise vlm.ProtocolError('parsed_does_not_match_raw')
    elif value.get('outcome') in ('parse_error', 'request_error', 'runtime_error'):
        if value.get('parsed') is not None or not value.get('errors'):
            raise vlm.ProtocolError('failed_response_must_not_supply_decisions')
    else:
        raise vlm.ProtocolError('invalid_response_outcome')
    return value


class StdioClient:
    """Deadline-bounded pipe I/O, no per-request model process/timer.

    External systemd supplies the hard process lifetime. close() only ensures
    child cleanup on normal exit/I/O failure. Tests inject a stdlib fake command;
    production CLI has no fake/model/command override.
    """
    def __init__(self, command, log_path, owner_guard, startup_timeout=180, response_timeout=120):
        self.process = None
        self.log = None
        self.buffer = bytearray()
        self.guard = owner_guard
        self.timeout = response_timeout
        for value in (startup_timeout, response_timeout):
            if not math.isfinite(value) or value <= 0:
                raise ValueError('positive_finite_timeout_required')
        self.guard(None)
        self.log = Path(log_path).open('wb')
        try:
            self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                            stderr=self.log, cwd=ROOT, env=os.environ.copy(), start_new_session=True)
            os.set_blocking(self.process.stdin.fileno(), False)
            os.set_blocking(self.process.stdout.fileno(), False)
            self.ready = check_ready(self._read(time.monotonic() + startup_timeout))
            self.guard(self.process.pid)
        except BaseException:
            self.close()
            raise

    def _read(self, deadline):
        fd = self.process.stdout.fileno()
        while True:
            if b'\n' in self.buffer:
                line, _, rest = self.buffer.partition(b'\n')
                self.buffer = bytearray(rest)
                if len(line) + 1 > vlm.MAX_LINE_BYTES:
                    raise vlm.ProtocolError('oversized_worker_line')
                return vlm.strict_json(line.decode('utf-8'))
            if len(self.buffer) > vlm.MAX_LINE_BYTES:
                raise vlm.ProtocolError('oversized_worker_line')
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([fd], [], [], remaining)[0]:
                raise TimeoutError('worker_stdio_response_deadline')
            block = os.read(fd, 65536)
            if not block:
                raise EOFError('worker_stdout_closed_before_complete_reply')
            self.buffer.extend(block)

    def request(self, request):
        vlm.decode_request(request)  # strict allowlist before crossing process boundary
        self.guard(self.process.pid)
        raw = (json.dumps(request, separators=(',', ':'), allow_nan=False) + '\n').encode()
        if len(raw) > vlm.MAX_LINE_BYTES:
            raise vlm.ProtocolError('request_too_large')
        deadline = time.monotonic() + self.timeout
        fd = self.process.stdin.fileno()
        remaining_bytes = memoryview(raw)
        while remaining_bytes:
            remaining_time = deadline - time.monotonic()
            if remaining_time <= 0 or not select.select([], [fd], [], remaining_time)[1]:
                raise TimeoutError('worker_stdio_write_deadline')
            sent = os.write(fd, remaining_bytes[:65536])
            if sent <= 0:
                raise BrokenPipeError('worker_stdin_closed')
            remaining_bytes = remaining_bytes[sent:]
        response = self._read(deadline)
        self.guard(self.process.pid)
        return check_response(response, request)

    def close(self):
        process = self.process
        if process is not None:
            if process.stdin and not process.stdin.closed:
                process.stdin.close()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait(timeout=5)
            if process.stdout:
                process.stdout.close()
        if self.log:
            self.log.close()


def saved_requests():
    """Only six already-saved TRAIN montages + their integrity metadata, no HDF."""
    from PIL import Image
    previous_protocol = json.loads((SOURCE / 'protocol.json').read_text())
    prepared = json.loads((SOURCE / 'prepared.json').read_text())
    if sha_file(SOURCE / 'protocol.json') != prepared['protocol_sha256']:
        raise vlm.ProtocolError('historical_probe_protocol_integrity')
    if previous_protocol['prompt'] != vlm.VERIFY_PROMPT:
        raise vlm.ProtocolError('must_use_identical_registered_CPU_prompt')
    cases = previous_protocol['cases']
    if tuple(c['case_id'] for c in cases) != CASE_IDS:
        raise vlm.ProtocolError('historical_six_TRAIN_cases_changed')
    result, ledger = [], []
    for case, old in zip(cases, prepared['cases']):
        cid = case['case_id']
        if old['case_id'] != cid:
            raise vlm.ProtocolError('historical_input_receipt_order')
        path = SOURCE / cid / 'native3rgb_montage.png'
        if sha_file(path) != old['montage_sha256']:
            raise vlm.ProtocolError('historical_montage_integrity')
        with Image.open(path) as image_file:
            image = image_file.convert('RGB')
        if image.size != (960, 264):
            raise vlm.ProtocolError('historical_montage_geometry')
        rgb = {}
        for col, view in enumerate(vlm.VIEWS):
            pixels = image.crop((col * 320, 24, (col + 1) * 320, 264)).tobytes()
            if digest := next((v['rgb_pixels_sha256'] for v in old['views'] if v['view'] == view), None):
                if vlm.digest(pixels) != digest:
                    raise vlm.ProtocolError('historical_native_pixels_integrity')
            else:
                raise vlm.ProtocolError('missing_native_pixel_receipt')
            rgb[view] = {'encoding': 'rgb8_base64', 'shape': list(vlm.SHAPE),
                         'data': base64.b64encode(pixels).decode('ascii')}
        req = {'version': 1, 'id': 'probe-' + cid, 'operation': 'verify',
               'current': {'episode_id': Path(case['episode']).stem,
                           'observation_id': case['frame_index'], 'rgb': rgb}}
        decoded = vlm.decode_request(req)
        reconstructed = vlm.montage(decoded['current'])
        ledger.append({'case_id': cid, 'source_montage': str(path), 'source_sha256': old['montage_sha256'],
                       'source_frame_index': case['frame_index'], 'native_pixels_unchanged': True,
                       'source_montage_pixel_sha256': vlm.digest(image.tobytes()),
                       'worker_montage_pixel_sha256': vlm.digest(reconstructed.tobytes()),
                       'all_montage_pixels_equal': image.tobytes() == reconstructed.tobytes(),
                       'binding': vlm.request_binding(decoded)})
        result.append((cid, req))
    return result, ledger


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-dir', type=Path, required=True)
    ap.add_argument('--gpu-lock-owner', required=True)
    ap.add_argument('--dtype', choices=('bf16', 'fp16'), default='bf16')
    ap.add_argument('--startup-timeout', type=float, default=180)
    ap.add_argument('--response-timeout', type=float, default=120)
    args = ap.parse_args(argv)
    if Path(sys.executable).resolve() != PYTHON.resolve():
        raise RuntimeError('use_existing_lilawam_environment_no_install_or_fallback')
    vlm.validate_owner(args.gpu_lock_owner)
    def guard(pid):
        vlm.check_gpu_owner(args.gpu_lock_owner, exclusive=True, allowed_pids=() if pid is None else (pid,))
    guard(None)  # No worker/GPU initialization unless an external lock is already held.
    run = args.run_dir.resolve()
    if run.exists() and any(run.iterdir()):
        raise RuntimeError('run_directory_must_be_new_or_empty_no_overwrite')
    run.mkdir(parents=True, exist_ok=True)
    requests, ledger = saved_requests()
    command = [str(PYTHON), '-B', '-m', 'harness.robotwin_vlm_worker',
               '--gpu-lock-owner', args.gpu_lock_owner, '--dtype', args.dtype]
    sources = {str(Path(__file__).resolve()): sha_file(__file__), str(Path(vlm.__file__).resolve()): sha_file(vlm.__file__)}
    protocol = {'registered_at': dt.datetime.now().astimezone().isoformat(), 'status': vlm.STATUS,
                'run_kind': 'exclusive_GPU_visual_reply_probe_only', 'model_id': vlm.MODEL_ID,
                'revision': vlm.REVISION, 'dtype': args.dtype, 'max_new_tokens': 192,
                'gpu_lock_owner': args.gpu_lock_owner, 'lock_acquired_or_released_here': False,
                'command': command, 'sources': sources, 'prompt': vlm.VERIFY_PROMPT,
                'prompt_sha256': vlm.digest(vlm.VERIFY_PROMPT.encode()), 'input_ledger': ledger,
                'response_timeout_s': args.response_timeout, 'startup_timeout_s': args.startup_timeout,
                'hard_process_timeout': 'provided externally by main systemd unit; not installed by this script',
                'claim': 'Reply/schema usability only; no state accuracy, task outcome or admission claim.',
                'source_data': 'Only existing TRAIN six montages/metadata; no datasets, GT or other splits.'}
    atomic(run / 'protocol.json', protocol)
    client = None
    records = []
    started = time.monotonic()
    result = {'status': vlm.STATUS, 'outcome': 'running', 'records': records, 'sources': sources}
    try:
        for cid, req in requests:
            (run / cid).mkdir()
            atomic(run / cid / 'request.json', req)
            (run / cid / 'prompt.txt').write_text(vlm.VERIFY_PROMPT)
        client = StdioClient(command, run / 'worker.stderr.log', guard, args.startup_timeout, args.response_timeout)
        atomic(run / 'ready.json', client.ready)
        atomic(run / 'process.json', {'worker_pid': client.process.pid, 'client_pid': os.getpid()})
        print(f'WORKER_PID={client.process.pid} RUN_DIR={run}', flush=True)
        for cid, req in requests:
            begin = time.monotonic()
            response = client.request(req)
            record = {'case_id': cid, 'roundtrip_s': time.monotonic() - begin, 'response': response}
            atomic(run / cid / 'response.json', record)
            (run / cid / 'raw_text.txt').write_text(response['raw_text'])
            records.append(record)
            atomic(run / 'result.json', result)
            if response['outcome'] in ('request_error', 'runtime_error'):
                raise RuntimeError('worker_reported_' + response['outcome'])
        result['outcome'] = 'completed' if all(r['response']['outcome'] == 'ok' for r in records) else 'completed_with_parse_errors'
    except Exception as exc:
        result.update(outcome='failed', error=type(exc).__name__ + ': ' + str(exc), traceback=traceback.format_exc())
    finally:
        if client:
            client.close()
            result['worker_exit_code'] = client.process.returncode
        result['total_s'] = time.monotonic() - started
        result['not_completed_case_ids'] = [cid for cid, _ in requests if cid not in {r['case_id'] for r in records}]
        atomic(run / 'result.json', result)
    return 0 if result['outcome'] == 'completed' else 2


if __name__ == '__main__':
    raise SystemExit(main())
