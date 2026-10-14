#!/usr/bin/env python3
"""One user-authorized <=180s GPU TRAIN follow-up; never edits existing worker.

--prepare registers 2 drawer questions + 3 ground instructions, image-only data.
--run acquires the project GPU lock only if immediately available, supervises one
isolated model process, releases only its own lock, and never retries a case.
Main invocation should additionally use timeout --kill-after=5s 180s.
"""
from __future__ import annotations
import argparse
import copy
import ctypes
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness import robotwin_vlm_worker as v

OUT = ROOT / 'outputs/validation/robotwin_vlm_train_followup_20261006'
OLD = ROOT / 'outputs/validation/robotwin_visible_probe_gpu_20261006_0420'
TRAIN = ROOT / 'outputs/validation/robotwin_visible_probe_train_20261006'
OWNER = 'p2-vlm-train-followup-20261006'
PYTHON = ROOT / '.conda-envs/lilawam/bin/python'
LIMIT = 180
WORKER_SHA = '2ffae6da0a0f09edf11a9c6457a77908f88c04e118d0aec3e65379cf1282073f'
SHORT_PROMPT = '''Is any drawer visibly open?
Inspect only these simultaneous RGB views: cam_head, cam_left_wrist, cam_right_wrist. Answer from visible geometry, not a task plan, a command, an assumed stage, or prior history.
true requires a drawer front visibly displaced from the cabinet with a visible opening, interior, or sidewall. A handle alone is insufficient. false requires clearly visible closed drawer fronts aligned with the cabinet and no visible opening; do not treat an obstructed or missing view as false. If occlusion or ambiguous geometry prevents either conclusion, answer unknown.
Return only JSON with exactly three fields: state (one of the strings "true", "false", "unknown"), views (the camera labels supporting your answer), and evidence (a concrete visible clue or obstruction, at most 20 words). Do not copy a default answer. Do not report task success, actions or plans.'''
GROUND_CASES = [
    ('ground_explicit', 'Run the whole put_object_cabinet task: put the object into the cabinet drawer.', 'execute'),
    ('ground_ambiguous', 'Put it away.', 'ask'),
    ('ground_unsupported', 'Drive to the kitchen and wash the dishes.', 'unsupported'),
]


def now(): return dt.datetime.now().astimezone().isoformat(timespec='milliseconds')
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def atomic(path, value):
    tmp = Path(path).with_suffix(Path(path).suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)
def append(path, value):
    with Path(path).open('a') as f:
        f.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')


def configure():
    os.environ.update({'CUDA_VISIBLE_DEVICES': '0', 'PYTHONDONTWRITEBYTECODE': '1',
                       'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2',
                       'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'TOKENIZERS_PARALLELISM': 'false'})
    sys.dont_write_bytecode = True


def load_native(case_name):
    path = OLD / case_name / 'request.json'
    request = json.loads(path.read_text())
    decoded = v.decode_request(request)
    assert decoded['previous'] is None
    image = v.montage(decoded['current'])
    response = json.loads((OLD / case_name / 'response.json').read_text())['response']
    if response['binding'] != v.request_binding(decoded) or response['montage_pixel_sha256'] != [v.digest(image.tobytes())]:
        raise ValueError('recorded_input_binding_mismatch')
    return request, image


def prepare():
    if OUT.exists():
        raise RuntimeError('new_output_directory_required_no_overwrite')
    OUT.mkdir(parents=True)
    if sha(v.__file__) != WORKER_SHA:
        raise RuntimeError('frozen_worker_changed_do_not_run')
    cases = [
        {'id': 'drawer_first', 'kind': 'drawer_short', 'source': 'episode_0000000_first',
         'human_visible_reference_not_model_input': 'false', 'reference_scope': 'visible closed front drawer; not hidden joint state'},
        {'id': 'drawer_last', 'kind': 'drawer_short', 'source': 'episode_0000000_last',
         'human_visible_reference_not_model_input': 'true', 'reference_scope': 'open cavity/sidewalls and extended front directly visible'},
    ] + [{'id': cid, 'kind': 'ground', 'source': 'episode_0000000_first', 'instruction': ins,
          'intent_reference_not_model_input': expected} for cid, ins, expected in GROUND_CASES]
    inputs = {}
    for name in ('episode_0000000_first', 'episode_0000000_last'):
        req, image = load_native(name)
        inputs[name] = {'request_file': str(OLD / name / 'request.json'),
                        'request_file_sha256': sha(OLD / name / 'request.json'),
                        'source_native_montage': str(TRAIN / name / 'native3rgb_montage.png'),
                        'binding': v.request_binding(v.decode_request(req)),
                        'montage_pixel_sha256': v.digest(image.tobytes()), 'shape': list(image.size)}
    protocol = {'schema': 'single_GPU_TRAIN_followup_v1', 'registered_at': now(), 'before_gpu_lock_or_model_load': True,
                'status': 'advisory_unadmitted', 'model': v.MODEL_ID, 'revision': v.REVISION,
                'dtype': 'bf16', 'device': 'cuda:0', 'torch_threads': 2, 'max_new_tokens': 192,
                'do_sample': False, 'owner': OWNER, 'hard_total_s': LIMIT, 'cleanup_reserve_s': 5,
                'acquire_policy': 'existing project gpu_lock acquire(max_wait=0); refuse occupied lock or >=4000MiB used; no stopping others',
                'resources': 'system RAM85% abort supervised; same worker before/after VRAM15300MiB; no training',
                'cases': cases, 'inputs': inputs, 'short_prompt': SHORT_PROMPT,
                'short_prompt_sha256': v.digest(SHORT_PROMPT.encode()), 'ground_prompt_unchanged': v.GROUND_PROMPT,
                'original_verify_prompt_sha256': v.digest(v.VERIFY_PROMPT.encode()),
                'source_hashes': {str(Path(__file__).resolve()): sha(__file__), str(Path(v.__file__).resolve()): sha(v.__file__),
                                  str(ROOT / 'scripts/gpu_lock.py'): sha(ROOT / 'scripts/gpu_lock.py')},
                'read_scope': 'Only saved TRAIN episode0 first/last RGB requests/montages and local fixed model; no HDF, sim GT, dev/VAL/heldout',
                'no_model_inputs': ['case/episode/frame IDs', 'manual reference labels', 'GT', 'gripper commands', 'other case replies'],
                'instrumentation': 'Isolated instance generate wrapper saves ACTUAL input token ids/counts, image grids/pixel tensor hashes immediately before original generate. Original full prompt processor-only reference; no extra model inference.',
                'comparison_limits': 'Two TRAIN images only. Shortening/isolating drawer question and removing filled unknown example change multiple prompt factors; cannot attribute causality solely to schema anchoring.',
                'frozen_files_never_modified': ['existing worker', 'original prompts', 'original probe results', 'progress.md']}
    atomic(OUT / 'protocol.json', protocol)
    for case in cases:
        directory = OUT / case['id']; directory.mkdir()
        prompt = SHORT_PROMPT if case['kind'] == 'drawer_short' else v.GROUND_PROMPT + json.dumps(case['instruction'], ensure_ascii=False)
        (directory / 'prompt.txt').write_text(prompt)
    atomic(OUT / 'prepared.json', {'protocol_sha256': sha(OUT / 'protocol.json'), 'model_loaded': False, 'gpu_called': False})
    print('PREPARED ' + str(OUT / 'protocol.json'), flush=True)


def parse_short(raw):
    text = raw.strip()
    if text.startswith('```json\n') and text.endswith('\n```'): text = text[8:-4].strip()
    elif text.startswith('```\n') and text.endswith('\n```'): text = text[4:-4].strip()
    d = v.strict_json(text)
    v.keys(d, ('state', 'views', 'evidence'))
    if type(d['state']) is not str or d['state'] not in ('true', 'false', 'unknown'):
        raise v.ProtocolError('invalid_state')
    if type(d['views']) is not list or not d['views'] or any(type(x) is not str or x not in v.VIEWS for x in d['views']):
        raise v.ProtocolError('invalid_views')
    if type(d['evidence']) is not str or not d['evidence'].strip(): raise v.ProtocolError('missing_evidence')
    return d


def tensor_audit(inputs, backend):
    torch = backend.torch
    cfg = backend.model.config
    token_ids = inputs['input_ids'].detach().cpu().reshape(-1).tolist()
    grid = inputs['image_grid_thw'].detach().cpu().tolist()
    merge = backend.processor.image_processor.merge_size
    expected = sum(t * h * w // (merge * merge) for t, h, w in grid)
    pixel = inputs['pixel_values'].detach().float().cpu().contiguous()
    result = {'input_token_count': len(token_ids), 'image_token_id': cfg.image_token_id,
              'actual_image_token_count': token_ids.count(cfg.image_token_id), 'expected_merged_image_tokens': expected,
              'vision_start_count': token_ids.count(cfg.vision_start_token_id),
              'vision_end_count': token_ids.count(cfg.vision_end_token_id), 'image_grid_thw': grid,
              'merge_size': merge, 'pixel_shape': list(pixel.shape), 'pixel_input_dtype': str(inputs['pixel_values'].dtype),
              'pixel_tensor_sha256_float32': v.digest(pixel.numpy().tobytes()),
              'pixel_min': pixel.min().item(), 'pixel_max': pixel.max().item(), 'pixel_std': pixel.std().item(),
              'input_token_ids': token_ids, 'input_shapes': {k: list(x.shape) for k, x in inputs.items()}}
    assert result['actual_image_token_count'] == expected > 0
    assert result['vision_start_count'] == result['vision_end_count'] == len(grid)
    assert pixel.shape[0] == sum(t * h * w for t, h, w in grid)
    assert bool(torch.isfinite(pixel).all()) and result['pixel_std'] > 0
    return result


def child():
    parent = os.getppid()
    if ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
        raise RuntimeError('parent_death_guard_failed')
    if parent == 1 or os.getppid() != parent: raise RuntimeError('supervisor_gone')
    configure()
    protocol = json.loads((OUT / 'protocol.json').read_text())
    atomic(OUT / 'phase.json', {'phase': 'loading', 'at': now(), 'pid': os.getpid()})
    backend = v.GpuBackend(OWNER, 'bf16')
    atomic(OUT / 'ready.json', v.ready_message(backend))
    audit_target = {'case_id': None}
    original_generate = backend.model.generate
    def audited_generate(**kwargs):
        case_id = audit_target['case_id']
        tensors = {key: kwargs[key] for key in ('input_ids', 'attention_mask', 'pixel_values', 'image_grid_thw')}
        details = tensor_audit(tensors, backend)
        details.update(at=now(), captured_from='actual_generate_kwargs', max_new_tokens=kwargs.get('max_new_tokens'), do_sample=kwargs.get('do_sample'))
        assert details['max_new_tokens'] == 192 and details['do_sample'] is False
        atomic(OUT / case_id / 'actual_input_audit.json', details)
        return original_generate(**kwargs)
    backend.model.generate = audited_generate  # isolated new object only; no existing process/source mutation
    images, requests = {}, {}
    for source in protocol['inputs']:
        requests[source], images[source] = load_native(source)
        # Processor-only reconstruction of the OLD prompt. Does not call model.generate.
        image = images[source]
        messages = [{'role': 'user', 'content': [{'type': 'image', 'image': image}, {'type': 'text', 'text': v.VERIFY_PROMPT}]}]
        chat = backend.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        old_inputs = backend.processor(text=[chat], images=[image], return_tensors='pt')
        check = tensor_audit(old_inputs, backend)
        check.update(source=source, scope='old_prompt_processor_only_no_inference', prompt_sha256=v.digest(v.VERIFY_PROMPT.encode()))
        atomic(OUT / (source + '_old_input_audit.json'), check)
    service = v.Service(backend, backend.guard)
    results = []
    for case in protocol['cases']:
        cid = case['id']; directory = OUT / cid; audit_target['case_id'] = cid
        atomic(OUT / 'phase.json', {'phase': 'inference', 'case_id': cid, 'at': now(), 'pid': os.getpid()})
        started = time.monotonic()
        record = {'case_id': cid, 'kind': case['kind'], 'status': 'advisory_unadmitted', 'errors': [], 'parsed': None}
        try:
            if case['kind'] == 'drawer_short':
                backend.guard()
                raw, generation = backend.complete(SHORT_PROMPT, [images[case['source']]])
                record.update(raw_text=raw, generation=generation)
                (directory / 'raw_text.txt').write_text(raw)
                try:
                    record['parsed'] = parse_short(raw)
                    record['outcome'] = 'ok'
                except Exception as exc:
                    record['outcome'] = 'parse_error'; record['errors'].append(str(exc))
                record['human_reference_not_model_input'] = case['human_visible_reference_not_model_input']
            else:
                req = copy.deepcopy(requests[case['source']])
                req.update(id=cid, operation='ground', instruction=case['instruction'])
                atomic(directory / 'request.json', req)
                response = service.handle(req)
                record.update(response=response, raw_text=response['raw_text'], parsed=response['parsed'],
                              outcome=response['outcome'], errors=response['errors'], generation=response.get('generation'))
                (directory / 'raw_text.txt').write_text(response['raw_text'])
                record['intent_reference_not_model_input'] = case['intent_reference_not_model_input']
            record.update(elapsed_s=time.monotonic() - started, resources=backend.resources())
            backend.guard()
        except Exception as exc:
            record.update(outcome='runtime_error', elapsed_s=time.monotonic() - started, traceback=traceback.format_exc())
            record['errors'].append(str(exc))
        atomic(directory / 'result.json', record)
        results.append(record)
        atomic(OUT / 'child_result.json', {'status': 'advisory_unadmitted', 'records': results, 'completed': len(results) == 5})
        print(f'CASE_DONE {cid} {record["outcome"]} {record["elapsed_s"]:.3f}s', flush=True)
        if record['outcome'] == 'runtime_error': return 2
    return 0


def run():
    if Path(sys.executable).resolve() != PYTHON.resolve(): raise RuntimeError('existing_lilawam_required')
    if (OUT / 'supervisor.json').exists(): raise RuntimeError('no_automatic_retry')
    protocol = json.loads((OUT / 'protocol.json').read_text())
    if sha(OUT / 'protocol.json') != json.loads((OUT / 'prepared.json').read_text())['protocol_sha256']:
        raise RuntimeError('protocol_changed')
    for path, expected in protocol['source_hashes'].items():
        if sha(path) != expected: raise RuntimeError('frozen_source_changed')
    from scripts import gpu_lock
    start = time.monotonic(); acquired = False; process = None; reason = None; cancelled = [None]
    result = {'status': 'advisory_unadmitted', 'supervisor_pid': os.getpid(), 'started_at': now(), 'outcome': 'running'}
    atomic(OUT / 'supervisor.json', result)
    signal.signal(signal.SIGTERM, lambda a, b: cancelled.__setitem__(0, 'SIGTERM'))
    signal.signal(signal.SIGINT, lambda a, b: cancelled.__setitem__(0, 'SIGINT'))
    try:
        wait = gpu_lock.acquire(OWNER, max_wait=0)
        acquired = True
        v.check_gpu_owner(OWNER, exclusive=True)
        result.update(lock_owner=OWNER, lock_acquired_at=now(), lock_wait_s=wait)
        atomic(OUT / 'supervisor.json', result)
        with (OUT / 'child.log').open('wb') as log:
            configure()
            process = subprocess.Popen([str(PYTHON), '-B', str(Path(__file__).resolve()), '--child'],
                                       cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            result['child_pid'] = process.pid
            atomic(OUT / 'supervisor.json', result)
            print(f'SUPERVISOR_PID={os.getpid()} CHILD_PID={process.pid} OWNER={OWNER} OUTPUT={OUT}', flush=True)
            while process.poll() is None:
                snapshot = v.cpu_resources()
                append(OUT / 'resources.jsonl', {'at': now(), 'elapsed_s': time.monotonic()-start, **snapshot})
                if snapshot['system_ram_percent'] >= 85: reason = 'system_ram_85_percent'
                if time.monotonic() - start >= LIMIT - 5: reason = reason or 'hard_budget_cleanup'
                reason = reason or cancelled[0]
                if reason:
                    os.killpg(process.pid, signal.SIGKILL); process.wait(timeout=3); break
                time.sleep(0.2)
            result['child_exit_code'] = process.wait(timeout=3)
        result['outcome'] = 'completed' if result['child_exit_code'] == 0 else 'failed'
    except Exception as exc:
        result.update(outcome='failed', error=type(exc).__name__ + ': ' + str(exc), traceback=traceback.format_exc())
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL); process.wait(timeout=3)
        result['lock_released'] = gpu_lock.release(OWNER) if acquired else False
        result.update(finished_at=now(), total_s=time.monotonic()-start, stop_reason=reason)
        if (OUT / 'child_result.json').exists(): result['child_result'] = json.loads((OUT / 'child_result.json').read_text())
        result['not_completed_cases'] = [c['id'] for c in protocol['cases'] if not (OUT / c['id'] / 'result.json').exists()]
        atomic(OUT / 'result.json', result); atomic(OUT / 'supervisor.json', result)
        print(json.dumps({k:result[k] for k in ('outcome','total_s','lock_released','not_completed_cases')}), flush=True)
    return 0 if result['outcome'] == 'completed' else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare', action='store_true'); group.add_argument('--run', action='store_true'); group.add_argument('--child', action='store_true')
    args = parser.parse_args()
    if args.prepare: prepare(); return 0
    if args.run: return run()
    return child()


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception:
        if OUT.exists(): atomic(OUT / f'failure_{os.getpid()}.json', {'at': now(), 'traceback': traceback.format_exc()})
        raise
