#!/usr/bin/env python3
"""One frozen TRAIN-only, CPU-only Qwen2.5-VL-3B visibility probe.

--prepare writes protocol + six image-only inputs; --run supervises six fresh
children. No simulator, action/state reads, model fallback, installs or downloads.
Results are usability observations, NEVER physical-verifier admission.
"""
from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import hashlib
import importlib.metadata
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
OUT = ROOT / 'outputs/validation/robotwin_visible_probe_train_20261006'
MANIFEST = ROOT / 'data/benchmarks/robotwin2/put_object_cabinet_500_v1/split_manifest.json'
MANIFEST_SHA = '5feb138a226241e2e31ac83fa142a82dc3645ad2022a644fefe8a25650a5b339'
MODEL = Path('/home/user/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3')
VIEWS = ('cam_head', 'cam_left_wrist', 'cam_right_wrist')
PREDICATES = ('drawer_visibly_open', 'object_visibly_held', 'visible_release')
INFERENCE_PYTHON = ROOT / '.conda-envs/lilawam/bin/python'
CASE_LIMIT = 120.0
TOTAL_LIMIT = 900.0
ABORT_PERCENT = 85.0
MAX_TOKENS = 256
PROMPT = '''You are a cautious visual observer, not a robot planner. The attached image contains three simultaneous native RGB camera views. The labels above the tiles are cam_head, cam_left_wrist, cam_right_wrist. All tiles show the SAME timestamp, NOT a sequence.
Use ONLY visible pixels in these three views. No task plan, expected stage, gripper commands, hidden state, earlier/later images, or assumed actions are evidence. Multiple objects may be present: name the particular visible object in your evidence; do not assume the intended target. Occluded, out-of-view, ambiguous or insufficient evidence MUST be unknown. False requires a clearly visible contradictory state, not absence of evidence.
Judge:
- drawer_visibly_open: a drawer is visibly displaced from its cabinet with a visible opening/interior. A handle or reaching hand alone is insufficient.
- object_visibly_held: a particular object is visibly retained between identifiable gripper fingers. Nearby fingers, overlap, or a resting object alone is insufficient.
- visible_release: release/separation of that object from its gripper is directly visually established. Open fingers or an object resting somewhere alone do not prove release. A single timestamp cannot establish a prior holding/release transition; use unknown if that history or object-gripper correspondence is needed but unobserved.
For EACH item return state true, false or unknown as a JSON STRING; cite at least one exact camera label and a concrete visible clue (or obstruction). Each evidence string must be at most 15 words. Do not invent coordinates, earlier events, identities or commands. No actions, plans, confidence scores or success judgments.
Return ONLY one JSON object with exactly these three keys: drawer_visibly_open, object_visibly_held, visible_release. Each value must have exactly: {"state":"unknown","views":["cam_head"],"evidence":"specific visible clue or obstruction"}. The example is a schema, not a label for these images.'''


def now():
    return dt.datetime.now().astimezone().isoformat(timespec='milliseconds')


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)


def append(path, value):
    line = (json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n').encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        if os.write(fd, line) != len(line):
            raise OSError('short ledger write')
    finally:
        os.close(fd)


def ledger(kind, **fields):
    append(OUT / 'read_ledger.jsonl', {'at': now(), 'pid': os.getpid(), 'kind': kind, **fields})


def identity(path, content=False):
    p = Path(path)
    s = p.stat()
    result = {'path': str(p), 'resolved_path': str(p.resolve()), 'size': s.st_size,
              'mtime_ns': s.st_mtime_ns, 'inode': s.st_ino, 'device': s.st_dev}
    if content:
        result['sha256'] = sha_file(p)
    return result


def setup_environment():
    # Override inherited settings; do NOT call any GPU discovery/initialization API.
    env = {'CUDA_VISIBLE_DEVICES': '', 'NVIDIA_VISIBLE_DEVICES': 'none',
           'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
           'HF_DATASETS_OFFLINE': '1', 'HF_HUB_DISABLE_TELEMETRY': '1',
           'PYTHONDONTWRITEBYTECODE': '1', 'TOKENIZERS_PARALLELISM': 'false',
           'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2',
           'OPENBLAS_NUM_THREADS': '2', 'NUMEXPR_NUM_THREADS': '2',
           'HF_HOME': str(OUT / 'runtime_cache/hf'),
           'XDG_CACHE_HOME': str(OUT / 'runtime_cache'),
           'TORCH_HOME': str(OUT / 'runtime_cache/torch'),
           'TMPDIR': str(OUT / 'tmp')}
    os.environ.update(env)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(ROOT))
    affinity = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, affinity[-2:])
    # Offline means no network attempts, even if a library ignores HF flags.
    def offline_audit(event, args):
        if event in ('socket.connect', 'socket.getaddrinfo'):
            raise RuntimeError('network_disabled_for_cpu_visible_probe')
    sys.addaudithook(offline_audit)
    return env


def mem_snapshot(pid=None):
    mem = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key, value = line.split(':', 1)
        mem[key] = int(value.strip().split()[0]) * 1024
    result = {'at': now(), 'monotonic': time.monotonic(),
              'system_ram_percent': 100 * (1 - mem['MemAvailable'] / mem['MemTotal']),
              'mem_available_bytes': mem['MemAvailable'], 'mem_total_bytes': mem['MemTotal']}
    if pid is not None:
        try:
            for line in Path(f'/proc/{pid}/status').read_text().splitlines():
                key, _, value = line.partition(':')
                if key in ('VmRSS', 'VmHWM', 'VmSwap', 'VmSize'):
                    result[key + '_bytes'] = int(value.strip().split()[0]) * 1024
                elif key == 'Threads':
                    result['threads'] = int(value.strip())
        except (FileNotFoundError, ProcessLookupError):
            result['child_exited_during_sample'] = True
    return result


class Guard:
    def __init__(self):
        self.previous = None
        self.cpu = None

    def sample(self, pid=None):
        result = mem_snapshot(pid)
        # Global CPU over >=1 second, not a noisy short-window estimate.
        counters = list(map(int, Path('/proc/stat').read_text().splitlines()[0].split()[1:9]))
        current = (result['monotonic'], sum(counters), counters[3] + counters[4])
        if self.previous is None:
            self.previous = current
        elif current[0] - self.previous[0] >= 1.0:
            delta = current[1] - self.previous[1]
            self.cpu = 100 * (1 - (current[2] - self.previous[2]) / delta) if delta > 0 else None
            result['cpu_sample_window_s'] = current[0] - self.previous[0]
            self.previous = current
        result['system_cpu_percent'] = self.cpu
        return result

    @staticmethod
    def reason(sample):
        if sample['system_ram_percent'] >= ABORT_PERCENT:
            return 'system_ram_at_or_above_85_percent'
        if sample.get('system_cpu_percent') is not None and sample['system_cpu_percent'] >= ABORT_PERCENT:
            return 'system_cpu_at_or_above_85_percent'
        return None


def parse_reply(raw):
    text = raw.strip()
    stripped_fence = False
    if text.startswith('```json\n') and text.endswith('\n```'):
        text, stripped_fence = text[8:-4].strip(), True
    elif text.startswith('```\n') and text.endswith('\n```'):
        text, stripped_fence = text[4:-4].strip(), True
    def no_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate JSON key: ' + key)
            result[key] = value
        return result
    parsed = json.loads(text, object_pairs_hook=no_duplicates,
                        parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    if not isinstance(parsed, dict) or set(parsed) != set(PREDICATES):
        raise ValueError('expected exactly the three predicate keys')
    for name, item in parsed.items():
        if not isinstance(item, dict) or set(item) != {'state', 'views', 'evidence'}:
            raise ValueError(name + ': invalid fields')
        if item['state'] not in ('true', 'false', 'unknown') or type(item['state']) is not str:
            raise ValueError(name + ': state must be a three-valued string')
        if (not isinstance(item['views'], list) or not item['views']
                or any(type(v) is not str or v not in VIEWS for v in item['views'])
                or len(set(item['views'])) != len(item['views'])):
            raise ValueError(name + ': invalid/missing camera evidence')
        if not isinstance(item['evidence'], str) or not item['evidence'].strip():
            raise ValueError(name + ': missing visible clue')
    return parsed, {'outer_json_fence_removed': stripped_fence,
                    'semantic_accuracy_evaluated': False, 'gt_used': False,
                    'evidence_word_count': {k: len(v['evidence'].split()) for k, v in parsed.items()}}


def prepare():
    if (OUT / 'protocol.json').exists():
        raise RuntimeError('protocol already exists; refusing overwrite/reselection')
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'tmp').mkdir(exist_ok=True)
    env = setup_environment()
    import h5py
    import numpy as np
    import cv2
    from PIL import Image, ImageDraw
    from harness.robotwin_dataset import decode_rgb
    cv2.setNumThreads(2)
    guard = Guard()
    sample = guard.sample()
    if guard.reason(sample):
        raise RuntimeError(guard.reason(sample))
    if sha_file(MANIFEST) != MANIFEST_SHA:
        raise RuntimeError('frozen split manifest SHA mismatch')
    manifest = json.loads(MANIFEST.read_text())
    ledger('split_manifest_metadata_only', **identity(MANIFEST, True),
           note='Only TRAIN payloads opened; other split names are manifest metadata only.')
    selected = sorted(manifest['splits']['train'])[:2]
    if len(selected) != 2 or len(set(selected)) != 2:
        raise RuntimeError('expected two unique TRAIN episodes')
    if any(x in manifest['splits']['val'] or x in manifest['splits']['heldout'] for x in selected):
        raise RuntimeError('split overlap')
    root = Path(manifest['data_root']).resolve()
    cases, data_ids = [], []
    for episode_name in selected:
        path = root / episode_name
        if path.is_symlink() or path.resolve().parent != root or path.name != episode_name:
            raise RuntimeError('episode path escape/alias')
        data_ids.append(identity(path))
        with h5py.File(path, 'r') as hdf:
            sizes = [hdf[f'vision/{v}/colors'].shape[0] for v in VIEWS]
            ledger('train_hdf_rgb_shape_metadata', file=str(path),
                   datasets=[f'vision/{v}/colors' for v in VIEWS], lengths=sizes)
        if len(set(sizes)) != 1 or sizes[0] < 3:
            raise RuntimeError('invalid/unsynchronized RGB frame counts')
        for label, frame in (('first', 0), ('middle', sizes[0] // 2), ('last', sizes[0] - 1)):
            cases.append({'case_id': f'{Path(episode_name).stem}_{label}', 'episode': episode_name,
                          'path': str(path), 'frame_index': frame, 'rgb_frame_count': sizes[0],
                          'selection_label_not_sent_to_model': label})
    sources = [Path(__file__).resolve(), ROOT / 'harness/robotwin_dataset.py',
               ROOT / 'third_party/RoboTwin/data/decode_image_bit.py', ROOT / 'harness/grounding.py']
    source_ids = [identity(p, True) for p in sources]
    for item in source_ids:
        ledger('source_hash', **item)
    model_index = MODEL / 'model.safetensors.index.json'
    weights = sorted(set(json.loads(model_index.read_text())['weight_map'].values()))
    model_names = weights + ['model.safetensors.index.json', 'config.json', 'generation_config.json',
                            'preprocessor_config.json', 'tokenizer_config.json', 'tokenizer.json',
                            'chat_template.json', 'vocab.json', 'merges.txt']
    model_ids = [identity(MODEL / name, not name.endswith('.safetensors')) for name in model_names]
    for item in model_ids:
        ledger('local_model_inventory', **item,
               note='Weights stat only; small config/tokenizer files hashed. No model tensors loaded.')
    versions = {}
    for name in ('torch', 'transformers', 'accelerate', 'h5py', 'numpy', 'Pillow', 'opencv-python'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    protocol = {'schema': 'robotwin-visible-cpu-probe-v1', 'registered_at': now(),
                'registered_before_model_load': True, 'claim': 'visual usability probe; NOT admission',
                'admitted': False, 'model': str(MODEL), 'model_id': 'Qwen/Qwen2.5-VL-3B-Instruct',
                'revision': MODEL.name, 'python': str(INFERENCE_PYTHON), 'data_preparation_python': sys.executable,
                'data_preparation_packages_metadata': versions,
                'preparation_environment_note': 'Initial preparation in lilawam failed: h5py absent, preserved in bootstrap_failure_*.json. Existing behavior environment decodes the same native RGB via the unchanged reader; inference remains in the historical lilawam Qwen environment. No install/model/interface fallback.',
                'device_map': 'cpu', 'dtype': 'bfloat16', 'quantization': None,
                'torch_threads': 2, 'torch_interop_threads': 1,
                'cpu_affinity': sorted(os.sched_getaffinity(0)), 'environment': env,
                'max_new_tokens': MAX_TOKENS, 'do_sample': False, 'seed': 0,
                'case_hard_limit_s': CASE_LIMIT, 'total_hard_limit_s': TOTAL_LIMIT,
                'supervisor_cleanup_reserve_s': 10, 'monitor_poll_s': 0.2,
                'system_ram_abort_percent': ABORT_PERCENT, 'system_cpu_abort_percent': ABORT_PERCENT,
                'resource_at_registration': sample,
                'memory_budget_note': 'User estimate 6GB is not a limit. Local BF16 weight files total about 6.994GiB; historical CPU RSS about 8GiB. Abort using system MemAvailable-based 85% gate.',
                'selection': 'lexicographically first two TRAIN names; RGB frame indices 0, T//2, T-1',
                'manifest': identity(MANIFEST, True), 'selected_train': selected, 'data_identities': data_ids,
                'cases': cases, 'source_identities': source_ids, 'model_identities': model_ids,
                'prompt': PROMPT, 'prompt_sha256': sha_bytes(PROMPT.encode()),
                'model_inputs_only': ['same_timestamp_three_native_RGB_montage', 'identical_prompt'],
                'never_model_inputs': ['state', 'action', 'gripper_command', 'instructions', 'GT',
                                      'seed', 'episode_name', 'frame_index', 'first_middle_last',
                                      'static_plan', 'third_view', 'other_case_images_or_replies'],
                'read_scope': 'Only selected TRAIN vision/{cam_head,cam_left_wrist,cam_right_wrist}/colors rows; no lowdim/actions/GT or VAL/heldout/dev payloads.',
                'release_semantics': 'Unknown if single timestamp cannot establish holding-release transition or object-gripper correspondence; no imposed ground-truth label.',
                'parsing': 'Strict JSON exact schema, optionally remove only one outer code fence; no repair/requery/GT correction.',
                'failure_policy': 'One child per case. SIGKILL process group at 120s. System 85% aborts entire run. Interface/load failure stops remaining cases without fallback/install/model change. Timeouts retained, no retries.',
                'external_total_guard': 'timeout --signal=TERM --kill-after=5s 900s <same python> -B <script> --run'}
    atomic(OUT / 'protocol.json', protocol)
    # Protocol precedes image preparation and any model inference.
    image_receipts = []
    for case in cases:
        sample = guard.sample()
        if guard.reason(sample):
            raise RuntimeError(guard.reason(sample))
        directory = OUT / case['case_id']
        directory.mkdir(exist_ok=False)
        (directory / 'prompt.txt').write_text(PROMPT)
        canvas = Image.new('RGB', (320 * 3, 264), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        pixels = []
        with h5py.File(case['path'], 'r') as hdf:
            for col, view in enumerate(VIEWS):
                key = f'vision/{view}/colors'
                encoded = hdf[key][case['frame_index']]
                image = decode_rgb(encoded)
                if image.shape != (240, 320, 3):
                    raise RuntimeError('unexpected native RGB geometry')
                Image.fromarray(image).save(directory / f'{view}.png')
                canvas.paste(Image.fromarray(image), (col * 320, 24))
                draw.text((col * 320 + 8, 6), view, fill=(0, 0, 0))
                raw = encoded if isinstance(encoded, bytes) else np.asarray(encoded).tobytes()
                item = {'view': view, 'shape': list(image.shape), 'encoded_payload_sha256': sha_bytes(raw),
                        'rgb_pixels_sha256': sha_bytes(image.tobytes()),
                        'native_png_sha256': sha_file(directory / f'{view}.png')}
                pixels.append(item)
                ledger('train_rgb_row_read', file=case['path'], dataset=key,
                       frame_index=case['frame_index'], case_id=case['case_id'], **item)
        canvas.save(directory / 'native3rgb_montage.png')
        receipt = {**case, 'views': pixels, 'montage_size': list(canvas.size),
                   'montage_sha256': sha_file(directory / 'native3rgb_montage.png'),
                   'prompt_sha256': sha_file(directory / 'prompt.txt'),
                   'resized_native_tiles': False, 'labels_only_outside_native_pixels': True}
        atomic(directory / 'input.json', receipt)
        image_receipts.append(receipt)
    atomic(OUT / 'prepared.json', {'prepared_at': now(), 'protocol_sha256': sha_file(OUT / 'protocol.json'),
                                 'cases': image_receipts, 'model_loaded': False, 'gpu_used': False})
    print(json.dumps({'status': 'prepared', 'protocol': str(OUT / 'protocol.json'), 'cases': cases,
                      'packages': versions, 'RAM': sample}, ensure_ascii=False), flush=True)


def case_worker(case_id):
    # A supervisor exit cannot strand a model process; do this before ML imports.
    parent = os.getppid()
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), 'PR_SET_PDEATHSIG failed')
    if os.getppid() != parent or parent == 1:
        raise RuntimeError('supervisor vanished')
    setup_environment()
    directory = OUT / case_id
    protocol = json.loads((OUT / 'protocol.json').read_text())
    if case_id not in [c['case_id'] for c in protocol['cases']]:
        raise RuntimeError('case not in registered protocol')
    receipt = json.loads((directory / 'input.json').read_text())
    prompt = (directory / 'prompt.txt').read_text()
    if (sha_bytes(prompt.encode()) != protocol['prompt_sha256']
            or sha_file(directory / 'native3rgb_montage.png') != receipt['montage_sha256']):
        raise RuntimeError('input changed after registration')
    result = {'case_id': case_id, 'pid': os.getpid(), 'started_at': now(), 'admitted': False,
              'claim': 'visual usability only; no semantic correctness/admission evaluation',
              'errors': [], 'parsed': None, 'raw_reply': '', 'gt_used': False}
    started = time.monotonic()
    phase = 'imports'
    def mark(name, **more):
        nonlocal phase
        phase = name
        atomic(directory / 'phase.json', {'phase': name, 'at': now(), 'pid': os.getpid(),
                                         'elapsed_s': time.monotonic() - started, **more})
    try:
        mark('imports')
        import torch
        from PIL import Image
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, TextStreamer
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        torch.manual_seed(0)
        if torch.cuda.is_initialized():
            raise RuntimeError('CUDA must remain uninitialized')
        result['runtime'] = {'torch': torch.__version__, 'transformers': importlib.metadata.version('transformers'),
                             'python': sys.version, 'torch_threads': torch.get_num_threads(),
                             'torch_interop_threads': torch.get_num_interop_threads(),
                             'cpu_affinity': sorted(os.sched_getaffinity(0)), 'cuda_initialized': False}
        mark('model_load')
        for item in protocol['model_identities']:
            ledger('model_loader_allowed_file', case_id=case_id, **item,
                   note='Declared local loader file set; native safetensors mmap reads are not syscall traced.')
        load_start = time.monotonic()
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(MODEL), device_map='cpu', dtype=torch.bfloat16, local_files_only=True,
            trust_remote_code=False)
        model.eval()
        processor = AutoProcessor.from_pretrained(str(MODEL), local_files_only=True, trust_remote_code=False)
        result['model_and_processor_load_s'] = time.monotonic() - load_start
        devices = sorted({str(p.device) for p in model.parameters()})
        dtypes = sorted({str(p.dtype) for p in model.parameters()})
        if devices != ['cpu'] or dtypes != ['torch.bfloat16'] or torch.cuda.is_initialized():
            raise RuntimeError(f'CPU bf16 contract violated: {devices}, {dtypes}')
        result['runtime'].update(parameter_devices=devices, parameter_dtypes=dtypes,
                                 model_class=type(model).__name__, processor_class=type(processor).__name__)
        mark('processor')
        image = Image.open(directory / 'native3rgb_montage.png').convert('RGB')
        ledger('model_image_input', case_id=case_id, path=str(directory / 'native3rgb_montage.png'),
               sha256=receipt['montage_sha256'], only_native_rgb_plus_camera_labels=True)
        msgs = [{'role': 'user', 'content': [{'type': 'image', 'image': image}, {'type': 'text', 'text': prompt}]}]
        chat = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        (directory / 'applied_chat_template.txt').write_text(chat)
        inputs = processor(text=[chat], images=[image], return_tensors='pt').to('cpu')
        result['model_input_shapes'] = {k: list(v.shape) for k, v in inputs.items()}
        result['image_grid_thw'] = inputs['image_grid_thw'].tolist()
        raw_path = directory / 'raw_reply.txt'
        raw_path.write_text('')
        class DiskStreamer(TextStreamer):
            def on_finalized_text(self, text, stream_end=False):
                with raw_path.open('a') as stream:
                    stream.write(text)
                    stream.flush()
        streamer = DiskStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)
        mark('generation', model_and_processor_load_s=result['model_and_processor_load_s'],
             runtime=result['runtime'], model_input_shapes=result['model_input_shapes'],
             image_grid_thw=result['image_grid_thw'])
        generate_start = time.monotonic()
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=MAX_TOKENS, do_sample=False, streamer=streamer)
        result['generation_s'] = time.monotonic() - generate_start
        new_ids = generated[:, inputs['input_ids'].shape[1]:]
        raw = processor.batch_decode(new_ids, skip_special_tokens=True)[0]
        raw_path.write_text(raw)
        result.update(raw_reply=raw, reply_complete=True, new_tokens=int(new_ids.shape[1]),
                      reached_token_cap=new_ids.shape[1] >= MAX_TOKENS)
        mark('parse')
        try:
            result['parsed'], result['parse_details'] = parse_reply(raw)
            result['status'] = 'completed_schema_valid_not_admitted'
        except (ValueError, TypeError) as exc:
            result['status'] = 'completed_parse_error'
            result['errors'].append({'stage': 'parse', 'type': type(exc).__name__, 'message': str(exc)})
        result['cuda_initialized_at_end'] = torch.cuda.is_initialized()
    except BaseException as exc:
        result.update(status='interface_or_runtime_failure', failure_stage=phase)
        result['errors'].append({'stage': phase, 'type': type(exc).__name__, 'message': str(exc),
                                 'traceback': traceback.format_exc()})
        (directory / 'error_traceback.txt').write_text(traceback.format_exc())
    finally:
        result.update(finished_at=now(), child_total_s=time.monotonic() - started,
                      max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      resource_at_end=mem_snapshot(os.getpid()))
        atomic(directory / 'result.json', result)
    return 0 if result['status'].startswith('completed_') else 2


def terminate_group(child):
    try:
        os.killpg(child.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    child.wait(timeout=5)


def run():
    setup_environment()
    if Path(sys.executable).resolve() != INFERENCE_PYTHON.resolve():
        raise RuntimeError('must use registered historical lilawam inference environment')
    if (OUT / 'summary.json').exists() or (OUT / 'supervisor.json').exists():
        raise RuntimeError('run already started; no automatic overwrite/retry')
    prepared = json.loads((OUT / 'prepared.json').read_text())
    if sha_file(OUT / 'protocol.json') != prepared['protocol_sha256']:
        raise RuntimeError('protocol changed after preparation')
    protocol = json.loads((OUT / 'protocol.json').read_text())
    for item in protocol['source_identities']:
        if sha_file(item['path']) != item['sha256']:
            raise RuntimeError('source changed after protocol: ' + item['path'])
    for item in protocol['model_identities']:
        fresh = identity(item['path'], 'sha256' in item)
        if fresh != item:
            raise RuntimeError('model identity changed after protocol: ' + item['path'])
    started = time.monotonic()
    summary = {'schema': 'robotwin-visible-cpu-probe-result-v1', 'started_at': now(),
               'supervisor_pid': os.getpid(), 'protocol_sha256': prepared['protocol_sha256'],
               'admitted': False, 'semantic_accuracy_evaluated': False, 'gpu_used': False,
               'cases': [], 'status': 'running'}
    atomic(OUT / 'supervisor.json', summary)
    print(f'SUPERVISOR_PID={os.getpid()} OUTPUT={OUT}', flush=True)
    guard = Guard()
    child = None
    abort = None
    cancelled = [None]
    def on_signal(signum, frame):
        cancelled[0] = f'signal_{signum}'
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    try:
        for case in protocol['cases']:
            directory = OUT / case['case_id']
            sample = guard.sample()
            abort = abort or guard.reason(sample) or cancelled[0]
            if time.monotonic() - started >= TOTAL_LIMIT - 10:
                abort = abort or 'total_wall_budget'
            if abort:
                result = {'case_id': case['case_id'], 'status': 'not_started', 'reason': abort,
                          'admitted': False, 'parsed': None, 'raw_reply': '', 'errors': [abort]}
                atomic(directory / 'result.json', result)
                summary['cases'].append(result)
                continue
            with (directory / 'child.log').open('wb') as log:
                case_start = time.monotonic()
                child = subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()),
                                          '--case', case['case_id']], cwd=ROOT, stdout=log,
                                         stderr=subprocess.STDOUT, env=os.environ.copy(), start_new_session=True)
                atomic(directory / 'process.json', {'pid': child.pid, 'supervisor_pid': os.getpid(),
                                                    'started_at': now(), 'hard_limit_s': CASE_LIMIT})
                print(f'CASE_PID={child.pid} CASE={case["case_id"]}', flush=True)
                kill_reason = None
                peaks = {}
                while child.poll() is None:
                    sample = guard.sample(child.pid)
                    append(OUT / 'resources.jsonl', {'case_id': case['case_id'], 'child_pid': child.pid, **sample})
                    for key in ('system_ram_percent', 'system_cpu_percent', 'VmRSS_bytes', 'VmHWM_bytes', 'VmSwap_bytes'):
                        if sample.get(key) is not None:
                            peaks[key] = max(peaks.get(key, 0), sample[key])
                    abort = guard.reason(sample) or cancelled[0]
                    if time.monotonic() - started >= TOTAL_LIMIT - 10:
                        abort = abort or 'total_wall_budget'
                    kill_reason = abort
                    if time.monotonic() - case_start >= CASE_LIMIT:
                        kill_reason = kill_reason or 'case_hard_timeout_120s'
                    if kill_reason:
                        terminate_group(child)
                        break
                    time.sleep(0.2)
                exit_code = child.wait(timeout=5)
                elapsed = time.monotonic() - case_start
                child = None
            path = directory / 'result.json'
            if path.exists():
                result = json.loads(path.read_text())
                if kill_reason:
                    result['child_report_before_kill_status'] = result['status']
            else:
                phase_path = directory / 'phase.json'
                result = {'case_id': case['case_id'], 'parsed': None, 'errors': [], 'admitted': False,
                          'last_phase': json.loads(phase_path.read_text()) if phase_path.exists() else None}
            result.update(child_exit_code=exit_code, supervised_elapsed_s=elapsed, sampled_resource_peaks=peaks,
                          supervisor_recorded_at=now())
            if kill_reason or not path.exists() or (exit_code != 0 and 'status' not in result):
                result['status'] = 'terminated' if kill_reason else 'child_failed_without_result'
                result['errors'].append({'stage': 'supervisor', 'message': kill_reason or 'no complete child result'})
                raw = directory / 'raw_reply.txt'
                result.update(raw_reply=raw.read_text() if raw.exists() else '', reply_complete=False, parsed=None)
            atomic(path, result)
            summary['cases'].append(result)
            if result['status'] in ('interface_or_runtime_failure', 'child_failed_without_result'):
                abort = 'interface_or_runtime_failure_no_fallback'
            atomic(OUT / 'supervisor.json', {**summary, 'last_completed_case': case['case_id']})
            print(f'CASE_DONE={case["case_id"]} STATUS={result["status"]} ELAPSED_S={elapsed:.3f}', flush=True)
    except BaseException:
        abort = 'supervisor_exception'
        summary['supervisor_error'] = traceback.format_exc()
    finally:
        if child is not None and child.poll() is None:
            terminate_group(child)
        existing = {r['case_id'] for r in summary['cases']}
        for case in protocol['cases']:
            if case['case_id'] not in existing:
                result = {'case_id': case['case_id'], 'status': 'not_completed', 'reason': abort,
                          'admitted': False, 'parsed': None, 'raw_reply': '', 'errors': [abort]}
                atomic(OUT / case['case_id'] / 'result.json', result)
                summary['cases'].append(result)
        summary.update(status='finished_with_errors' if abort or any(
            r['status'] != 'completed_schema_valid_not_admitted' for r in summary['cases']) else 'finished_not_admitted',
            stopped_reason=abort, finished_at=now(), total_s=time.monotonic() - started)
        atomic(OUT / 'summary.json', summary)
        atomic(OUT / 'supervisor.json', summary)
        lines = ['# TRAIN CPU visual-usability probe — NOT admission', '',
                 f'Status: `{summary["status"]}`; total {summary["total_s"]:.2f}s; no GT labels/corrections.', '',
                 '| Case | Status | Drawer open | Held | Release | Supervised seconds |',
                 '|---|---|---|---|---|---:|']
        for result in summary['cases']:
            parsed = result.get('parsed') or {}
            values = [parsed.get(p, {}).get('state', 'not parsed') for p in PREDICATES]
            lines.append('| ' + ' | '.join([result['case_id'], result['status'], *values,
                         str(round(result.get('supervised_elapsed_s', 0), 3))]) + ' |')
        lines += ['', 'Values above are raw model judgments, not verified facts. Missing/timeout output is not a false state.',
                  'Single-timestamp images cannot establish temporal release, stable placement, or a successful task.',
                  'No automatic repair, retries, model substitution, installation, or admission occurred.']
        (OUT / 'summary.md').write_text('\n'.join(lines) + '\n')
        print(f'FINISHED STATUS={summary["status"]} TOTAL_S={summary["total_s"]:.3f}', flush=True)
    return 0 if summary['status'] == 'finished_not_admitted' else 2


def self_test():
    OUT.mkdir(parents=True, exist_ok=True)
    setup_environment()
    valid = {p: {'state': 'unknown', 'views': ['cam_head'], 'evidence': 'Gripper relationship is occluded.'} for p in PREDICATES}
    assert parse_reply(json.dumps(valid))[0] == valid
    assert parse_reply('```json\n' + json.dumps(valid) + '\n```')[1]['outer_json_fence_removed']
    rejected = 0
    for value in ({'only_one': {}}, {**valid, 'action': 'open'},
                  {**valid, 'visible_release': {'state': True, 'views': ['cam_head'], 'evidence': 'x'}},
                  {**valid, 'object_visibly_held': {'state': 'true', 'views': ['third_view'], 'evidence': 'x'}}):
        try:
            parse_reply(json.dumps(value))
        except ValueError:
            rejected += 1
    assert rejected == 4
    assert Guard.reason({'system_ram_percent': 85, 'system_cpu_percent': None})
    assert Guard.reason({'system_ram_percent': 1, 'system_cpu_percent': 85})
    assert Guard.reason({'system_ram_percent': 84.99, 'system_cpu_percent': None}) is None
    # Verify SIGKILL group cleanup using a pure-stdlib sleeper, not a model/simulator.
    child = subprocess.Popen([sys.executable, '-B', '-c', 'import time; time.sleep(30)'], start_new_session=True)
    terminate_group(child)
    assert child.returncode == -signal.SIGKILL
    receipt = {'at': now(), 'status': 'pass', 'checks': 10, 'model_loaded': False,
               'datasets_read': False, 'gpu_used': False, 'sleeper_exit_code': child.returncode,
               'scope': 'synthetic schema/guard/group-kill unit checks only; not VLM or physical admission'}
    atomic(OUT / 'self_test.json', receipt)
    print(json.dumps(receipt), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--run', action='store_true')
    mode.add_argument('--case')
    mode.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.run:
        return run()
    elif args.case:
        return case_worker(args.case)
    else:
        self_test()
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception:
        OUT.mkdir(parents=True, exist_ok=True)
        atomic(OUT / f'bootstrap_failure_{os.getpid()}.json', {'at': now(), 'traceback': traceback.format_exc(),
                                                            'admitted': False, 'model_fallback': False})
        raise
