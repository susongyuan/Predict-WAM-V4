#!/usr/bin/env python3
"""Paired5 interpreter transport, adapted from the historical strict bootstrap.
Original sources stay byte-identical; only this separately fingerprinted transport
installs the receipt-bound resource overlay and once-per-launch large-file hashing.
"""
import atexit
import hashlib
import importlib.abc
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import runpy
import sys

IMAGE = Path(__file__).resolve().parents[1]
FROZEN = IMAGE / 'frozen_runtime'
BASE = IMAGE.parents[1]
PROJECT = Path('/home/user/Robotic/project2')
VARIANT = IMAGE.name
ENV = sys.argv[1]
ARGS = sys.argv[2:]
while ARGS and ARGS[0] in ('-B','-I','-E','-s','-u'): ARGS.pop(0)
if not ARGS: raise SystemExit('Interactive interpreter disabled')
for path, digest in json.loads((BASE / 'runner/code_integrity.json').read_text()).items():
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
        raise SystemExit('Paired5 transport changed: ' + path)

def load_admission():
    spec = importlib.util.spec_from_file_location('paired5_admission', BASE / 'runner/admission.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module

admission = load_admission()
CPU = ARGS[0] == '-c'

def output_directory():
    for flag in ('--log-dir','--output-dir'):
        if flag in ARGS:
            out = Path(ARGS[ARGS.index(flag)+1])
            return out.parent if flag == '--output-dir' else out
    if '--run-name' in ARGS:
        return FROZEN / 'outputs/behavior/closed_loop' / ARGS[ARGS.index('--run-name')+1]
    return None

out = output_directory()
if not CPU:
    if out is None or out.parent != FROZEN / 'outputs/behavior/closed_loop' or out.resolve() != out:
        raise SystemExit('No isolated registered output')
    admission.live_ticket(VARIANT, out.name)
# CPU helpers remain CPU even when spawned by an executing launcher. Systemd
# children get explicit visibility here, independent of their manager environment.
os.environ.update(CUDA_VISIBLE_DEVICES='' if CPU else '0', P2_TASK_ID='0', P2_HEAD_RES='720',
                  P2_WRIST_RES='224', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2',
                  MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
os.chdir(FROZEN)
sys.dont_write_bytecode = True
sys.path[:] = [str(FROZEN), str(FROZEN / 'scripts')] + [
    p for p in sys.path if p and Path(p).resolve() not in
    (PROJECT, PROJECT / 'scripts', BASE / 'runner', IMAGE / 'runner')]
inventory = json.loads((IMAGE / 'runner/freeze_inventory.json').read_text())
local_hashes = {str(Path(r['runtime']).resolve()):r['sha256'] for r in inventory['files'] if r['copied']}
local_scripts = {Path(p).stem for p in local_hashes if Path(p).parent == FROZEN / 'scripts'}
loaded = {}

class FrozenFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if CPU and (fullname == 'omnigibson' or fullname.startswith(('omnigibson.','isaacsim'))):
            # find_spec('omnigibson') is used for origin checking; do not import its loader.
            if fullname != 'omnigibson': raise ImportError('CPU transport forbids simulator import')
        if not (fullname == 'harness' or fullname.startswith('harness.') or fullname in local_scripts): return None
        search = [str(FROZEN)] if fullname == 'harness' else ([str(FROZEN.joinpath(*fullname.split('.')[:-1]))]
                 if fullname.startswith('harness.') else [str(FROZEN / 'scripts')])
        spec = importlib.machinery.PathFinder.find_spec(fullname, search)
        if spec is None or spec.origin is None: raise ImportError('Unfrozen module refused: '+fullname)
        source = Path(spec.origin).resolve()
        if str(source) not in local_hashes or hashlib.sha256(source.read_bytes()).hexdigest() != local_hashes[str(source)]:
            raise ImportError('Unverified frozen import: '+str(source))
        loaded[fullname] = str(source)
        return spec

sys.meta_path.insert(0, FrozenFinder())
import harness
assert Path(harness.__file__).resolve() == FROZEN / 'harness/__init__.py'
if ENV == 'behavior6':
    spec = importlib.util.find_spec('omnigibson')
    expected = PROJECT / 'third_party/BEHAVIOR-1K-isaac6/OmniGibson/omnigibson/__init__.py'
    if spec is None or Path(spec.origin).resolve() != expected: raise SystemExit('Wrong OmniGibson origin')
admission.install_overlay()

def receipt():
    if out is None or not out.is_dir(): return
    directory = out / 'runtime_imports'; directory.mkdir(exist_ok=True)
    modules = {k:str(Path(v.__file__).resolve()) for k,v in list(sys.modules.items())
               if v is not None and getattr(v,'__file__',None)
               and (k == 'harness' or k.startswith('harness.') or k in local_scripts)}
    payload = {'schema':2, 'pid':os.getpid(), 'environment':ENV, 'argv':ARGS, 'cwd':str(Path.cwd()),
               'strict_imports':loaded, 'modules':modules, 'task_id':'0', 'head_resolution':720, 'wrist_resolution':224,
               'cuda_visible_devices':os.environ['CUDA_VISIBLE_DEVICES'], 'cpu_probe':CPU,
               'bootstrap_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'authorization_sha256':admission.AUTH_SHA, 'variant':VARIANT,
               'resource_adapter_sha256':admission.sha(BASE / 'runner/admission.py')}
    with (directory / f'{ENV}_{os.getpid()}.json').open('x') as f: json.dump(payload,f,indent=2)

atexit.register(receipt)
if CPU:
    # CPU probes may import torch, but cannot initialize CUDA or spawn any GPU worker.
    def audit(event, args):
        if event == 'import' and (args[0] == 'omnigibson' or args[0].startswith(('omnigibson.','isaacsim'))):
            raise RuntimeError('CPU transport forbids simulator import')
        if event == 'subprocess.Popen':
            tokens = args[1] if isinstance(args[1], (list,tuple)) else [args[1]]
            if any(Path(str(t)).name in ('nvidia-smi','systemd-run') for t in tokens):
                raise RuntimeError('CPU transport forbids GPU launch/query')
    sys.addaudithook(audit)
    sys.argv = ['-c', *ARGS[2:]]
    exec(compile(ARGS[1], '<paired5-cpu-python>', 'exec'), {'__name__':'__main__'})
elif ARGS[0] == '-m':
    if ARGS[1] != 'omnigibson.eval.eval': raise SystemExit('Only frozen evaluator module allowed')
    sys.argv = ARGS[1:]; runpy.run_module(ARGS[1], run_name='__main__', alter_sys=True)
else:
    entry = Path(ARGS[0]).resolve()
    expected = local_hashes.get(str(entry))
    if expected is None or hashlib.sha256(entry.read_bytes()).hexdigest() != expected: raise SystemExit('Unverified entry')
    sys.argv = [str(entry), *ARGS[1:]]
    if entry == FROZEN / 'scripts/behavior_closed_loop.py':
        import behavior_closed_loop as closed
        original_sha = closed.sha256_file
        cache = {}
        expected_large = {str(Path(r['runtime']).resolve()):r['sha256'] for r in inventory['files']
                          if Path(r['runtime']).stat().st_size >= 16*1024**2}
        def file_stat(p):
            s = p.stat(); return [s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_ino,s.st_dev]
        def verified_sha(path):
            p = Path(path).resolve()
            if str(p) not in expected_large: return original_sha(p)
            current = file_stat(p)
            if str(p) in cache:
                admission.require(cache[str(p)]['stat'] == current, 'Large payload changed during launch')
                return cache[str(p)]['sha256']
            digest = original_sha(p)
            admission.require(file_stat(p) == current and digest == expected_large[str(p)], 'Large payload integrity mismatch')
            cache[str(p)] = {'sha256':digest, 'stat':current, 'fresh_content_hash':True}
            return digest
        closed.sha256_file = verified_sha
        try: closed.main()
        finally:
            if out.is_dir():
                with (out / 'large_payload_verification.json').open('x') as f: json.dump(cache,f,indent=2)
    else:
        runpy.run_path(str(entry), run_name='__main__')
