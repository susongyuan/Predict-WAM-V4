#!/usr/bin/env python3
"""Interpreter transport only: run unchanged baseline source with a strict local import root.
No simulation/model construction here. Invoked by BOTH local Python wrappers, including systemd children.
"""
import atexit
import hashlib
import importlib.abc
import importlib.machinery
import json
import os
from pathlib import Path
import runpy
import sys

BASE = Path(__file__).resolve().parents[1]
FROZEN = BASE / 'frozen_runtime'
PROJECT = BASE.parents[2]
ENV = sys.argv[1]
ARGS = sys.argv[2:]
while ARGS and ARGS[0] in ('-B', '-I', '-E', '-s', '-u'):
    ARGS.pop(0)
if not ARGS:
    raise SystemExit('Interactive interpreter use is disabled in the frozen image')
os.chdir(FROZEN)
os.environ.update(P2_TASK_ID='0', P2_HEAD_RES='720', P2_WRIST_RES='224', PYTHONDONTWRITEBYTECODE='1')
sys.dont_write_bytecode = True
sys.path[:] = [str(FROZEN), str(FROZEN / 'scripts')] + [
    p for p in sys.path if p and Path(p).resolve() not in (PROJECT, PROJECT / 'scripts', BASE / 'runner')]
inventory = json.loads((BASE / 'runner/freeze_inventory.json').read_text())
local_hashes = {str(Path(r['runtime']).resolve()): r['sha256'] for r in inventory['files'] if r['copied']}
local_scripts = {Path(p).stem for p in local_hashes if Path(p).parent == FROZEN / 'scripts'}
loaded = {}


class FrozenFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if not (fullname == 'harness' or fullname.startswith('harness.') or fullname in local_scripts):
            return None
        if fullname == 'harness':
            search = [str(FROZEN)]
        elif fullname.startswith('harness.'):
            search = [str(FROZEN.joinpath(*fullname.split('.')[:-1]))]
        else:
            search = [str(FROZEN / 'scripts')]
        spec = importlib.machinery.PathFinder.find_spec(fullname, search)
        if spec is None or spec.origin is None:
            raise ImportError(f'Unfrozen project module refused: {fullname}')
        source = Path(spec.origin).resolve()
        if str(source) not in local_hashes or hashlib.sha256(source.read_bytes()).hexdigest() != local_hashes[str(source)]:
            raise ImportError(f'Unverified frozen import refused: {fullname}: {source}')
        loaded[fullname] = str(source)
        return spec


sys.meta_path.insert(0, FrozenFinder())
# Verify source selected before either torch or OmniGibson is imported.
import harness
assert Path(harness.__file__).resolve() == FROZEN / 'harness/__init__.py'
if ENV == 'behavior6':
    spec = importlib.util.find_spec('omnigibson')  # find_spec does not import the package
    expected = PROJECT / 'third_party/BEHAVIOR-1K-isaac6/OmniGibson/omnigibson/__init__.py'
    if spec is None or Path(spec.origin).resolve() != expected:
        raise SystemExit(f'Wrong installed OmniGibson import origin: {spec}')


def output_directory():
    for flag in ('--log-dir', '--output-dir'):
        if flag in ARGS:
            return Path(ARGS[ARGS.index(flag) + 1])
    if '--run-name' in ARGS:
        return FROZEN / 'outputs/behavior/closed_loop' / ARGS[ARGS.index('--run-name') + 1]
    return None


def receipt():
    out = output_directory()
    if out is None or not out.is_dir() or not out.resolve().is_relative_to(FROZEN / 'outputs/behavior'):
        return
    if '--output-dir' in ARGS:
        out = out.parent
    directory = out / 'runtime_imports'
    directory.mkdir(exist_ok=True)
    modules = {k: str(Path(v.__file__).resolve()) for k, v in list(sys.modules.items())
               if v is not None and getattr(v, '__file__', None)
               and (k == 'harness' or k.startswith('harness.') or k in local_scripts)}
    payload = {'schema': 1, 'pid': os.getpid(), 'environment': ENV, 'argv': ARGS,
               'cwd': str(Path.cwd()), 'strict_imports': loaded, 'modules': modules,
               'task_id': os.environ['P2_TASK_ID'], 'head_resolution': 720, 'wrist_resolution': 224,
               'bootstrap_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    with (directory / f'{ENV}_{os.getpid()}.json').open('x') as f:
        json.dump(payload, f, indent=2)


atexit.register(receipt)
if ARGS[0] == '-c':
    sys.argv = ['-c', *ARGS[2:]]
    exec(compile(ARGS[1], '<frozen-python-c>', 'exec'), {'__name__': '__main__'})
elif ARGS[0] == '-m':
    if ARGS[1] != 'omnigibson.eval.eval':
        raise SystemExit('Only the original evaluator module entrypoint is enabled')
    sys.argv = ARGS[1:]
    runpy.run_module(ARGS[1], run_name='__main__', alter_sys=True)
else:
    entry = Path(ARGS[0]).resolve()
    expected = local_hashes.get(str(entry))
    if expected is None or hashlib.sha256(entry.read_bytes()).hexdigest() != expected:
        raise SystemExit(f'Unverified file entrypoint refused: {entry}')
    sys.argv = [str(entry), *ARGS[1:]]
    runpy.run_path(str(entry), run_name='__main__')
