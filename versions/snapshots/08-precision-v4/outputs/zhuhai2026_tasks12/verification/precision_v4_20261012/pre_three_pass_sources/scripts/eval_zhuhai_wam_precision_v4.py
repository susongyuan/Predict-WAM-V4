#!/usr/bin/env python3
"""V4 policy on the existing native16 Isaac evaluator, with recording disabled."""
from pathlib import Path
import os
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness import zhuhai_precision_v4_training as v4
from harness import zhuhai_precision_v2_training as v2
from scripts import eval_zhuhai_wam_precision_v3_v42 as previous

base = previous.base
_verify_owner = previous.verify_owner
_serve = base.serve_native16
_write_json = base.atomic_json
_loaded = None


def verify_owner(expected):
    _verify_owner(expected)
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            argv = (proc / 'cmdline').read_bytes().decode(errors='replace').split('\0')
        except OSError:
            continue
        if any(Path(arg).name == 'train_zhuhai_wam_precision_v4.py'
               or arg == 'scripts.train_zhuhai_wam_precision_v4' for arg in argv):
            raise base.InfrastructureError('Live V4 trainer prevents evaluation')


def load_policy(*args, **kwargs):
    global _loaded
    wrapper, condition, contract = v4.load_policy(*args, **kwargs)
    identity = v4.validate_model_identity(wrapper.action_model.state_dict())
    alpha = float(wrapper.action_model.chunk_vtt16.alpha.detach().cpu())
    _loaded = dict(identity=identity, alpha=alpha, spec=contract['precision_v4'])
    return wrapper, condition, contract


def serve(channel, policy):
    if _loaded is None:
        raise base.InfrastructureError('V4 policy must be strictly loaded before serving')
    policy.runtime.update(system_mode='precision_wam_v4_official_braking',
        precision_v4_actual=_loaded['identity'], chunk_vtt16=_loaded['spec'],
        chunk_vtt16_alpha=_loaded['alpha'], chunk_vtt16_used_online=True,
        observer_policy_input=False,
        v3_warmstart_baseline=policy.runtime['checkpoint_step'] == 80000 and _loaded['alpha'] == 0.)
    return _serve(channel, policy)


def write_json(path, value):
    if Path(path).name == 'result.json':
        value = dict(value, record_video=False, third_view=False, observer_policy_input=False)
    return _write_json(path, value)


def configure():
    # Reuse the trained V3 head, sampler and physical loop without installing its observer.
    api = SimpleNamespace(FORMAT=v4.FORMAT, read_contract=v4.read_contract,
        load_policy=load_policy, protocol_binding=v2.protocol_binding,
        robot_contract=v2.robot_contract, validate_postprocessing=v2.validate_postprocessing,
        validate_training_contract=v4.validate_training_contract)
    previous.precision = api
    previous.FROZEN_READ_CONTRACT = v4.read_contract
    previous.verify_owner = verify_owner
    previous.__file__ = __file__
    base.zt = api
    base.worker = previous.worker
    base.log = previous.log
    base.__file__ = __file__
    base.verify_owner = verify_owner
    base.read_contract = previous.read_contract
    base.Native16PolicyClient = previous.BrakingClient
    base.serve_native16 = serve
    base.atomic_json = write_json


def main():
    configure()
    return base.main()


if __name__ == '__main__':
    raise SystemExit(main())
