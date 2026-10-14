#!/usr/bin/env python3
"""Exec16 runtime entry: strict frozen V4 or V2 weights, unchanged physical loop."""
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from harness import zhuhai_precision_v4_training as v4
from harness import zhuhai_precision_v2_training as v2
from harness import zhuhai_official_braking as braking
from harness.zhuhai_exec16_deployment import (DEPLOYMENT_VERSION, deployment_protocol, protocol,
    overlay, Exec16SmoothedChunkPolicy)
from scripts import eval_zhuhai_wam_precision_v3_v42 as previous
from scripts import eval_zhuhai_wam_precision_v4 as frozen_v4
from scripts.progress1_log_v41 import log

base = previous.base
_client = base.Native16PolicyClient
_serve = base.serve_native16
_write = base.atomic_json
_worker = previous.worker
_loaded = None
_precision = None
RAY_SCOPE = frozen_v4.RAY_SCOPE


def loader_for(path):
    # Choose only the strict original format-specific loader; never mutate a checkpoint.
    manifest = v2.read_json(Path(path).parent / 'manifest.json')
    fmt = manifest.get('contract', {}).get('format')
    if fmt == v4.FORMAT:
        return v4
    if fmt == v2.FORMAT:
        return v2
    raise base.InfrastructureError('Exec16 supports only original verified V2 or V4 model formats')


def read_contract(path):
    frozen, receipt = loader_for(path).read_contract(path)
    return overlay(frozen), receipt


def load_policy(path, *args, **kwargs):
    global _loaded
    api = loader_for(path)
    wrapper, condition, frozen = api.load_policy(path, *args, **kwargs)
    head = wrapper.action_model.precision
    if api is v4:
        identity = v4.validate_model_identity(wrapper.action_model.state_dict())
        actual = dict(version=3, refined_flow_steps=32, dense_kv_tokens=588, queries_per_view=16,
            global_visual_tokens=48, patch_projection_input_dim=1029, ray_scope=RAY_SCOPE, executed_prefix=16)
        _loaded = dict(family='v4', actual=actual, identity=identity,
            alpha=float(wrapper.action_model.chunk_vtt16.alpha.detach().cpu()), spec=frozen['precision_v4'])
    else:
        if (int(head.version) != 2 or head.precision_steps != 8 or head.patch_proj.in_features != 1026
                or head.configuration.detach().cpu().tolist() != [16, 8, 0, 1, 2]):
            raise base.InfrastructureError('Original V2 architecture must remain strict; exec16 does not extend its refiner')
        _loaded = dict(family='v2', actual=dict(version=2, refined_flow_steps=8, dense_kv_tokens=588,
            queries_per_view=16, global_visual_tokens=48, patch_projection_input_dim=1026, executed_prefix=16))
    return wrapper, condition, frozen


def actual_precision_binding(wrapper):
    if _loaded is None:
        raise base.InfrastructureError('Strict original loader must run before exec16 serving')
    return _loaded['actual']


def serve(channel, policy):
    if _loaded is None:
        raise base.InfrastructureError('Exec16 strict model load receipt missing')
    d = deployment_protocol()
    policy.runtime.update(deployment=d, deployment_sha256=base.digest(d), deployment_version=DEPLOYMENT_VERSION,
        training_execution_prefix=8, runtime_execution_prefix=16, execution_prefix=16, action_chunk=32,
        observer_policy_input=False, system_mode='precision_wam_' + _loaded['family'] + '_official_braking_exec16')
    if _loaded['family'] == 'v4':
        policy.runtime.update(precision_v3_actual=_loaded['actual'], precision_v4_actual=_loaded['identity'],
            chunk_vtt16=_loaded['spec'], chunk_vtt16_alpha=_loaded['alpha'], chunk_vtt16_used_online=True,
            v3_warmstart_baseline=False)
    else:
        policy.runtime.pop('precision_v3_actual', None)
        policy.runtime.update(precision_v2_actual=_loaded['actual'], chunk_vtt16_used_online=False,
                              v3_warmstart_baseline=False)
    return _serve(channel, policy)


class Exec16Client(_client):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        rt = self.ready.get('runtime', {})
        d = deployment_protocol()
        if (rt.get('action_postprocessing') != protocol()
                or rt.get('action_postprocessing_sha256') != base.digest(protocol())
                or rt.get('evaluation_version') != braking.VERSION
                or rt.get('evaluation_source_sha256') != base.identity(__file__)['sha256']
                or rt.get('deployment') != d or rt.get('deployment_sha256') != base.digest(d)
                or rt.get('execution_prefix') != 16 or rt.get('runtime_execution_prefix') != 16
                or rt.get('training_execution_prefix') != 8):
            self.close()
            raise base.InfrastructureError('Explicit exec16 worker/deployment identity mismatch')


def write_json(path, value):
    value = dict(value)
    if Path(path).name in ('result.json', 'plan.json'):
        d = deployment_protocol()
        value.update(record_video=False, third_view=False, observer_policy_input=False,
            deployment=d, deployment_sha256=base.digest(d), runtime_execution_prefix=16,
            training_execution_prefix=8)
        if 'control' in value:
            value['control'] = dict(value['control'], execution_prefix=16)
    return _write(path, value)


def configure(checkpoint):
    global _precision
    original = loader_for(checkpoint)
    _precision = SimpleNamespace(FORMAT=original.FORMAT, read_contract=original.read_contract,
        load_policy=load_policy, protocol_binding=v2.protocol_binding, robot_contract=v2.robot_contract,
        validate_postprocessing=v2.validate_postprocessing, validate_training_contract=original.validate_training_contract)
    previous.precision = _precision
    previous.FROZEN_READ_CONTRACT = original.read_contract
    previous.overlay = overlay
    previous.original.clocked_policy = Exec16SmoothedChunkPolicy
    previous.braking = SimpleNamespace(VERSION=braking.VERSION, protocol=protocol,
                                      install_braking=lambda policy, settings: policy)
    previous.actual_precision_binding = actual_precision_binding
    previous.verify_owner = frozen_v4.verify_owner
    previous.__file__ = __file__
    base.zt = _precision
    base.worker = _worker; base.log = log; base.__file__ = __file__
    base.verify_owner = frozen_v4.verify_owner
    base.read_contract = read_contract
    base.Native16PolicyClient = Exec16Client
    base.serve_native16 = serve
    base.atomic_json = write_json


def main():
    import argparse
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--checkpoint', required=True)
    args, _ = p.parse_known_args()
    configure(args.checkpoint)
    return base.main()


if __name__ == '__main__':
    raise SystemExit(main())
