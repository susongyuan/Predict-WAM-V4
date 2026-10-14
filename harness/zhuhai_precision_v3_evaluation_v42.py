"""Read-only, self-contained loader for the frozen Precision-WAM V3 fork.

The training checkpoint intentionally retains its V2 base contract alongside a
V3 extension.  That inherited metadata does not describe the actual refinement
head.  Validate both contracts, construct V3 before loading, and require the
saved tensors to identify the full-32-step, 588-patch head.  No initializer,
encoder inference, optimizer or simulator is needed to load a milestone.

Camera rays are checked against the *existing training implementation*.  This
loader neither repairs its approximate geometry nor silently changes a trained
checkpoint's inputs.  A geometry correction needs a separately trained version.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
import tempfile

from harness import zhuhai_precision_v2_training as v2
from harness import zhuhai_precision_v3_training as v3

FORMAT = v3.FORMAT
TASK = v2.TASK
REFINE_STEPS = 32
CLOCK_CONFIG = v2.CLOCK_CONFIG
protocol_binding = v2.protocol_binding
robot_contract = v2.robot_contract
validate_postprocessing = v2.validate_postprocessing
V3_SOURCES = ('harness/wam_precision_v3.py',
              'harness/zhuhai_precision_v3_training.py',
              'scripts/train_zhuhai_wam_precision_v3.py')


def validate_training_contract(contract):
    """Keep the frozen V2 provenance gates and validate the separate V3 spec."""
    if not isinstance(contract, dict) or contract.get('format') != FORMAT:
        raise ValueError('Precision V3 checkpoint format required; V2 is not a V3 milestone')
    spec = contract.get('precision_v3')
    if (spec != v3.PRECISION_V3_SPEC
            or contract.get('config', {}).get('model', {}).get('precision_v3') != spec
            or spec.get('schema') != 'full_chunk_camera_ray_precision_v3'
            or type(spec.get('refine_steps')) is not int or spec['refine_steps'] != REFINE_STEPS):
        raise ValueError('Precision V3 full-32 refinement contract mismatch')
    # This validates statistics digests, TRAIN provenance, condition bank, splits,
    # frozen feature binding, IDM scales, native16/10Hz and the inherited V2 base.
    v2.validate_training_contract(dict(contract, format=v2.FORMAT))
    source_files = contract.get('source_files', {})
    if any(str(v2.ROOT / rel) not in source_files for rel in V3_SOURCES):
        raise ValueError('Precision V3 implementation source binding missing')
    return contract


def validate_model_identity(state_dict):
    """Validate actual V3 identity before allocating the production action model.

    The production V3 fork uses DINO-1024, hidden-768, 14x14 patches per view.
    The version/configuration/ray buffers are exact identity checks, not a loose
    inference from stale inherited spec fields or merely compatible dimensions.
    """
    import torch
    from harness.wam_precision_v3 import camera_ray_grid

    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ValueError('Missing Precision V3 model state')
    expected_shapes = {
        'precision.version': (),
        'precision.configuration': (5,),
        'precision.patch_xy': (1, 1, 196, 2),
        'precision.patch_ray': (1, 1, 196, 3),
        'precision.patch_proj.weight': (768, 1029),
        'precision.patch_proj.bias': (768,),
        'precision.clock.weight': (768, 6),
        'precision.residual.weight': (16, 768),
        'precision.residual.bias': (16,),
        'dino_adapters.0.query_embed': (1, 16, 768),
        'action_proj.weight': (768, 16),
        'proprio_proj.weight': (768, 16),
        'output_proj.weight': (16, 768),
        'action_pos_emb': (1, 32, 768),
    }
    for name, shape in expected_shapes.items():
        value = state_dict.get(name)
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
            raise ValueError(f'Precision V3 tensor shape mismatch: {name}; expected {shape}')
    version = state_dict['precision.version']
    configuration = state_dict['precision.configuration']
    if version.dtype != torch.int64 or int(version.item()) != 3:
        raise ValueError('Precision V3 actual model version must be 3')
    expected_configuration = torch.tensor([16, 32, 0, 1, 2], dtype=torch.int64)
    if configuration.dtype != torch.int64 or not torch.equal(configuration.cpu(), expected_configuration):
        raise ValueError('Precision V3 actual configuration must be [16,32,0,1,2]')
    axis = (torch.arange(14, dtype=torch.float32) + .5) * (2. / 14) - 1.
    yy, xx = torch.meshgrid(axis, axis, indexing='ij')
    expected_xy = torch.stack((xx, yy), dim=-1).reshape(1, 1, 196, 2)
    xy, ray = state_dict['precision.patch_xy'], state_dict['precision.patch_ray']
    if xy.dtype != torch.float32 or not torch.equal(xy.cpu(), expected_xy):
        raise ValueError('Precision V3 patch-centre XY identity mismatch')
    if ray.dtype != torch.float32 or not torch.equal(ray.cpu(), camera_ray_grid(expected_xy)):
        raise ValueError('Precision V3 fixed training ray identity mismatch')
    for name, value in state_dict.items():
        if not isinstance(value, torch.Tensor):
            raise ValueError(f'Non-tensor Precision V3 model state entry: {name}')
        if (value.is_floating_point() or value.is_complex()) and not torch.isfinite(value).all():
            raise ValueError(f'Nonfinite Precision V3 model state: {name}')
    return dict(version=3, refine_steps=32, execution_prefix=8, views=3,
                patches_per_view=196, dense_kv_tokens=588,
                queries_per_view=16, global_visual_tokens=48,
                hidden_dim=768, action_dim=16, patch_projection_input=1029,
                ray_semantics='exact fixed approximate camera-frame training ray; no extrinsics/depth')


def build_wrapper(cfg, statistics, *, variant, idm_statistics,
                  state_dict, device='cpu'):
    """Construct fresh V3, then strict-load it; never strict-load V3 into V2."""
    import torch
    from harness import zhuhai_training as zt
    from harness.wam_v3 import WamV3Wrapper
    from harness.wam_hjepa import upgrade_wrapper
    from harness.wam_precision_v3 import upgrade_precision_v3

    zt.validate_statistics(statistics, cfg)
    model = zt.create_action_model(cfg, device=device)
    with tempfile.TemporaryDirectory(prefix='precision-v3-eval-normalization-') as directory:
        normalization_path = Path(directory) / 'norm.json'
        normalization_path.write_text(json.dumps(statistics['normalization'], allow_nan=False))
        base = WamV3Wrapper(
            model, views=cfg['model']['views'], stats=statistics['temporal_stats'],
            norm_stats_path=normalization_path, normalization_key=zt.NORMALIZATION_KEY,
            proprio_mask='native_all16', proprio_keep_override=statistics['proprio_keep_mask'],
            train_cfg=cfg['training'], vision_encoder=None,
            feat_layers=(-12, -8, -4), num_register_tokens=4, device=device).to(device)
    wrapper = upgrade_wrapper(base, variant=variant,
                              idm_scale=idm_statistics['scale16'], **v2.AUX_CONFIG)
    wrapper = upgrade_precision_v3(wrapper, queries_per_view=16,
                                   refine_steps=REFINE_STEPS,
                                   aux_seed=v2.PRECISION_CONFIG['aux_seed'])
    wrapper.action_model.load_state_dict(state_dict, strict=True)
    expected_scale = torch.tensor(idm_statistics['scale16'], dtype=torch.float32, device=device)
    if not torch.equal(wrapper.action_model.hjepa_aux.idm_scale, expected_scale):
        raise ValueError('Precision V3 checkpoint IDM scale differs from TRAIN statistics')
    # State loading must not alter the requested architecture or its identity.
    validate_model_identity(wrapper.action_model.state_dict())
    wrapper.eval()
    return wrapper


def _load_verified_checkpoint(path):
    """Verify bytes and the saved training contract before constructing a model."""
    import torch

    path = v2.no_symlink(v2.resolve(path))
    receipt_path = v2.no_symlink(path.with_name(path.name + '.sha256.json'))
    receipt = v2.read_json(receipt_path)
    if (type(receipt.get('size')) is not int or receipt['size'] != path.stat().st_size
            or receipt.get('sha256') != v2.sha256_file(path)):
        raise ValueError('Precision V3 checkpoint receipt mismatch')
    checkpoint = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    step = checkpoint.get('step')
    if (checkpoint.get('format') != FORMAT
            or checkpoint.get('model_only') is not True or receipt.get('model_only') is not True
            or type(step) is not int or step < 0
            or type(receipt.get('step')) is not int or receipt['step'] != step):
        raise ValueError('Immutable Precision V3 milestone required; V2/resume files are rejected')
    contract = checkpoint.get('contract')
    if (not isinstance(contract, dict)
            or checkpoint.get('contract_sha256') != v2.digest(contract)):
        raise ValueError('Precision V3 contract SHA mismatch')
    validate_training_contract(contract)
    v2.verify_code_hashes(contract['source_files'])
    return path, receipt, checkpoint


def read_contract(path):
    """CPU plan gate: verify checkpoint, manifest and collection/postprocessing.

    Match the evaluator's native16 read_contract API without calling its V2
    format gate. No large model or encoder is constructed here.
    """
    path, receipt, checkpoint = _load_verified_checkpoint(path)
    contract = checkpoint['contract']
    manifest = v2.read_json(v2.no_symlink(path.parent / 'manifest.json'))
    if (manifest.get('contract') != contract
            or manifest.get('contract_sha256') != checkpoint['contract_sha256']):
        raise ValueError('Precision V3 saved checkpoint/manifest contract mismatch')
    validate_model_identity(checkpoint.get('model_state_dict'))
    for source in contract['sources'].values():
        if source.get('engine', {}).get('name') != 'Isaac Sim':
            raise ValueError('Precision V3 evaluation requires Isaac Sim collection provenance')
        scene = source['scene']
        if v2.sha256_file(v2.no_symlink(scene['path'])) != scene['sha256']:
            raise ValueError('Precision V3 collection scene changed')
        for row in source['manifests']:
            if v2.sha256_file(v2.no_symlink(row['path'])) != row['sha256']:
                raise ValueError('Precision V3 collection manifest changed')
            for field in ('config_hashes', 'scene_dependency_sha256'):
                mapping = row['contract'].get(field, {})
                if mapping:
                    v2.verify_code_hashes(mapping)
    from harness.zhuhai_action_smoothing import protocol, protocol_hash
    validate_postprocessing(contract['action_postprocessing'])
    current_protocol = protocol()
    if (contract['action_postprocessing'] != current_protocol
            or contract.get('action_postprocessing_sha256') != protocol_hash(current_protocol)):
        raise ValueError('Precision V3 frozen TRAIN action postprocessing changed')
    return contract, receipt


def load_policy(path, task=TASK, device='cpu'):
    """Load one immutable V3 milestone; return (wrapper, TRAIN condition, contract).

    Signature matches V2 with an optional default task. CPU loading is the
    default and does not initialize CUDA or execute DINO. GPU loading is a
    caller-owned operation subject to the project's separate exclusive lock.
    """
    import numpy as np
    path, receipt, checkpoint = _load_verified_checkpoint(path)
    contract = checkpoint['contract']
    if task != TASK or task not in contract['conditions']:
        raise ValueError('Precision V3 task/condition mismatch')
    actual_architecture = validate_model_identity(checkpoint.get('model_state_dict'))
    wrapper = build_wrapper(
        contract['config'], contract['statistics'], variant=contract['hjepa']['variant'],
        idm_statistics=contract['idm_statistics'], device=device,
        state_dict=checkpoint['model_state_dict'])
    # Keep verification provenance outside the frozen checkpoint contract.
    wrapper.precision_v3_load_receipt = dict(
        checkpoint_path=str(path), checkpoint_sha256=receipt['sha256'],
        step=checkpoint['step'], contract_sha256=checkpoint['contract_sha256'],
        strict_load=True, source_hashes_verified=True,
        self_contained=True, initializer_loaded=False, vision_encoder_loaded=False,
        actual_architecture=actual_architecture,
        inherited_v2_fields_are_actual_architecture=False)
    return wrapper, np.asarray(contract['conditions'][task], dtype=np.float32), contract
