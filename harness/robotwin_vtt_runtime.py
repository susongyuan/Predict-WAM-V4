"""Explicit trained VTT deployment; completion remains an advisory nomination.

Loading requires the completed paired validation receipt and the exact frozen
base and training sources. No simulator outcomes are accepted as inputs.
"""
import json
from pathlib import Path
import numpy as np
from harness.robotwin_policy_worker import InfrastructureError, sha256


def adapter_metadata(path, base_hash):
    path = Path(path).resolve()
    protocol = json.loads((path.parent / 'protocol.json').read_text())
    result = json.loads((path.parent / 'result.json').read_text())
    if (result.get('status') != 'completed'
            or result.get('offline_compatibility_admitted') is not True
            or result.get('heldout_used') is not False
            or protocol.get('base_checkpoint_sha256') != base_hash
            or protocol.get('backbone_frozen') is not True
            or protocol.get('heldout_used') is not False
            or result.get('steps') != protocol.get('steps')
            or path.name != f"adapter_step_{result['steps']:06d}.pt"):
        raise InfrastructureError('VTT artifact lacks matching completed compatibility receipt')
    before, after = result['validation_before'], result['validation_after']
    vals = np.array([before['base'], before['adapter'], after['base'], after['adapter']])
    if (not np.isfinite(vals).all() or np.min(vals) <= 0
            or before['base'] != before['adapter']
            or after['adapter'] > 1.05 * after['base']):
        raise InfrastructureError('invalid VTT paired compatibility values')
    for source, expected in protocol['source_sha256'].items():
        if sha256(source) != expected:
            raise InfrastructureError('VTT frozen training source differs: ' + source)
    return protocol, {'sha256': sha256(path), 'path': str(path),
                      'protocol_sha256': sha256(path.parent / 'protocol.json'),
                      'result_sha256': sha256(path.parent / 'result.json'),
                      'steps': result['steps'], 'offline_compatibility_admitted': True,
                      'physical_success_verified': False}


def load_adapter(path, action_model, base_hash, *, task=None):
    import torch
    from harness.robotwin_vtt_adapter import install
    protocol, metadata = adapter_metadata(path, base_hash)
    artifact_task = protocol.get('task_name', 'put_object_cabinet')
    if task is not None and artifact_task != task:
        raise InfrastructureError('VTT adapter belongs to another task')
    metadata['task'] = artifact_task
    ck = torch.load(path, map_location='cpu', weights_only=False)
    proto = ck.get('prototype')
    state = ck.get('state_dict', {})
    if (ck.get('format') != 'robotwin-vtt-residual-adapter-v1'
            or ck.get('base_checkpoint_sha256') != base_hash
            or ck.get('protocol') != protocol or ck.get('steps') != protocol['steps']
            or not isinstance(proto, torch.Tensor) or tuple(proto.shape) != (1024,)
            or not torch.isfinite(proto).all() or proto.norm() < 1e-3
            or set(state) != {'weight'}
            or tuple(state['weight'].shape) != (768, 1024)
            or not torch.isfinite(state['weight']).all()):
        raise InfrastructureError('invalid VTT adapter tensors or contract')
    install(action_model, state)
    return proto.float().numpy().copy(), metadata


class VTTMonitor:
    """DINO head CLS progress can nominate closure; it cannot certify grasp."""
    def __init__(self, bank, *, task=None, train_episodes=None):
        bank = Path(bank).resolve()
        result = json.loads((bank / 'result.json').read_text())
        count = result.get('TRAIN_episodes')
        bank_task = result.get('task', 'put_object_cabinet')
        if (type(count) is not int or count < 2
                or train_episodes is not None and count != train_episodes
                or task is not None and bank_task != task):
            raise InfrastructureError('TRAIN monitor task/count mismatch')
        self.prototype = np.load(bank / 'prototypes.npy', allow_pickle=False)[0]
        if self.prototype.shape != (1024,) or not np.isfinite(self.prototype).all():
            raise InfrastructureError('invalid monitoring prototype')
        self.norm = float(np.linalg.norm(self.prototype))
        if self.norm < 1e-3:
            raise InfrastructureError('degenerate monitoring prototype')
        self.unit = self.prototype / self.norm
        self.progress_threshold = float(result['done_progress'])
        self.alignment_threshold = float(result['done_alignment'])
        self.min_steps = result['min_steps']
        if (not np.isfinite([self.progress_threshold, self.alignment_threshold]).all()
                or type(self.min_steps) is not int or self.min_steps < 1):
            raise InfrastructureError('invalid TRAIN monitor receipt')
        self.identity = {'prototype_sha256': sha256(bank / 'prototypes.npy'),
                         'result_sha256': sha256(bank / 'result.json'),
                         'task': bank_task, 'TRAIN_episodes': count,
                         'physical_verifier_admitted': False}
        self.reset()

    def reset(self):
        self.start = None

    def observe(self, cls, step):
        cls = np.asarray(cls, np.float32)
        if cls.shape != (1024,) or not np.isfinite(cls).all():
            raise InfrastructureError('invalid current head CLS')
        if self.start is None:
            if step != 0:
                raise InfrastructureError('VTT start must bind the first observation')
            self.start = cls.copy()
        delta = cls - self.start
        along = float(delta @ self.unit)
        progress = along / self.norm
        alignment = along / max(float(np.linalg.norm(delta)), 1e-8)
        return {'progress': progress, 'alignment': alignment,
                'closure_candidate': bool(step >= self.min_steps
                    and progress >= self.progress_threshold
                    and alignment >= self.alignment_threshold),
                'task_success_verified': None, 'status': 'advisory_unadmitted',
                'feature_source': 'current_RGB_DINO_layer_minus4_head_CLS'}
