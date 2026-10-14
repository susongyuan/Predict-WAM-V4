"""Frozen TRAIN demonstration retrieval, distinct from verified online outcomes.

Every candidate is a TRAIN endpoint transition used during condition adaptation.
Initial head CLS ranks scene context only; this does not identify the target.
"""
import json
from pathlib import Path
import numpy as np
from harness.robotwin_policy_worker import InfrastructureError, sha256

class DemoConditionMemory:
    def __init__(self, bank, adapter, *, task=None, train_episodes=None):
        bank, adapter = Path(bank).resolve(), Path(adapter).resolve()
        protocol = json.loads((adapter.parent / 'protocol.json').read_text())
        for name, key in [('action_vtt_prototype.npy', 'VTT_prototype_sha256'),
                          ('vtt_instances.npy', 'VTT_instances_sha256')]:
            if sha256(bank / name) != protocol[key]:
                raise InfrastructureError('TRAIN condition bank changed')
        result = json.loads((bank / 'result.json').read_text())
        if protocol.get('task_name') is not None:
            for name, key in [('context_cls.npy', 'VTT_context_sha256'), ('result.json', 'VTT_bank_result_sha256')]:
                if sha256(bank / name) != protocol.get(key):
                    raise InfrastructureError('task-scoped TRAIN retrieval metadata changed')
            if (result.get('completed') is not True or result.get('heldout_used') is not False
                    or result.get('source_split') != 'train'
                    or result.get('split_sha256') != protocol.get('split_sha256')
                    or result.get('cache_id') != protocol.get('feature_cache_id')):
                raise InfrastructureError('task-scoped TRAIN retrieval provenance mismatch')
        count = result.get('TRAIN_episodes')
        bank_task = result.get('task', 'put_object_cabinet')
        if (type(count) is not int or count < 2
                or task is not None and (bank_task != task or protocol.get('task_name', 'put_object_cabinet') != task)
                or train_episodes is not None and count != train_episodes):
            raise InfrastructureError('TRAIN condition library task/count mismatch')
        self.names = [row['episode'] for row in result['records']]
        self.instances = np.load(bank / 'vtt_instances.npy', allow_pickle=False)
        self.contexts = np.load(bank / 'context_cls.npy', allow_pickle=False)
        self.prototype = np.load(bank / 'action_vtt_prototype.npy', allow_pickle=False)
        if (len(self.names) != count or len(set(self.names)) != count
                or self.instances.shape != (count, 1024) or self.contexts.shape != (count, 1024)
                or self.prototype.shape != (1024,)
                or not all(np.isfinite(a).all() for a in (self.instances, self.contexts, self.prototype))):
            raise InfrastructureError('invalid TRAIN condition library')
        self.context_units = self.contexts / np.maximum(np.linalg.norm(self.contexts, axis=1, keepdims=True), 1e-8)
        self.reset()

    def reset(self):
        self.used = set()

    def select(self, cls):
        cls = np.asarray(cls, np.float32)
        if cls.shape != (1024,) or not np.isfinite(cls).all() or np.linalg.norm(cls) < 1e-3:
            raise InfrastructureError('invalid initial context for retrieval')
        scores = self.context_units @ (cls / np.linalg.norm(cls))
        ranks = np.argsort(-scores, kind='stable')
        chosen = next((int(i) for i in ranks if int(i) not in self.used), None)
        if chosen is None:
            raise InfrastructureError('TRAIN candidates exhausted')
        self.used.add(chosen)
        return self.instances[chosen].copy(), 'TRAIN-demo:' + self.names[chosen]
