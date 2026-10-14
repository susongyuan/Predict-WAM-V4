"""V4 with +16/+32 Future supervision; preserve the frozen legacy implementation."""
from bisect import bisect_right
import copy
import numpy as np

from harness import zhuhai_precision_v4_training as old
from harness import zhuhai_precision_v2_training as v2
from harness.zhuhai_hjepa_dataset import _last_real_observation, PATCH_SHAPE
from harness.wam_precision_v4 import PrecisionV4Wrapper

FORMAT, TASK, CLOCK_CONFIG = old.FORMAT, old.TASK, old.CLOCK_CONFIG
PRECISION_V4_SPEC = copy.deepcopy(old.PRECISION_V4_SPEC)
PRECISION_V4_SPEC.pop('future_auxiliary_horizons_unchanged')
PRECISION_V4_SPEC['future_auxiliary_horizons'] = [16, 32]
FUTURE_PREDICTION = dict(schema='precision-v4-future16-32-v1', horizons=[16, 32],
    short_weight=.25, long_weight=.25, target_layer=-4, stop_gradient=True,
    legacy_parameter_key='hjepa_aux.future8_offset', execution_prefix=8,
    future_is_policy_input=False, dagger_data_used=False)
SOURCES = ('harness/zhuhai_precision_v4_future16.py',
    'scripts/train_zhuhai_wam_precision_v4_future16.py', 'scripts/train_zhuhai_wam_precision_v4.py',
    'scripts/eval_zhuhai_wam_precision_v4_future16.py', 'scripts/eval_precision_v4_future16_final5.py',
    'scripts/eval_zhuhai_wam_precision_v4.py', 'scripts/eval_precision_v4_final5.py',
    'scripts/test_precision_v4_future16.py')
_new_contract, _validate = old.new_contract, old.validate_training_contract
_datasets, _wrapper = old.build_datasets, old.build_wrapper
_read, _load = old.read_contract, old.load_policy
validate_model_identity = old.validate_model_identity
protocol_binding, robot_contract = v2.protocol_binding, v2.robot_contract
validate_postprocessing = v2.validate_postprocessing


def new_contract(base):
    contract = _new_contract(base)
    contract['future_prediction'] = copy.deepcopy(FUTURE_PREDICTION)
    contract['precision_v4'] = copy.deepcopy(PRECISION_V4_SPEC)
    contract['config']['model']['precision_v4'] = copy.deepcopy(PRECISION_V4_SPEC)
    contract['hjepa']['labels']['short_future_offset'] = 16
    settings = contract['hjepa']['settings']
    settings['future16_weight'] = settings.pop('future8_weight')
    weights = contract['config']['training']['auxiliary_loss']
    weights['future16'] = weights.pop('future8')
    for relative in SOURCES:
        path = v2.ROOT / relative
        contract['source_files'][str(path)] = v2.sha256_file(path)
    contract['hjepa']['code_hashes'] = copy.deepcopy(contract['source_files'])
    return contract


def validate_training_contract(contract):
    if (contract.get('future_prediction') != FUTURE_PREDICTION
            or contract.get('precision_v4') != PRECISION_V4_SPEC
            or contract.get('config', {}).get('model', {}).get('precision_v4') != PRECISION_V4_SPEC
            or contract['hjepa']['labels'].get('short_future_offset') != 16):
        raise ValueError('Require the explicit +16/+32 Future training contract')
    legacy = copy.deepcopy(contract)
    legacy.pop('future_prediction')
    legacy['precision_v4'] = copy.deepcopy(old.PRECISION_V4_SPEC)
    legacy['config']['model']['precision_v4'] = copy.deepcopy(old.PRECISION_V4_SPEC)
    legacy['hjepa']['labels']['short_future_offset'] = 8
    settings = legacy['hjepa']['settings']
    if 'future8_weight' in settings:
        raise ValueError('Legacy +8 loss is not allowed in Future16 training')
    settings['future8_weight'] = settings.pop('future16_weight')
    weights = legacy['config']['training']['auxiliary_loss']
    if 'future8' in weights:
        raise ValueError('Legacy +8 loss is not allowed in Future16 training')
    weights['future8'] = weights.pop('future16')
    _validate(legacy)
    rows = contract['splits'][TASK]['train']
    if (len(rows) != 160 or any('dagger' in row['path'].lower() for row in rows)
            or any(str(v2.ROOT / rel) not in contract['source_files'] for rel in SOURCES)):
        raise ValueError('Require original TRAIN160 and bound Future16 implementation')
    return contract


class Future16Dataset(old.ChunkVTTDataset):
    def __getitem__(self, index):
        sample = super().__getitem__(index)
        if 'future_feats_8' in sample or 'future_valid_mask_8' in sample:
            raise ValueError('Disable the legacy +8 labels before Future16 sampling')
        episode = self.base.episodes[bisect_right(self.base.ends, index)]
        frame = int(sample['anchor']) + 16
        valid = frame <= _last_real_observation(episode)
        sample['future_feats_16'] = (self.base.store.get(episode, frame)[0][:, -1, 5:].copy()
                                    if valid else np.zeros(PATCH_SHAPE, np.uint16))
        sample['future_valid_mask_16'] = np.bool_(valid)
        return sample


def build_datasets(*args, **kwargs):
    datasets = _datasets(*args, **kwargs)
    for tasks in datasets.values():
        for task, dataset in list(tasks.items()):
            dataset.base.base.multi_horizon = False
            tasks[task] = Future16Dataset(dataset.base)
    return datasets


class Future16Wrapper(PrecisionV4Wrapper):
    def _future8(self, batch, out):
        # Keep the trained tensor key; feed this decoder only explicit +16 labels.
        if 'future_feats_8' in batch or 'future_valid_mask_8' in batch:
            raise ValueError('Future16 wrapper rejects legacy +8 labels')
        return super()._future8(dict(batch, future_feats_8=batch['future_feats_16'],
                                    future_valid_mask_8=batch['future_valid_mask_16']), out)

    def forward(self, batch, lambdas=None, action_valid_mask=None, future_valid_mask=None):
        if lambdas is not None:
            if 'future8' in lambdas or 'future16' not in lambdas:
                raise ValueError('Explicit Future16 loss weight required')
            lambdas = dict(lambdas)
            lambdas['future8'] = lambdas.pop('future16')
        total, info, out = super().forward(batch, lambdas, action_valid_mask, future_valid_mask)
        for key in ('future8', 'future8_valid_samples', 'future8_weight'):
            info[key.replace('future8', 'future16')] = info.pop(key)
        out['aux_losses']['future16'] = out['aux_losses'].pop('future8')
        info['future_prediction_horizons'] = [16, 32]
        return total, info, out


def build_wrapper(*args, **kwargs):
    wrapper, notes = _wrapper(*args, **kwargs)
    result = Future16Wrapper(wrapper.base, wrapper.settings['variant'])
    result.settings['future_horizons'] = [16, 32]
    result.training = wrapper.training
    notes['future_prediction'] = copy.deepcopy(FUTURE_PREDICTION)
    return result, notes


def install():
    # The legacy trainer/loader resolves this module at call time; disk sources stay frozen.
    old.new_contract = new_contract
    old.validate_training_contract = validate_training_contract
    old.build_datasets = build_datasets
    old.build_wrapper = build_wrapper


def read_contract(path):
    install()
    return _read(path)


def load_policy(*args, **kwargs):
    install()
    return _load(*args, **kwargs)
