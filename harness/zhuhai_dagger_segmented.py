"""Versioned multi-intervention archives with verified local expert labels.

Original suffix-only archives stay immutable and remain readable. New archives
preserve every real observation/action, but only verified intervention spans
are labels. Whole-task outcome is recorded separately from local success.
"""
from bisect import bisect_right
import json
import math
import types
import numpy as np

FORMAT = 'zhuhai-native16-segmented-expert-mask-v2'
SCHEMA = 'zhuhai-task2-segmented-dagger50-v5'
POLICY_PROTOCOL = 'precision-v2-causal-history-resume-v1'


def goal_met(evidence, goal, key):
    if evidence.get('physics_safety', {}).get('passed') is not True:
        return False
    item = evidence['score']['items'][key]
    obj = evidence['objects'][key]
    if goal == 'grasp':
        return bool(item['grasp_awarded'] and item['hold_seconds'] >= 5.-1e-8
                    and obj['touching_gripper'] and not obj['released'])
    if goal == 'placement':
        return bool(item['grasp_awarded'] and item['placement_valid']
                    and item['stable_seconds'] >= 1.-1e-8 and obj['released']
                    and not obj['touching_robot'])
    raise ValueError('Unknown local expert goal')


def choose_goal(evidence, previous_segments):
    # Historical grasp credit alone cannot mean the object is currently held.
    pending = [k for k in ('bowl', 'spoon', 'cup', 'plate')
               if not evidence['score']['items'][k]['placement_valid']]
    if not pending:
        raise ValueError('No remaining local task')
    held = [k for k in pending if goal_met(evidence, 'grasp', k)]
    key = held[0] if held else pending[0]
    previously_grasped = any(s['passed'] and s['key'] == key and s['goal'] == 'grasp'
                            for s in previous_segments)
    # After a failed handback from a verified grasp, finish THIS object rather
    # than repeat the same grasp indefinitely. Other objects stay with WAM.
    return ('placement' if held or previously_grasped else 'grasp'), key


def finish_segments(recorder, authorities, phases, segments):
    import h5py
    n = recorder.length-1
    authority = np.asarray(authorities, np.uint8)
    if authority.shape != (n,) or len(phases) != n or not np.isin(authority, [0, 1]).all():
        raise ValueError('Every real action needs one authority and phase')
    mask = np.zeros(n, bool)
    ids = np.full(n, -1, np.int32)
    last = 0
    for i, segment in enumerate(segments):
        start, end = segment['start'], segment['end']
        if not (type(start) is int and type(end) is int and last <= start < end <= n
                and authority[start:end].all() and not np.any(ids[start:end] >= 0)):
            raise ValueError('Expert spans must be real, ordered, disjoint actions')
        ids[start:end] = i
        if segment['passed']:
            rule = json.loads(recorder.file['rule_evidence'][end])
            if not goal_met(rule, segment['goal'], segment['key']):
                raise ValueError('Local goal is not physically verified at its boundary')
            mask[start:end] = True
        last = end
    if not np.array_equal(ids >= 0, authority == 1):
        raise ValueError('Every expert action, including failures, needs a span')
    recorder.file.attrs['source_format'] = FORMAT
    recorder.file.attrs['training_requires_expert_mask'] = True
    recorder.file.attrs['successful_expert_scope'] = 'verified local intervention spans only; full-task outcome independent'
    recorder.file.attrs['expert_segments'] = json.dumps(segments, allow_nan=False)
    recorder.file.create_dataset('control_authority', data=authority)
    recorder.file.create_dataset('expert_action_mask', data=mask)
    recorder.file.create_dataset('expert_segment_id', data=ids)
    recorder.file.create_dataset('control_phase', data=np.asarray(phases, object), dtype=h5py.string_dtype('utf-8'))
    return int(mask.sum())


class SegmentedEpisode:
    def __init__(self, row):
        import h5py
        from harness.zhuhai_dataset import SourceEpisode
        from harness.zhuhai_dagger import DaggerEpisode, FORMAT as OLD_FORMAT
        self.row = dict(row)
        with h5py.File(row['path'], 'r') as f:
            fmt = f.attrs['source_format']
        if fmt == OLD_FORMAT:
            self.native = DaggerEpisode(row)
            self.authority = self.native.authority.copy()
            self.mask = self.authority == 1
            self.segments = [dict(start=self.native.expert_start, end=len(self.native),
                                  passed=True, goal='legacy_full_task', key='all')]
            self.variant = 'legacy_full_continuation'
        elif fmt == FORMAT:
            if row.get('split') != 'train':
                raise ValueError('Only TRAIN correction families are allowed')
            original = SourceEpisode.__init__
            validate = types.FunctionType(original.__code__, {**original.__globals__, 'SOURCE_FORMAT': FORMAT},
                                          original.__name__, original.__defaults__, original.__closure__)
            self.native = SourceEpisode.__new__(SourceEpisode)
            validate(self.native, row)
            with h5py.File(row['path'], 'r') as f:
                self.authority = f['control_authority'][:]
                self.mask = f['expert_action_mask'][:]
                self.phases = f['control_phase'].asstr()[:]
                ids = f['expert_segment_id'][:]
                self.segments = json.loads(f.attrs['expert_segments'])
                metadata = json.loads(f.attrs['metadata'])
                if (metadata.get('family_id') != row['family_id'] or metadata.get('split') != 'train'
                        or f.attrs.get('training_requires_expert_mask') != True):
                    raise ValueError('Segmented family/supervision binding differs')
                n = len(self.native)
                if (self.authority.shape != (n,) or self.mask.shape != (n,) or ids.shape != (n,)
                        or len(self.phases) != n or not np.isin(self.authority, [0, 1]).all()
                        or self.mask.dtype != np.bool_ or not self.mask.any()):
                    raise ValueError('Invalid segmented authority/mask')
                expected_mask = np.zeros(n, bool)
                expected_ids = np.full(n, -1, np.int32)
                last = 0
                for i, segment in enumerate(self.segments):
                    start, end = segment['start'], segment['end']
                    if not (type(start) is int and type(end) is int and last <= start < end <= n
                            and self.authority[start:end].all()):
                        raise ValueError('Invalid expert segment boundaries')
                    expected_ids[start:end] = i
                    if type(segment['passed']) is not bool:
                        raise ValueError('Explicit local success flag required')
                    if segment['passed']:
                        if not goal_met(json.loads(f['rule_evidence'][end]), segment['goal'], segment['key']):
                            raise ValueError('Unverified local segment outcome')
                        expected_mask[start:end] = True
                    last = end
                if (not np.array_equal(expected_mask, self.mask) or not np.array_equal(expected_ids, ids)
                        or not np.array_equal(ids >= 0, self.authority == 1)):
                    raise ValueError('Expert labels leak across a failed/policy span')
            self.variant = 'segmented_local_interventions'
        else:
            raise ValueError('Unknown correction format')
        self.anchors = np.flatnonzero(self.mask)
        self.valid_segments = [s for s in self.segments if s['passed']]
        self.policy_count = int((self.authority == 0).sum())
        self.segment_end = np.zeros(len(self.native), np.int64)
        for s in self.valid_segments:
            self.segment_end[s['start']:s['end']] = s['end']

    def __getattr__(self, name):
        return getattr(self.native, name)

    def __len__(self):
        return len(self.native)

    @property
    def supervised_count(self):
        return len(self.anchors)


class SegmentedPrecisionDataset:
    def __init__(self, task, rows, store, condition, chunk=32, future_offset=32,
                 proprio_lags=(2, 4, 8, 16, 32), memory_lags=(4, 16)):
        self.task = task
        self.episodes = [SegmentedEpisode(r) for r in rows]
        self.store, self.condition = store, np.asarray(condition, np.float32).copy()
        self.chunk, self.future_offset = chunk, future_offset
        self.proprio_lags, self.memory_lags = list(proprio_lags), list(memory_lags)
        self.ends = np.cumsum([ep.supervised_count for ep in self.episodes]).tolist()
        self.spans = [(i, s['start'], s['end']) for i, ep in enumerate(self.episodes) for s in ep.valid_segments]

    def __len__(self):
        return self.ends[-1] if self.ends else 0

    def sample(self, rng):
        # Equal intervention probability; long ordinary continuations cannot
        # drown out short, informative recovery segments.
        i, start, end = self.spans[int(rng.integers(len(self.spans)))]
        return self.at(i, int(rng.integers(start, end)))

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        i = bisect_right(self.ends, index)
        anchor = index-(self.ends[i-1] if i else 0)
        return self.at(i, int(self.episodes[i].anchors[anchor]))

    def at(self, i, t):
        import h5py
        from harness.wam_v3 import history_frames
        from harness.robotwin_training_data import FEAT_SHAPE
        from harness.wam_event_clock import episode_event_features
        from harness.zhuhai_precision_v2_training import CLOCK_CONFIG
        ep = self.episodes[i]
        if not ep.mask[t]:
            raise ValueError('Only verified expert anchors are positive labels')
        end = int(ep.segment_end[t])
        feat, pool = self.store.get(ep, t)
        pools = [pool]+[pool if old == t else self.store.get(ep, old)[1]
                        for old in history_frames(t, 0, self.memory_lags)]
        valid = min(self.chunk, end-t)
        actions = np.zeros((self.chunk, 16), np.float32)
        actions[:valid] = ep.action[t:t+valid]
        future, short = t+self.future_offset, t+8
        fv, sv, iv = future <= end, short <= end, t+4 <= end
        sample = dict(state=ep.state[[t]+history_frames(t, 0, self.proprio_lags)].copy(),
            action_sequence=actions, action_valid_mask=np.arange(self.chunk) < valid,
            feats=feat, pool=np.stack(pools, axis=1),
            future_feats=self.store.get(ep, future)[0][:, -1, 5:].copy() if fv else np.zeros((3, 196, 1024), np.uint16),
            future_valid_mask=np.bool_(fv),
            future_feats_8=self.store.get(ep, short)[0][:, -1, 5:].copy() if sv else np.zeros((3, 196, 1024), np.uint16),
            future_valid_mask_8=np.bool_(sv), idm_valid_mask=np.bool_(iv),
            idm_future_feats=np.stack([self.store.get(ep, t+k)[0] for k in range(1, 5)]) if iv else np.zeros((4, *FEAT_SHAPE), np.uint16),
            idm_target_raw=(ep.action[t:t+4]-ep.state[t]).copy() if iv else np.zeros((4, 16), np.float32),
            task_cond=self.condition.copy(), phase=np.int64(-1), active_hand=np.int64(-1), dist=np.float32(0),
            task=self.task, episode=ep.path.name, anchor=t, supervision_source='expert_correction')
        start = max(0, t-math.ceil(CLOCK_CONFIG['max_seconds']/CLOCK_CONFIG['dt'])-2)
        if start == 0:
            with h5py.File(ep.path, 'r') as f:
                initial = f['target_command'][0].copy()
        else:
            initial = ep.action[start-1]
        sample['event_state'] = episode_event_features(ep.state[start:t+1], ep.action[start:t+1],
                                                       initial_target=initial, **CLOCK_CONFIG)[-1].copy()
        return sample


def required_visual_frames(step, memory_lags):
    if not 0 < step <= 2400:
        raise ValueError('Bounded nonempty causal prefix required')
    lag = max(memory_lags, default=0)
    return sorted({0, *range(max(0, step-lag-1), step)})


def limiter_state(limiter):
    return {key: getattr(limiter, key).tolist() for key in ('previous', 'velocity', 'acceleration')}


def restore_policy(policy, episode, step, states, commands, visuals, control):
    """Exact required causal context, without encoding unused ancient RGB.

Only start + latest visual-lag frames need real pools: older pools are never
selected by V3Context.inputs now or later. All measured/command frames replay
into the bounded proprio context and causal event clock. No model action runs.
"""
    if episode != policy.episode or policy.next_step != 0 or policy.context.steps != 0:
        raise ValueError('History restore requires a fresh worker for the same episode')
    q, command = np.asarray(states, np.float32), np.asarray(commands, np.float32)
    if q.shape != (step, 16) or command.shape != q.shape or not np.isfinite(q).all() or not np.isfinite(command).all():
        raise ValueError('Continuous measured/command prefix required')
    required = required_visual_frames(step, policy.context.memory_lags)
    if sorted(visuals) != required:
        raise ValueError('Exact required real visual frames must be supplied')
    pools = {}
    for index in required:
        obs = visuals[index]
        if (not np.array_equal(np.asarray(obs['measured_joint_gripper_state'], np.float32), q[index])
                or not np.array_equal(np.asarray(obs['commanded_joint_gripper_state'], np.float32), command[index])):
            raise ValueError('Visual observation and causal state index differ')
        _, pools[index] = policy.encode(obs['rgb'])
    zero = np.zeros_like(pools[0])
    policy.queue.clear()
    policy.clock.reset()
    for index in range(step):
        policy.context.push(q[index], pools.get(index, zero))
        policy.event_state = policy.clock.update(q[index], command[index])
    if set(control) != {'previous', 'velocity', 'acceleration'}:
        raise ValueError('Full inherited braking state required')
    for key in control:
        values = np.asarray(control[key], np.float64)
        if values.shape != (14,) or not np.isfinite(values).all():
            raise ValueError('Invalid inherited command dynamics')
        setattr(policy.limiter, key, values.copy())
    if (np.any(np.abs(policy.limiter.velocity) > policy.limiter.vmax+1e-9)
            or np.any(np.abs(policy.limiter.acceleration) > policy.limiter.amax+1e-8)
            or np.any(policy.limiter.previous < policy.limiter.lower-1e-10)
            or np.any(policy.limiter.previous > policy.limiter.upper+1e-10)):
        raise ValueError('Inherited braking state exceeds the command limits')
    lo, hi = policy.limiter.feasible_interval()
    if (lo > hi+1e-10).any():
        raise ValueError('Inherited command state is not braking feasible')
    policy.next_step = step
    return dict(next_step=step, causal_frames_replayed=step, real_visual_frames_encoded=required,
                history_preserved=True, cached_actions_discarded=True, braking_state_inherited=True,
                expert_truth_inputs=False)
