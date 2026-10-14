"""Transparent single-threaded native14 adapter; NOT a safety/completion verifier.
Driver(raw_action) must apply existing bounds, dispatch once, and return bool ACK.
ACK means consumed, not achieved. Only reset after a real NEW environment episode.
No hold, queue clearing, history reset on error, condition injection or memory writes.
"""
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, is_dataclass
import re
import time
import numpy as np
from harness.robotwin_policy_worker import VIEWS, check_reply, vector


class CapabilityUnavailable(RuntimeError):
    pass


def reject_gt(value):
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, Mapping):
        for key, item in value.items():
            name = re.sub(r'([a-z])([A-Z])', r'\1_\2', str(key)).lower()
            if re.search(r'(^|[^a-z])(gt|ground_truth|reward|eval|native|simulator|privileged|privbase|bddl|actor_pose|object_state|target_pose|arm_tag|object_pose_world|cabinet_joint_qpos|third_view)([^a-z]|$)', name):
                raise ValueError('GT/evaluator field forbidden: ' + str(key))
            if name in ('source', 'verifier_kind') and re.search(r'gt|ground.?truth|simulator|native|reward|eval|privileged|privbase|bddl', str(item), re.I):
                raise ValueError('GT/evaluator provenance forbidden')
            reject_gt(item)
    elif isinstance(value, np.ndarray) and value.dtype.kind == "O":
        reject_gt(value.tolist())
    elif isinstance(value, (list, tuple)):
        for item in value:
            reject_gt(item)


class RobotwinSystemAdapter:
    def __init__(self, client, driver, *, max_age_s, clock=time.monotonic):
        if not np.isfinite(max_age_s) or max_age_s <= 0:
            raise ValueError('positive finite max_age_s required')
        self.client, self.driver, self.clock, self.max_age = client, driver, clock, max_age_s
        self.episode, self.next_step, self.pending = None, 0, None
        self.episodes, self.observations, self.failed = set(), set(), True

    def reset(self, episode, seed):
        if not isinstance(episode, str) or not episode or episode in self.episodes or self.pending is not None:
            raise ValueError('new episode required; cannot reset pending transaction')
        self.episodes.add(episode)
        self.failed = True
        reply = self.client.reset(episode, seed)
        check_reply(reply, 'reset_ok', episode, 0)
        self.episode, self.next_step, self.failed = episode, 0, False
        return reply

    def prepare(self, episode, step, observation_id, observed_at, observation):
        if self.failed or self.pending is not None or episode != self.episode or type(step) is not int or step != self.next_step:
            raise ValueError('inactive, pending or wrong episode/step')
        if not isinstance(observation_id, str) or not observation_id or (episode, observation_id) in self.observations:
            raise ValueError('fresh observation ID required')
        self.failed = True  # Any validation/IPC failure requires termination, never retry.
        if not np.isfinite(observed_at) or not 0 <= self.clock() - observed_at <= self.max_age:
            raise ValueError('stale/future observation')
        reject_gt(observation)
        if set(observation) != {'rgb', 'commanded_joint_gripper_state'} or set(observation['rgb']) != set(VIEWS):
            raise ValueError('three RGB + commanded state only; no actual-state substitution')
        vector(observation['commanded_joint_gripper_state'], 14, 'commanded state')
        for image in observation['rgb'].values():
            a = np.asarray(image)
            if a.dtype != np.uint8 or a.ndim != 3 or a.shape[2] != 3 or min(a.shape[:2]) <= 0:
                raise ValueError('native uint8 HWC3 RGB required')
        reply = self.client.action(episode, step, observation)
        check_reply(reply, 'action', episode, step)
        action = vector(reply['raw_action'], 14, 'raw action').tolist()
        self.pending = ((episode, step, observation_id), observed_at, action)
        self.observations.add((episode, observation_id))
        self.next_step += 1
        self.failed = False
        return deepcopy(reply)

    def execute(self, episode, step, observation_id):
        if self.failed or type(step) is not int or self.pending is None or self.pending[0] != (episode, step, observation_id):
            raise ValueError('wrong or already consumed transaction')
        _, captured, action = self.pending
        self.pending, self.failed = None, True  # Burn before driver call, including exceptions.
        if not 0 <= self.clock() - captured <= self.max_age:
            raise ValueError('expired before dispatch')
        ack = self.driver(action)
        if type(ack) is not bool or not ack:
            raise RuntimeError('dispatch rejected/unknown; no automatic retry')
        self.failed = False
        return ack

    def record_visible_outcome(self, outcome=None):
        reject_gt(outcome)
        return None  # Unknown: no independent verifier/admission or memory integration here.

    def replan(self, reason="independently_requested_reobserve"):
        if self.pending is not None:
            self.pending, self.failed = None, True
            raise ValueError('unacknowledged transaction revoked; terminate without resetting history')
        if self.failed or self.episode is None:
            raise ValueError('cannot replan an inactive or failed episode')
        clear = getattr(self.client, 'clear_actions', None)
        if not callable(clear):
            self.failed = True
            raise CapabilityUnavailable('worker clear/replan unavailable; do not reset history')
        self.failed = True
        reply = clear(self.episode, self.next_step, reason)
        check_reply(reply, 'cleared', self.episode, self.next_step)
        if reply.get('history_preserved') is not True or reply.get('next_step') != self.next_step:
            raise RuntimeError('history-preserving queue invalidation not acknowledged')
        self.failed = False
        return deepcopy(reply)
