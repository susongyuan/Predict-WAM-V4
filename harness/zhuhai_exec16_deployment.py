"""Explicit exec16 deployment over frozen 32-step policies and original braking.

Sixteen is the execution horizon, independently of the native16 robot vector.
Model/training contracts retain their original exec8 provenance unchanged.
"""
from copy import deepcopy
import hashlib
from pathlib import Path
import numpy as np

from harness import zhuhai_action_smoothing as smoothing
from harness import zhuhai_official_braking as braking
from harness.wam_event_clock import EventClock
from harness.zhuhai_precision_v2_training import CLOCK_CONFIG

DEPLOYMENT_VERSION = 'precision-wam-chunk32-exec16-official-braking-v1'
EXECUTION_PREFIX = 16


def deployment_protocol():
    return dict(version=DEPLOYMENT_VERSION, execution_prefix=16, training_execution_prefix=8,
        action_chunk=32, euler_steps=10, frequency_hz=10, replan_interval_seconds=1.6,
        action_dim=16, state_dim=16, checkpoint_modified=False,
        change='execute first16 of the existing full32 spline; replan every16 observations',
        unchanged=['model/weights', '32-step flow objective', '10 Euler', 'full32 spline',
                   'gripper samples', 'official per-step physical braking', '10Hz causal history/clock',
                   'original scoring', 'three onboard cameras'],
        future_policy_inputs=False, offline_rule_inputs=False,
        implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def protocol():
    settings = deepcopy(braking.protocol())
    settings.update(smoothing_stage='after physical denormalization, before first-sixteen selection; once per full32 chunk',
        execution_prefix=16, training_execution_prefix=8, deployment_version=DEPLOYMENT_VERSION,
        deployment=deployment_protocol(), deployment_sha256=smoothing.protocol_hash(deployment_protocol()))
    # rate_limit is deliberately the unchanged original official braking contract.
    return settings


def overlay(frozen):
    result = deepcopy(frozen)
    result['action_postprocessing'] = protocol()
    result['action_postprocessing_sha256'] = smoothing.protocol_hash(result['action_postprocessing'])
    result['evaluation_overlay'] = dict(version=DEPLOYMENT_VERSION,
        training_contract_sha256=smoothing.protocol_hash(frozen), checkpoint_modified=False,
        training_execution_prefix=8, execution_prefix=16,
        changes=['official Panda braking overlay', 'exec8 to exec16 deployment only'],
        unchanged=['checkpoint contents', 'scene/scoring', 'seeds', 'full32 spline', 'event clock', 'gripper samples'])
    result['deployment'] = deployment_protocol()
    result['deployment_sha256'] = smoothing.protocol_hash(result['deployment'])
    return result


class Exec16SmoothedChunkPolicy(smoothing.SmoothedChunkPolicy):
    """Every control observation updates history/clock; only sampling is every16."""
    def __init__(self, context, encode, sample, settings):
        if settings != protocol():
            raise ValueError('Exact explicit exec16 deployment protocol required')
        self.clock = EventClock(**CLOCK_CONFIG)
        self.event_state = None
        self.sample_with_clock = sample
        super().__init__(context, encode,
            lambda feats, inputs, seed, step: sample(feats, inputs, seed, step, self.event_state), settings)
        braking.install_braking(self, settings)

    def reset(self, episode, seed):
        super().reset(episode, seed)
        self.clock.reset(); self.event_state = None

    def observe(self, episode, step, observation):
        if episode != self.episode or step != self.next_step:
            raise ValueError('out-of-order observation')
        from scripts.eval_zhuhai_wam_hjepa import validate_observation
        observed = validate_observation(observation)
        measured = observed['measured_joint_gripper_state']
        commanded = observed['commanded_joint_gripper_state']
        self.event_state = self.clock.update(measured, commanded)
        if self.limiter.previous is None:
            self.limiter.reset(commanded)
        features, pool = self.encode(observed['rgb'])
        self.context.push(measured, pool)
        replanned = not self.queue
        extra = {}
        if replanned:
            raw = np.asarray(self.sample(features, self.context.inputs(), self.seed, step), dtype=np.float64)
            smoothed = smoothing.smooth_chunk(raw, self.settings)
            self.queue.extend((raw[i].copy(), smoothed[i].copy()) for i in range(EXECUTION_PREFIX))
            extra = dict(raw_denormalized_chunk=raw.tolist(), smoothed_denormalized_chunk=smoothed.tolist(),
                chunk_smooth_delta_rms_rad=float(np.sqrt(np.mean((smoothed[:, smoothing.ARM] - raw[:, smoothing.ARM])**2))),
                chunk_smooth_delta_max_rad=float(np.max(np.abs(smoothed[:, smoothing.ARM] - raw[:, smoothing.ARM]))))
        raw, smoothed = self.queue.popleft()
        limited, stats = self.limiter.step(smoothed)
        self.next_step += 1
        return dict(action_dim=16, state_dim=16, state_semantics=smoothing.STATE_SEMANTICS,
            action_semantics=smoothing.ACTION_SEMANTICS, raw_action=limited.tolist(), model_raw_action=raw.tolist(),
            smoothed_action=smoothed.tolist(), limited_action=limited.tolist(), rate_limit_stats=stats,
            smooth_delta_rms_rad=float(np.sqrt(np.mean((smoothed[smoothing.ARM] - raw[smoothing.ARM])**2))),
            replanned=replanned, history_steps=self.next_step, queue_remaining=len(self.queue),
            execution_prefix=16, training_execution_prefix=8,
            deployment_version=DEPLOYMENT_VERSION,
            execution_reason='full32_original_spline_exec16_then_original_perstep_official_braking', **extra)
