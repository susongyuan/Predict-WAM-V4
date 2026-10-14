"""LiLa-WAM B-spline inference smoothing and causal physical rate limits.

Apply the upstream spline exactly once to a denormalized 32-frame chunk, before
choosing the first eight frames. Preserve both gripper dimensions to avoid
shifting opening/closing events. Rate limits are applied only when a command is
actually dequeued, with state preserved across replanning boundaries.
"""
from __future__ import annotations

import ast
from collections import deque
import hashlib
import json
from pathlib import Path

import numpy as np

from harness import zhuhai_robot_spec as robot_spec
from harness.zhuhai_demo_recording import data_contract, data_contract_hash, STATE_SEMANTICS, ACTION_SEMANTICS

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / 'third_party/LiLa-WAM/robotwin_infer.py'
ACTION_DIM = STATE_DIM = robot_spec.ROBOT_DOF
ARM = np.asarray(robot_spec.ARM_INDICES, dtype=np.int64)
GRIPPER = np.asarray(robot_spec.GRIPPER_INDICES, dtype=np.int64)
if ACTION_DIM != 16 or ARM.tolist() != [*range(7), *range(8, 15)] or GRIPPER.tolist() != [7, 15]:
    raise ValueError('Smoothing is exclusively for native16 dual-Panda, never legacy14')


def bspline_smooth(action_seq: np.ndarray, degree: int = 3, num_ctrl_pts: int = 8) -> np.ndarray:
    """Exact upstream least-squares spline, including its internal-knot rule."""
    from scipy.interpolate import make_lsq_spline
    N, D = action_seq.shape
    if N <= num_ctrl_pts:
        return action_seq
    x = np.arange(N)
    num_internal_knots = num_ctrl_pts - degree
    internal_knots = np.linspace(0, N - 1, num_internal_knots + 2)[1:-1]
    knots = np.concatenate([[0] * (degree + 1), internal_knots, [N - 1] * (degree + 1)])
    spline = make_lsq_spline(x, action_seq, knots, k=degree)
    return spline(x)


def protocol():
    source = UPSTREAM.read_text()
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'bspline_smooth')
    body = ast.get_source_segment(source, node)
    robot = robot_spec.robot_contract()
    bounds = np.asarray(robot_spec.DUAL_JOINT_LIMITS, dtype=np.float64)
    return dict(enabled=True, type='LiLa-WAM_least_squares_cubic_B_spline', degree=3,
        action_dim=16, state_dim=16, state_semantics=STATE_SEMANTICS, action_semantics=ACTION_SEMANTICS,
        robot_contract=robot, robot_contract_sha256=protocol_hash(robot),
        data_contract=data_contract(), data_contract_sha256=data_contract_hash(),
        worker_scipy_version='1.15.3',
        num_ctrl_pts=8, internal_knots=5, actual_coefficients=9,
        upstream_source=str(UPSTREAM), upstream_file_sha256=hashlib.sha256(source.encode()).hexdigest(),
        upstream_function_sha256=hashlib.sha256(body.encode()).hexdigest(),
        implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        smoothing_stage='after physical denormalization, before first-eight selection; once per chunk',
        smoothed_dimensions=ARM.tolist(), preserved_gripper_dimensions=GRIPPER.tolist(),
        deviation='upstream smooths every action dimension; this run preserves gripper opening/closing samples',
        smooth_sigma=1.0, smooth_sigma_role='upstream YAML field is not used by bspline_smooth; no Gaussian second smoothing',
        rate_limit=dict(frequency_hz=10., action_dim=16, state_dim=16,
                        arm_velocity_rad_s=[min(1.2, v) for v in robot_spec.VELOCITY_LIMITS]*2,
                        arm_acceleration_rad_s2=[4.] * len(ARM),
                        position_lower_rad=bounds[:, 0].tolist(), position_upper_rad=bounds[:, 1].tolist(),
                        source='standard Panda position/velocity bounds; conservative 1.2rad/s and experimental 4rad/s^2 command limits',
                        reset_source='commanded_joint_gripper_state_only_for_target_continuity',
                        scope='commanded joint targets; measured dynamic tracking must be audited separately',
                        across_chunk=True, gripper_rate_limit=False))


def protocol_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def smooth_chunk(raw, settings):
    array = np.asarray(raw, dtype=np.float64)
    if array.shape != (32, ACTION_DIM) or not np.isfinite(array).all():
        raise ValueError('finite denormalized 32x16 chunk required; legacy14 is forbidden')
    if settings.get('action_dim') != 16 or settings.get('state_dim') != 16 or settings.get('preserved_gripper_dimensions') != [7, 15]:
        raise ValueError('Native16 smoothing settings required; old14 protocol cannot be reused')
    result = array.copy()
    if settings['enabled']:
        result[:, ARM] = bspline_smooth(array[:, ARM], settings['degree'], settings['num_ctrl_pts'])
    if not np.array_equal(result[:, GRIPPER], array[:, GRIPPER]):
        raise ValueError('gripper event samples changed')
    return result


class PhysicalRateLimiter:
    def __init__(self, settings):
        self.settings = settings
        if settings.get('action_dim') != 16 or settings.get('state_dim') != 16 or settings.get('frequency_hz') != 10.:
            raise ValueError('Native16 physical rate-limit contract must explicitly declare 16D at 10Hz')
        self.dt = 1 / float(settings['frequency_hz'])
        self.vmax = np.asarray(settings['arm_velocity_rad_s'], dtype=np.float64)
        self.amax = np.asarray(settings['arm_acceleration_rad_s2'], dtype=np.float64)
        self.lower = np.asarray(settings['position_lower_rad'], dtype=np.float64)
        self.upper = np.asarray(settings['position_upper_rad'], dtype=np.float64)
        if any(x.shape != (len(ARM),) or not np.isfinite(x).all() for x in (self.vmax, self.amax, self.lower, self.upper)):
            raise ValueError('14 finite declared arm limits required for two 7D arms')
        if (self.vmax <= 0).any() or (self.amax <= 0).any() or (self.lower >= self.upper).any():
            raise ValueError('invalid physical command limits')
        standard = np.asarray(robot_spec.DUAL_JOINT_LIMITS, dtype=np.float64)
        if not np.array_equal(self.lower, standard[:, 0]) or not np.array_equal(self.upper, standard[:, 1]):
            raise ValueError('Position limits must match official dual-Panda spec, never old ±10rad limits')
        if np.any(self.vmax > np.tile(robot_spec.VELOCITY_LIMITS, 2)):
            raise ValueError('Command velocity cap cannot exceed official joint limits')
        self.previous = self.velocity = None

    def reset(self, commanded_state):
        values = np.asarray(commanded_state, dtype=np.float64)
        if values.shape != (ACTION_DIM,) or not np.isfinite(values).all():
            raise ValueError('finite native16 target state required to initialize command continuity')
        robot_spec.validate_action(values)
        self.previous = values[ARM].copy()
        if (self.previous < self.lower).any() or (self.previous > self.upper).any():
            raise ValueError('initial commanded joints outside physical range')
        self.velocity = np.zeros(len(ARM))

    def step(self, smoothed):
        values = np.asarray(smoothed, dtype=np.float64)
        if self.previous is None or values.shape != (ACTION_DIM,) or not np.isfinite(values).all():
            raise ValueError('reset limiter first; finite native16 command required')
        requested_v = (values[ARM] - self.previous) / self.dt
        v_lo = np.maximum(-self.vmax, self.velocity - self.amax * self.dt)
        v_hi = np.minimum(self.vmax, self.velocity + self.amax * self.dt)
        v_lo = np.maximum(v_lo, (self.lower - self.previous) / self.dt)
        v_hi = np.minimum(v_hi, (self.upper - self.previous) / self.dt)
        if (v_lo > v_hi + 1e-10).any():
            raise ValueError('no feasible simultaneous position/velocity/acceleration command; stop, never violate a limit')
        velocity = np.clip(requested_v, v_lo, v_hi)
        result = values.copy()
        result[ARM] = self.previous + velocity * self.dt
        result[GRIPPER] = np.clip(result[GRIPPER], 0., 1.)
        acceleration = (velocity - self.velocity) / self.dt
        stats = dict(action_dim=16, state_dim=16, state_semantics=STATE_SEMANTICS,
                     action_semantics=ACTION_SEMANTICS, gripper_indices=GRIPPER.tolist(),
                     max_abs_velocity_rad_s=float(np.max(np.abs(velocity))),
                     max_abs_acceleration_rad_s2=float(np.max(np.abs(acceleration))),
                     arm_limit_delta_rms_rad=float(np.sqrt(np.mean((result[ARM] - values[ARM]) ** 2))),
                     arm_limit_delta_max_rad=float(np.max(np.abs(result[ARM] - values[ARM]))),
                     limited_arm_dimensions=int(np.sum(np.abs(result[ARM] - values[ARM]) > 1e-10)),
                     gripper_physical_range_clipped=bool(np.any(result[GRIPPER] != values[GRIPPER])),
                     gripper_threshold_event_preserved=bool(np.array_equal(result[GRIPPER] >= .5, values[GRIPPER] >= .5)))
        self.previous, self.velocity = result[ARM].copy(), velocity
        return result, stats


class SmoothedChunkPolicy:
    """Same bounded observation history and execution prefix as ChunkPolicy."""
    def __init__(self, context, encode, sample, settings):
        self.context, self.encode, self.sample, self.settings = context, encode, sample, settings
        self.limiter = PhysicalRateLimiter(settings['rate_limit'])
        self.queue = deque()
        self.episode = None
        self.next_step = 0
        self.runtime = None

    def reset(self, episode, seed):
        from harness.robotwin_policy_worker import ids
        ids(episode, 0)
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError('invalid episode seed')
        self.context.reset()
        self.queue.clear()
        self.limiter.previous = self.limiter.velocity = None
        self.episode, self.seed, self.next_step = episode, seed, 0

    def clear_actions(self, episode, next_step, reason):
        if episode != self.episode or next_step != self.next_step or not reason:
            raise ValueError('invalid queue clear request')
        discarded = len(self.queue)
        self.queue.clear()
        return dict(discarded_actions=discarded, history_preserved=True, next_step=self.next_step)

    def observe(self, episode, step, observation):
        if episode != self.episode or step != self.next_step:
            raise ValueError('out-of-order observation')
        measured = np.asarray(observation.get('measured_joint_gripper_state'), dtype=np.float64)
        commanded = np.asarray(observation.get('commanded_joint_gripper_state'), dtype=np.float64)
        if measured.shape != (STATE_DIM,) or commanded.shape != (ACTION_DIM,) or not np.isfinite(measured).all() or not np.isfinite(commanded).all():
            raise ValueError('Policy requires explicit measured native16 state and separate target command')
        if set(observation['rgb']) != set(data_contract()['camera_names']):
            raise ValueError('Only the three onboard camera inputs are allowed')
        if self.limiter.previous is None:
            self.limiter.reset(commanded)
        features, pool = self.encode(observation['rgb'])
        self.context.push(measured, pool)
        replanned = not self.queue
        extra = {}
        if replanned:
            raw = np.asarray(self.sample(features, self.context.inputs(), self.seed, step), dtype=np.float64)
            smoothed = smooth_chunk(raw, self.settings)
            self.queue.extend((raw[i].copy(), smoothed[i].copy()) for i in range(8))
            extra = dict(raw_denormalized_chunk=raw.tolist(), smoothed_denormalized_chunk=smoothed.tolist(),
                chunk_smooth_delta_rms_rad=float(np.sqrt(np.mean((smoothed[:, ARM] - raw[:, ARM]) ** 2))),
                chunk_smooth_delta_max_rad=float(np.max(np.abs(smoothed[:, ARM] - raw[:, ARM]))))
        raw, smoothed = self.queue.popleft()
        limited, stats = self.limiter.step(smoothed)
        self.next_step += 1
        return dict(action_dim=16, state_dim=16, state_semantics=STATE_SEMANTICS,
            action_semantics=ACTION_SEMANTICS, raw_action=limited.tolist(), model_raw_action=raw.tolist(), smoothed_action=smoothed.tolist(),
            limited_action=limited.tolist(), rate_limit_stats=stats,
            smooth_delta_rms_rad=float(np.sqrt(np.mean((smoothed[ARM] - raw[ARM]) ** 2))),
            replanned=replanned, history_steps=self.next_step, queue_remaining=len(self.queue),
            execution_prefix=8, execution_reason='LiLa_spline_exec8_then_causal_cross_chunk_rate_limit', **extra)
