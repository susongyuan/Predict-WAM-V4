"""Independent arm-continuity overlay; frozen evaluators never import this.

Keep the original 32-frame spline, model, exec8 and gripper event samples.
Anchor each arm reference at the last dispatched position and velocity, then
send officially limited joint targets at every 4ms physics step. No task GT.
"""
from copy import deepcopy
import hashlib
from pathlib import Path
from types import MethodType

import numpy as np
from scipy.interpolate import CubicSpline

from harness import zhuhai_action_smoothing as spline
from harness import zhuhai_official_braking as official

VERSION = 'precision-v2-rtc-soft-overlap-c1-fine250-v37'
SIGNATURE = official.SIGNATURE
ARM, GRIPPER = spline.ARM, spline.GRIPPER
SUBSTEPS, DT, BRIDGE_SECONDS = 25, .004, .4
ACCELERATION_INNER_MARGIN = 1e-6


class FineBrakingLimiter(official.BrakingRateLimiter):
    def __init__(self, settings):
        if settings.get('frequency_hz') != 250.:
            raise ValueError('explicit250Hz fine-dispatch contract required')
        # Frozen parent validates native16 bounds at its original10Hz cadence.
        validated = deepcopy(settings)
        validated['frequency_hz'] = 10.
        super().__init__(validated)
        self.settings = settings
        self.dt = DT
        if np.any(2*self.amax/self.dt > self.jmax):
            raise ValueError('fine command cadence would exceed official jerk limits')

    def feasible_interval(self):
        # Reserve1e-6rad/s² INSIDE the official bound: subtracting two nearby
        # 4ms positions otherwise amplifies float roundoff beyond the old
        # guard's1e-8 tolerance. Official checks and braking remain unchanged.
        if self.previous is None:
            raise ValueError('reset before fine commands')
        decrement=(self.amax-ACCELERATION_INNER_MARGIN)*self.dt
        lo=np.maximum(-self.vmax,self.velocity-decrement)
        hi=np.minimum(self.vmax,self.velocity+decrement)
        return (np.maximum(lo,-self.safe_speed(self.previous-self.lower)),
                np.minimum(hi,self.safe_speed(self.upper-self.previous)))

    def safe_speed(self, distance):
        # Exact vectorization of the original per-segment braking inversion.
        distance = np.maximum(np.asarray(distance, dtype=np.float64), 0.)
        decrement = (self.amax-ACCELERATION_INNER_MARGIN)*self.dt
        m = np.arange(1, int(np.ceil(np.max(self.vmax/decrement)))+1)[:, None]
        candidate = np.minimum(m*decrement,
            (distance/self.dt+decrement*m*(m-1)/2.)/m)
        best = np.max(np.where(candidate >= (m-1)*decrement, candidate, 0.), axis=0)
        return np.minimum(best, self.vmax)

    def future_stop_distance(self, velocity):
        speed=np.abs(np.asarray(velocity,dtype=float))
        decrement=(self.amax-ACCELERATION_INNER_MARGIN)*self.dt
        n=np.maximum(np.ceil(speed/decrement)-1.,0.)
        return self.dt*np.maximum(n*speed-decrement*n*(n+1.)/2.,0.)


def protocol():
    settings = deepcopy(official.protocol())
    settings.update(continuity_version=VERSION,
        continuity_implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        continuous_reference=dict(type='arm_only_C1_Hermite_correction_on_original_chunk_spline',
            position_anchor='last_dispatched_target', velocity_anchor='last_dispatched_target_velocity',
            bridge_seconds=BRIDGE_SECONDS, target_sample_hz=250, observations_hz=10,
            gripper_events='original raw samples clipped[0,1], held25 substeps, no averaging',
            time_alignment='original action index0 is target at+0.1s; first fine sample at+0.004s',
            no_prefix_training=True, full_asynchronous_RTC=False,soft_overlap_guidance=True, no_task_state_intervention=True))
    settings['overlap_guidance'] = dict(source='https://arxiv.org/html/2506.07339v1',
        implementation_sha256=hashlib.sha256((Path(__file__).parent/'zhuhai_overlap_guidance_v37.py').read_bytes()).hexdigest(),
        synchronous=True, physical_delay_actions=0, old_indices=[8,32], new_indices=[0,24],
        guidance='exact input VJP of denoised estimate; paper Eq2-5 soft mask',
        beta=5., denoising_steps=10, checkpoint_modified=False, task_progress_oracle=False)
    settings['fine_rate_limit'] = deepcopy(settings['rate_limit'])
    settings['fine_rate_limit'].update(frequency_hz=250.,
        acceleration_inner_numerical_margin_rad_s2=ACCELERATION_INNER_MARGIN,
        scope='all actually dispatched4ms position targets; same official per-axis limits and discrete braking')
    return settings


def reference(chunk, position, velocity):
    """Return800 fine reference samples and32 original-time samples."""
    chunk = np.asarray(chunk,dtype=float)
    position, velocity = np.asarray(position,dtype=float), np.asarray(velocity,dtype=float)
    if (chunk.shape != (32,16) or not np.isfinite(chunk).all()
            or position.shape != (14,) or velocity.shape != (14,)
            or not np.isfinite(position).all() or not np.isfinite(velocity).all()):
        raise ValueError('finite native16 chunk required')
    curve = CubicSpline((np.arange(32)+1)*.1, chunk[:,ARM], axis=0)
    time = (np.arange(32*SUBSTEPS)+1)*DT
    u = np.minimum(time/BRIDGE_SECONDS, 1.)
    dq, dv = position-curve(0.), velocity-curve(0.,1)
    fine = np.repeat(chunk, SUBSTEPS, axis=0)
    fine[:,ARM] = curve(time)+(2*u**3-3*u**2+1)[:,None]*dq + (
        BRIDGE_SECONDS*(u**3-2*u**2+u))[:,None]*dv
    if not np.array_equal(fine[:,GRIPPER], np.repeat(chunk[:,GRIPPER],SUBSTEPS,axis=0)):
        raise ValueError('gripper event shifted')
    return fine, fine[SUBSTEPS-1::SUBSTEPS].copy()


def install_braking(policy, settings):
    if settings != protocol():
        raise ValueError('explicit frozen continuity protocol required')
    policy.limiter = FineBrakingLimiter(settings['fine_rate_limit'])

    def observe(self, episode, step, observation):
        if episode != self.episode or step != self.next_step:
            raise ValueError('out-of-order continuous observation')
        from scripts.eval_zhuhai_wam_precision_v2 import base
        base.validate_observation(observation)
        measured = np.asarray(observation['measured_joint_gripper_state'],dtype=float)
        commanded = np.asarray(observation['commanded_joint_gripper_state'],dtype=float)
        self.event_state = self.clock.update(measured, commanded)
        if self.limiter.previous is None:
            self.limiter.reset(commanded)
        elif not np.allclose(commanded[ARM],self.limiter.previous,rtol=0,atol=3e-7):
            raise ValueError('actual dispatched endpoint differs from reference controller')
        features, pool = self.encode(observation['rgb'])
        self.context.push(measured,pool)
        replanned = not self.queue
        extra = {}
        if replanned:
            raw = np.asarray(self.sample(features,self.context.inputs(),self.seed,step),dtype=float)
            original_spline = spline.smooth_chunk(raw,self.settings)
            anchor_q, anchor_v = self.limiter.previous.copy(), self.limiter.velocity.copy()
            fine, smoothed = reference(original_spline,anchor_q,anchor_v)
            self.queue.extend((raw[i].copy(),original_spline[i].copy(),smoothed[i].copy(),
                fine[i*SUBSTEPS:(i+1)*SUBSTEPS].copy()) for i in range(8))
            extra = dict(overlap_guidance=self.overlap_sampler.last_diagnostic,raw_denormalized_chunk=raw.tolist(),
                bspline_denormalized_chunk=original_spline.tolist(),
                smoothed_denormalized_chunk=smoothed.tolist(),
                continuity_anchor_q_rad=anchor_q.tolist(),continuity_anchor_v_rad_s=anchor_v.tolist())
        raw, original_spline, smoothed, requested = self.queue.popleft()
        targets, statistics = [], []
        for target in requested:
            command, stats = self.limiter.step(target)
            targets.append(command)
            statistics.append(stats)
        targets = np.stack(targets)
        stats = dict(statistics[-1])
        for key in ('max_abs_velocity_rad_s','max_abs_acceleration_rad_s2','max_abs_jerk_rad_s3'):
            stats[key] = max(item[key] for item in statistics)
        stats['minimum_braking_surplus_rad'] = min(item['minimum_braking_surplus_rad'] for item in statistics)
        self.next_step += 1
        return dict(action_dim=16,state_dim=16,state_semantics=spline.STATE_SEMANTICS,
            action_semantics=spline.ACTION_SEMANTICS,raw_action=targets[-1].tolist(),
            model_raw_action=raw.tolist(),bspline_action=original_spline.tolist(),
            smoothed_action=smoothed.tolist(),limited_action=targets[-1].tolist(),
            rate_limit_stats=stats,replanned=replanned,history_steps=self.next_step,
            queue_remaining=len(self.queue),execution_prefix=8,
            execution_reason='RTC_soft_overlap_generation_then_original_spline_C1_and_official_fine250_braking',
            smooth_delta_rms_rad=float(np.sqrt(np.mean((original_spline[ARM]-raw[ARM])**2))),
            continuity_delta_rms_rad=float(np.sqrt(np.mean((smoothed[ARM]-original_spline[ARM])**2))),
            substep_requested_targets=requested.tolist(),substep_targets=targets.tolist(),
            substep_dt_seconds=DT,substep_count=SUBSTEPS,**extra)

    policy.observe = MethodType(observe,policy)
    return policy


PENDING = None


def set_pending(reply):
    global PENDING
    if PENDING is not None:
        raise ValueError('unconsumed dispatch packet')
    values = np.asarray(reply['substep_targets'],dtype=float)
    if (values.shape != (SUBSTEPS,16) or not np.isfinite(values).all()
            or reply['substep_dt_seconds'] != DT or reply['substep_count'] != SUBSTEPS
            or not np.array_equal(values[-1],reply['limited_action'])):
        raise ValueError('invalid4ms dispatch packet')
    if not np.array_equal(values[:,GRIPPER],np.repeat(values[-1:,GRIPPER],SUBSTEPS,axis=0)):
        raise ValueError('gripper event changed inside control interval')
    PENDING = values


def install_environment():
    from harness import isaac_zhuhai_runtime as runtime
    OriginalEnv = runtime.IsaacZhuhaiEnv

    class ContinuousEnv(OriginalEnv):
        def step_action(self, action, interval_seconds=.1):
            global PENDING
            targets = PENDING
            if targets is None or interval_seconds != .1 or not np.array_equal(targets[-1],action):
                raise ValueError('driver must consume precisely the bound fine command packet')
            if self.physics_dt != DT:
                raise ValueError('continuous dispatch requires original4ms physics cadence')
            PENDING = None
            for target in targets:
                runtime.validate_action(target,self.joint_limits)
                self.command = target.copy()
                self._drive(target)
                self.world.step(render=False,update_fabric=True)
                self.physics_time += self.physics_dt
                self._audit_physics_contacts()
                self.judge.observe(self.physics_dt,self.body_states(),list(self._contact_pairs.values()))
            if self.on_control_step is not None:
                self.on_control_step(self)
            return self.rule_state()

    runtime.IsaacZhuhaiEnv = ContinuousEnv
