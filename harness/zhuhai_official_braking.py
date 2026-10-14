"""Versioned Panda command limits with discrete, recursively feasible braking.

This filters 10Hz position TARGETS, not measured plant dynamics. Original
training/scoring/smoothing files and model checkpoints remain untouched.
"""
from copy import deepcopy
from pathlib import Path
import hashlib
import numpy as np
from harness import zhuhai_action_smoothing as old
from harness import zhuhai_robot_spec as spec

PANDA_ACCELERATION = [15., 7.5, 10., 12.5, 15., 20., 20.]
PANDA_JERK = [7500., 3750., 5000., 6250., 7500., 10000., 10000.]
VERSION = 'panda-official-discrete-braking-v1'
SIGNATURE = 'no feasible braking-preserving Panda command; stop, never violate a limit'
SOURCE_URL = 'https://support.franka.de/docs/control_parameters.html#limits-for-panda'


def protocol():
    settings = deepcopy(old.protocol())
    settings['rate_limit'].update(
        arm_velocity_rad_s=list(spec.VELOCITY_LIMITS)*2,
        arm_acceleration_rad_s2=PANDA_ACCELERATION*2,
        arm_jerk_rad_s3=PANDA_JERK*2,
        source='Panda FCI per-axis position/velocity/acceleration/jerk constants',
        source_url=SOURCE_URL, braking=VERSION,
        braking_model='q_next=q+v_next*dt; future maximum deceleration at each dt; no continuous sqrt approximation',
        scope='10Hz commanded targets; not full FCI or continuous physical stopping certification',
        position_margin_rad=0., measured_state_is_not_command=True)
    settings.update(braking_implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    evaluation_change=VERSION,
                    gripper_events='unchanged clipped [0,1] samples; timing verified separately in physical evaluation')
    return settings


class BrakingRateLimiter(old.PhysicalRateLimiter):
    def __init__(self, settings):
        super().__init__(settings)
        self.jmax = np.asarray(settings['arm_jerk_rad_s3'], dtype=np.float64)
        if (settings.get('braking') != VERSION or self.jmax.shape != (14,)
                or not np.isfinite(self.jmax).all() or (self.jmax <= 0).any()
                or settings.get('position_margin_rad') != 0.):
            raise ValueError('Explicit versioned Panda braking contract required')
        if not np.array_equal(self.amax, PANDA_ACCELERATION*2):
            raise ValueError('Official per-axis Panda acceleration limits required')
        if not np.array_equal(self.vmax, list(spec.VELOCITY_LIMITS)*2):
            raise ValueError('Official per-axis Panda velocity limits required')
        # At this dt, every acceleration-limited target sequence already
        # satisfies the much larger Panda jerk limits; check rather than assume.
        if np.any(2*self.amax/self.dt > self.jmax):
            raise ValueError('Discrete jerk requires an explicit additional constraint')
        self.acceleration = None

    def reset(self, commanded_state):
        super().reset(commanded_state)
        self.acceleration = np.zeros(14)

    def future_stop_distance(self, velocity):
        """Distance AFTER the current target if braking starts next update."""
        speed = np.abs(np.asarray(velocity, dtype=np.float64))
        decrement = self.amax*self.dt
        n = np.maximum(np.ceil(speed/decrement)-1., 0.)
        return self.dt*np.maximum(n*speed-decrement*n*(n+1.)/2., 0.)

    def safe_speed(self, distance):
        """Largest next speed whose next target AND braking tail fit distance.

        For m positive-speed updates, total distance is
        dt*(m*v - a*dt*m*(m-1)/2). Invert every linear segment exactly.
        """
        distance = np.maximum(np.asarray(distance, dtype=np.float64), 0.)
        decrement = self.amax*self.dt
        best = np.zeros(14)
        for m in range(1, int(np.ceil(np.max(self.vmax/decrement)))+1):
            candidate = np.minimum(m*decrement,
                (distance/self.dt + decrement*m*(m-1)/2.)/m)
            valid = candidate >= (m-1)*decrement
            best = np.maximum(best, np.where(valid, candidate, 0.))
        return np.minimum(best, self.vmax)

    def feasible_interval(self):
        if self.previous is None:
            raise ValueError('Reset braking limiter before use')
        lo = np.maximum(-self.vmax, self.velocity-self.amax*self.dt)
        hi = np.minimum(self.vmax, self.velocity+self.amax*self.dt)
        lo = np.maximum(lo, -self.safe_speed(self.previous-self.lower))
        hi = np.minimum(hi, self.safe_speed(self.upper-self.previous))
        return lo, hi

    def step(self, smoothed):
        values = np.asarray(smoothed, dtype=np.float64)
        if self.previous is None or values.shape != (16,) or not np.isfinite(values).all():
            raise ValueError('Reset first; finite native16 command required')
        requested = (values[old.ARM]-self.previous)/self.dt
        lo, hi = self.feasible_interval()
        if (lo > hi+1e-10).any():
            raise ValueError(SIGNATURE)
        velocity = np.clip(requested, lo, hi)
        result = values.copy()
        # Clamp only floating-point roundoff, not an invalid trajectory.
        next_position = self.previous+velocity*self.dt
        if ((next_position < self.lower-1e-10) | (next_position > self.upper+1e-10)).any():
            raise ValueError(SIGNATURE)
        result[old.ARM] = np.clip(next_position, self.lower, self.upper)
        result[old.GRIPPER] = np.clip(result[old.GRIPPER], 0., 1.)
        velocity = (result[old.ARM]-self.previous)/self.dt
        acceleration = (velocity-self.velocity)/self.dt
        jerk = (acceleration-self.acceleration)/self.dt
        stopping = self.future_stop_distance(velocity)
        available = np.where(velocity >= 0, self.upper-result[old.ARM], result[old.ARM]-self.lower)
        if (np.any(np.abs(velocity) > self.vmax+1e-9)
                or np.any(np.abs(acceleration) > self.amax+1e-8)
                or np.any(np.abs(jerk) > self.jmax+1e-7)
                or np.any(stopping > available+1e-10)):
            raise ValueError(SIGNATURE)
        braking = ((requested < -self.safe_speed(self.previous-self.lower)-1e-10)
                   | (requested > self.safe_speed(self.upper-self.previous)+1e-10))
        delta = result[old.ARM]-values[old.ARM]
        stats = dict(action_dim=16, state_dim=16, state_semantics='measured_joint_gripper',
            action_semantics='joint_position_target', gripper_indices=[7, 15], braking_version=VERSION,
            max_abs_velocity_rad_s=float(np.abs(velocity).max()),
            max_abs_acceleration_rad_s2=float(np.abs(acceleration).max()),
            max_abs_jerk_rad_s3=float(np.abs(jerk).max()),
            arm_limit_delta_rms_rad=float(np.sqrt(np.mean(delta**2))),
            arm_limit_delta_max_rad=float(np.abs(delta).max()),
            limited_arm_dimensions=int((np.abs(delta)>1e-10).sum()),
            braking_limited_dimensions=int(braking.sum()),
            braking_tail_distance_rad=stopping.tolist(),
            braking_available_distance_rad=available.tolist(),
            minimum_braking_surplus_rad=float((available-stopping).min()),
            gripper_physical_range_clipped=bool(np.any(result[old.GRIPPER] != values[old.GRIPPER])),
            gripper_threshold_event_preserved=bool(np.array_equal(result[old.GRIPPER]>=.5, values[old.GRIPPER]>=.5)))
        self.previous, self.velocity, self.acceleration = result[old.ARM].copy(), velocity, acceleration
        return result, stats


def install_braking(policy, settings):
    """Keep the original clock, history, spline, exec8 and gripper events."""
    policy.limiter = BrakingRateLimiter(settings['rate_limit'])
    return policy
