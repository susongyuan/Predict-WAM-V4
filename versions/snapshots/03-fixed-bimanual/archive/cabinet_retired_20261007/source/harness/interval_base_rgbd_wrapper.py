"""Opt-in R1Pro HeadRGBD wrapper with interval encoder-derived base velocity.

Only base_qvel (first three of the existing 61D proprio) changes. We use the
robot's six virtual-base joint encoders and the simulator control clock, never
world/object poses. This is a disclosed simulated-proprioception adapter, not
real wheel-odometry validation or the untouched official observation protocol.
"""
import numpy as np
import omnigibson as og

from harness.base_encoder_velocity import interval_base_velocity, replace_base_velocity
from harness.head_rgbd_wrapper import HeadRGBDWrapper


class IntervalBaseRGBDWrapper(HeadRGBDWrapper):
    def __init__(self, env):
        super().__init__(env)
        self._velocity_robot = env.robots[0]
        if not self._velocity_robot.is_holonomic_base:
            raise ValueError('interval base adapter requires a holonomic encoder base')
        self._interval_velocity = None
        self._interval_end = None

    def _encoders(self):
        return self._velocity_robot.get_joint_positions()[self._velocity_robot.base_idx].detach().cpu().numpy().copy()

    def _patch(self, obs):
        # No invented zero velocity at reset, and no reuse after an untracked physics step.
        if self._interval_velocity is None or abs(float(og.sim.current_time)-self._interval_end) > 1e-7:
            return obs
        return replace_base_velocity(obs, self._velocity_robot.name, self._interval_velocity)

    def step(self, action, n_render_iterations=1):
        before, t0 = self._encoders(), float(og.sim.current_time)
        result = self.env.step(action, n_render_iterations=n_render_iterations)
        after, t1 = self._encoders(), float(og.sim.current_time)
        dt = t1-t0
        if abs(dt-1.0/30.0) > 1e-5:
            raise ValueError(f'interval base adapter requires the declared 30Hz control interval, got {dt}')
        self._interval_velocity = interval_base_velocity(before, after, dt)
        self._interval_end = t1
        obs, reward, terminated, truncated, info = result
        return self._patch(obs), reward, terminated, truncated, info

    def reset(self):
        self._interval_velocity = self._interval_end = None
        return self.env.reset()

    def get_obs(self, *args, **kwargs):
        obs, info = self.env.get_obs(*args, **kwargs)
        return self._patch(obs), info
