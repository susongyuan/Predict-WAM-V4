"""Explicit opt-in first-bin privbase side channel; normal physics/reset unchanged.

Inherits the separately disclosed interval-encoder proprio adapter. Additional
world poses live ONLY in the named numeric navigation packet, never proprio.
"""
import numpy as np
import omnigibson as og

from harness.interval_base_rgbd_wrapper import IntervalBaseRGBDWrapper
from harness.privileged_base_navigation import GOAL, KEY, MODEL, _vector


class PrivilegedBaseRGBDWrapper(IntervalBaseRGBDWrapper):
    def __init__(self, env):
        super().__init__(env)
        self._privbase_epoch = 0
        self._privbase_step = 0

    @staticmethod
    def _pose(entity):
        p, q = entity.get_position_orientation()
        return np.r_[_vector(p, 3), _vector(q, 4)]

    def _privbase_patch(self, obs):
        entity = self.env.task.object_scope.get(GOAL)
        if entity is None or entity.category != "trash_can" or entity.model != MODEL:
            raise ValueError(f"privbase requires named target {GOAL}, trash_can/{MODEL}; no nearest-object fallback")
        packet = np.r_[1., 90., self._privbase_epoch, self._privbase_step, float(og.sim.current_time),
                       self._pose(self._velocity_robot), self._pose(entity), _vector(entity.scale, 3)]
        return {**obs, KEY: packet}

    def step(self, action, n_render_iterations=1):
        obs, reward, terminated, truncated, info = super().step(action, n_render_iterations=n_render_iterations)
        self._privbase_step += 1
        return self._privbase_patch(obs), reward, terminated, truncated, info

    def reset(self):
        obs, info = super().reset()
        self._privbase_epoch += 1
        self._privbase_step = 0
        return self._privbase_patch(obs), info

    def get_obs(self, *args, **kwargs):
        obs, info = super().get_obs(*args, **kwargs)
        return self._privbase_patch(obs), info
