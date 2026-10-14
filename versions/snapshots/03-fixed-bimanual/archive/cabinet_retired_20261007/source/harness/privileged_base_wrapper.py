"""Explicit privileged-base-only Task90 diagnostic; ordinary wrappers are unchanged.

Adds one named numeric navigation leaf. No task predicates, joints, gripper poses,
rewards, object states or trajectory answers are read. Physics is still env.step.
Inherits the already-disclosed interval encoder velocity observation adapter.
"""
import numpy as np
import omnigibson as og

from harness.interval_base_rgbd_wrapper import IntervalBaseRGBDWrapper
from harness.privileged_base_navigation import GOAL, MODEL, KEY


def _array(value):
    return np.asarray(value.detach().cpu().numpy() if hasattr(value, "detach") else value, dtype=float)


class PrivilegedBaseRGBDWrapper(IntervalBaseRGBDWrapper):
    def __init__(self, env):
        super().__init__(env)
        self._privbase_epoch = 0
        self._privbase_step = 0

    def _patch(self, obs):
        obs = super()._patch(obs)
        entity = self.env.task.object_scope.get(GOAL)
        if entity is None or entity.category != "trash_can" or entity.model != MODEL:
            raise ValueError(f"privbase requires exact named {GOAL} with trash_can/{MODEL}; no nearest-object fallback")
        bp, bq = self._velocity_robot.get_position_orientation()
        tp, tq = entity.get_position_orientation()
        packet = np.r_[1, 90, self._privbase_epoch, self._privbase_step, float(og.sim.current_time),
                       _array(bp), _array(bq), _array(tp), _array(tq), _array(entity.scale)]
        if packet.shape != (22,) or not np.isfinite(packet).all():
            raise ValueError("privbase live pose unavailable")
        return {**obs, KEY: packet}

    def reset(self):
        self._privbase_epoch += 1
        self._privbase_step = 0
        obs, info = super().reset()
        return self._patch(obs), info

    def step(self, action, n_render_iterations=1):
        self._privbase_step += 1
        return super().step(action, n_render_iterations=n_render_iterations)
