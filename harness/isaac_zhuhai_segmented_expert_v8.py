"""Local teacher with support-aware planning and a pre-dispatch support guard."""
from copy import deepcopy
import numpy as np
from harness.isaac_zhuhai_segmented_expert import SegmentedExpert as PreviousExpert
from harness import isaac_zhuhai_expert as original
from harness.isaac_zhuhai_support_collision_v8 import PandaSupportCollision


class ExpertPathBlocked(RuntimeError):
    pass


class SegmentedExpert(PreviousExpert):
    def __init__(self, env, limiter):
        super().__init__(env, limiter)
        self.support_checker = PandaSupportCollision(self.fk_model, env.stage,
            env.support_contract['static_collision_paths'])
        self.fk_model.collision_checker = self.support_checker
        self.guarded_commands = 0
        self.rejected_commands = 0
        self.trace.append(dict(phase='recovery:actual_fixed_support_geometry_bound',
                               path_screen=self.support_checker.report()))

    def _drive(self, action):
        # A rejected proposal must not advance the real inherited controller.
        before = deepcopy(self.limiter.__dict__)
        accepted, stats = self.limiter.step(action)
        try:
            measured = np.asarray(self.env.actual_state(), dtype=float)
            if measured.shape != (16,) or not np.isfinite(measured).all():
                raise ValueError('Actual native16 state required for expert dispatch screening')
            for arm in ('left', 'right'):
                q = self._arm_slice(arm)
                if not self.support_checker.path_free(arm, measured[q], accepted[q]):
                    self.rejected_commands += 1
                    raise ExpertPathBlocked('Before dispatch: '+repr(self.support_checker.last_pair))
        except Exception:
            self.limiter.__dict__.clear()
            self.limiter.__dict__.update(before)
            raise
        self.env.expert_limit_stats = stats
        self.guarded_commands += 1
        return original.SmoothExpert._drive(self, accepted)

    def run_segment(self, goal, key):
        result = super().run_segment(goal, key)
        result['expert_path_screening'] = self.support_checker.report()
        result['expert_path_screening'].update(guarded_commands=self.guarded_commands,
                                              rejected_commands=self.rejected_commands)
        return result
