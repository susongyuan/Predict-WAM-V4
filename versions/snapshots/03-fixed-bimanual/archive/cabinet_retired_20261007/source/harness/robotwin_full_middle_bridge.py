"""Connect the original strict task supervisor to native14 dispatch.

No VLM or ASK is used. Sensor-bound grounding and independent outcome providers
must be supplied by the host; none is manufactured here. This connector does not
admit a detector, implement a physical risk estimator, or certify a full system.
Keep this separate from the frozen middle10 experiment.
"""
from copy import deepcopy
from dataclasses import dataclass
import time

from harness.robotwin_robot_feedback import robot_feedback, hold_measured
from harness.robotwin_system_adapter import reject_gt
from harness.robotwin_task_supervisor import ModelContract
from harness.robotwin_vtt_runtime import adapter_metadata
from harness.runtime import Observation


@dataclass(frozen=True)
class TrainedVTTContract(ModelContract):
    """Whole-task VTT residual compatibility, never physical capability admission."""
    condition_mode: str = 'trained_vtt_residual'
    adapter_path: str = ''
    base_checkpoint_sha256: str = ''
    adapter_sha256: str = ''

    def validate(self, scope):
        if self.condition_mode != 'trained_vtt_residual':
            raise ValueError('explicit trained VTT residual mode required')
        # Reuse every native14/state/view/horizon/version check of the original
        # zero contract. Do not relax that contract globally for running jobs.
        original = ModelContract(self.capability, self.wam_version,
                                 self.robot_version, self.controller_version,
                                 task_extent=self.task_extent,
                                 condition_dim=self.condition_dim,
                                 state_dim=self.state_dim, action_dim=self.action_dim,
                                 rgb_views=self.rgb_views,
                                 state_semantics=self.state_semantics,
                                 execution_horizon=self.execution_horizon)
        original.validate(scope)
        _, meta = adapter_metadata(self.adapter_path, self.base_checkpoint_sha256)
        if meta['sha256'] != self.adapter_sha256:
            raise ValueError('VTT artifact differs from pinned contract')
        expected_version = 'cabinet-vtt:' + self.base_checkpoint_sha256 + ':' + self.adapter_sha256
        if self.wam_version != expected_version:
            raise ValueError('scope must identify both base and trained adapter')


class SupervisorStop(RuntimeError):
    """Controlled task stop; a runner must keep this separate from IPC errors."""
    def __init__(self, directive):
        self.directive = directive
        super().__init__(directive.mode + ':' + directive.reason)


class FullMiddleBridge:
    """The supervisor actually gates dispatch, rather than producing shadow logs.

    `adapter` supplies the existing measured controller/transaction/VTT path.
    `visible_tracks` receives only allowed sensor channels, never an environment.
    Its output does not confer provider admission. The strict supervisor still
    checks independently registered grounding and outcome bindings. Unknown
    completion never writes successful experience. Risk admission is external.
    Recovery remains the strict supervisor's observation-bound callback path.
    """
    def __init__(self, adapter, supervisor, *, robot, visible_tracks,
                 event=lambda value: None, clock=time.monotonic):
        if supervisor.max_questions != 0:
            raise ValueError('full middle bridge requires ASK disabled')
        if not callable(visible_tracks):
            raise TypeError('sensor-only visible track ingress required')
        self.adapter, self.supervisor = adapter, supervisor
        self.robot, self.visible_tracks = robot, visible_tracks
        self.event, self.clock = event, clock
        self._pending = None

    def reset(self, episode, seed, execution_candidate):
        if self._pending is not None:
            raise ValueError('cannot reset a pending transaction')
        self.supervisor.begin_episode(episode, execution_candidate)
        return self.adapter.reset(episode, seed)

    def _stop(self, directive, *, revoked=False):
        # Invalidate queued policy commands and hold measured joints. This is
        # not a guarantee of contact stability or collision safety.
        hold = hold_measured(self.robot())
        clear = self.adapter.replan('strict_task_supervisor:' + directive.reason)
        if revoked:
            self.adapter.failed = True  # RPC consumed without dispatch: terminate, never resume.
        self.event({'kind': 'strict_task_stop', 'mode': directive.mode,
                    'reason': directive.reason, 'hold': hold, 'clear': clear,
                    'verified_outcome': directive.verified_outcome,
                    'transaction_revoked_terminal': revoked})
        raise SupervisorStop(directive)

    def prepare(self, episode, step, observation_id, observed_at, observation):
        if (self._pending is not None or self.adapter.pending is not None
                or self.adapter.failed or episode != self.adapter.episode
                or step != self.adapter.next_step):
            raise ValueError('wrong or pending transaction before task supervision')
        reject_gt(observation)
        feedback = robot_feedback(self.robot())
        sensors = {'rgb': observation['rgb'], 'proprioception': feedback}
        reject_gt(sensors)
        tracks = self.visible_tracks(deepcopy(sensors))
        if (not isinstance(tracks, tuple)
                or any(not isinstance(t, str) or not t for t in tracks)
                or len(set(tracks)) != len(tracks)):
            raise ValueError('unique sensor-bound track IDs required')
        frame = Observation(observation_id, observed_at, sensors)
        directive = self.supervisor.observe(frame, visible_objects=tracks)
        if directive.mode != 'DONE':
            directive = self.supervisor.ground()
        if directive.mode == 'EXECUTE':
            directive = self.supervisor.evaluate_visible()
        self.event({'kind': 'strict_task_decision', 'episode': episode,
                    'observation_id': observation_id, 'mode': directive.mode,
                    'reason': directive.reason,
                    'verified_outcome': directive.verified_outcome,
                    'memory_record_id': directive.memory_record_id})
        if directive.mode != 'EXECUTE':
            self._stop(directive)
        reply = self.adapter.prepare(episode, step, observation_id,
                                     observed_at, observation)
        self._pending = (episode, step, observation_id)
        return reply

    def execute(self, episode, step, observation_id):
        if self._pending != (episode, step, observation_id):
            raise ValueError('wrong or already consumed bridge transaction')
        self._pending = None
        directive = self.supervisor.decision()
        if directive.mode != 'EXECUTE':
            # The adapter has consumed a policy RPC but not dispatched. Its
            # pending command must be burned, not resubmitted after expiration.
            self.adapter.pending = None
            self._stop(directive, revoked=True)
        try:
            return self.adapter.execute(episode, step, observation_id)
        except BaseException:
            hold_measured(self.robot())
            raise

    def request_recovery(self, operation):
        if self._pending is not None or self.adapter.pending is not None:
            raise ValueError('cannot recover an unacknowledged transaction')
        if operation not in ('reobserve', 'replan'):
            raise ValueError('only strict observation-bound recovery is supported')
        return getattr(self.supervisor, 'request_' + operation)()
