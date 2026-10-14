"""Reuse the original strict supervisor as an actual native14 execution gate.

This kernel accepts real host providers. It supplies no detector, risk model or
recovery policy. Explicit test fakes prove software behavior, not robot ability.
"""
from dataclasses import dataclass
from copy import deepcopy
import hashlib
import json
import math
import time
from harness.robotwin_full_middle_bridge import FullMiddleBridge, SupervisorStop
from harness.robotwin_robot_feedback import hold_measured
from harness.robotwin_system_adapter import RobotwinSystemAdapter, reject_gt
from harness.robotwin_task_registry import require_full_providers
from harness.robotwin_task_supervisor import CapabilityCandidate, ProviderBinding, TaskSupervisor


@dataclass(frozen=True)
class RiskCalibrationBinding:
    provider: ProviderBinding
    calibration_id: str
    receipt_path: str
    receipt_sha256: str

    def validate(self, scope):
        from pathlib import Path
        provider = self.provider
        if (not isinstance(provider, ProviderBinding) or provider.scope_key != scope.key()
                or not callable(provider.callback) or not provider.provider_id or not provider.version
                or provider.admitted is not True or provider.trusted is not True
                or provider.independent is not True or not self.calibration_id):
            raise ValueError('registered, scoped, independently admitted risk provider required')
        path = Path(self.receipt_path)
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != self.receipt_sha256:
            raise ValueError('risk calibration receipt changed')
        receipt = json.loads(raw)
        expected = dict(format='native14-risk-calibration-v1', admitted=True,
                        provider_id=provider.provider_id, provider_version=provider.version,
                        scope_key=scope.key(), calibration_id=self.calibration_id,
                        robot_version=scope.robot_version, action_dim=14,
                        dispatch_horizon=1, evaluator_gt_used_at_runtime=False)
        if any(receipt.get(k) != v for k, v in expected.items()) or not receipt.get('validation_evidence'):
            raise ValueError('risk calibration receipt lacks matching validation evidence')
        # The host's admission is a trust boundary. This does not establish the
        # accuracy of a risk model merely by validating a JSON receipt.
        return receipt


@dataclass(frozen=True)
class ActionRiskEvidence:
    observation_id: str
    action_sha256: str
    observed_at: float
    calibration_id: str
    executable: bool | None
    risk_upper: float | None
    checked_steps: int
    source: str


def command_hash(action):
    if len(action) != 14 or any(type(v) not in (int, float) or not math.isfinite(v) for v in action):
        raise ValueError('finite native14 action required')
    return hashlib.sha256(json.dumps(list(action), separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def risk_decision(evidence, *, action, observation_id, observed_at, calibration_id, max_risk):
    """Evidence covers exactly the finally bounded single dispatch, never 8 unseen actions."""
    reject_gt(evidence)
    if not isinstance(evidence, ActionRiskEvidence):
        return 'HOLD', 'action_risk_unknown'
    if (evidence.observation_id != observation_id or evidence.action_sha256 != command_hash(action)
            or evidence.observed_at != observed_at or evidence.calibration_id != calibration_id
            or type(evidence.checked_steps) is not int or evidence.checked_steps != 1
            or not evidence.source or evidence.executable is not None and type(evidence.executable) is not bool
            or evidence.risk_upper is not None and (type(evidence.risk_upper) not in (int, float)
                or not math.isfinite(evidence.risk_upper) or not 0 <= evidence.risk_upper <= 1)):
        return 'HOLD', 'action_risk_binding_invalid'
    if evidence.executable is False:
        return 'HOLD', 'action_not_executable'
    if evidence.executable is None or evidence.risk_upper is None:
        return 'HOLD', 'action_risk_unknown'
    if evidence.risk_upper > max_risk:
        return 'HOLD', 'action_risk_limit_exceeded'
    return 'EXECUTE', 'single_bounded_dispatch_admitted'


@dataclass(frozen=True)
class CycleResult:
    mode: str
    reason: str
    reply: dict | None = None
    verified_outcome: str | None = None
    memory_record_id: int | None = None
    transaction_revoked_terminal: bool = False
    dispatch_ack: bool = False

    def __bool__(self):
        return self.mode == 'EXECUTE' and self.dispatch_ack is True


class StrictMiddleKernel:
    def __init__(self, *, definition, scope, contract, client, driver, robot,
                 visible_tracks, grounder, verifier, bound_command, risk_binding,
                 max_risk, memory=None, event=lambda e: None, clock=time.monotonic):
        if (definition.task != scope.task_id or definition.task != scope.skill_id
                or definition.robot_version != scope.robot_version):
            raise ValueError('whole-task scope differs from task registration')
        if (contract.action_dim != 14 or contract.state_dim != 14
                or not callable(bound_command) or not isinstance(risk_binding, RiskCalibrationBinding)
                or type(max_risk) not in (int, float) or not math.isfinite(max_risk) or not 0 <= max_risk <= 1):
            raise ValueError('native14 bounds and separately calibrated risk provider required')
        require_full_providers(scope, grounder, verifier, visible_tracks)
        self.risk_receipt = risk_binding.validate(scope)
        self.definition, self.event, self.clock = definition, event, clock
        self.bound_command, self.risk_binding = bound_command, risk_binding
        self.calibration_id, self.max_risk = risk_binding.calibration_id, max_risk
        self.robot, self.prepared, self._dispatch_bound = robot, None, None
        def checked_driver(raw_action):
            bounded = tuple(self.bound_command(raw_action))
            if self._dispatch_bound is None or bounded != self._dispatch_bound:
                raise ValueError('final driver command differs from risk-checked command')
            return driver(list(bounded))
        self.adapter = RobotwinSystemAdapter(client, checked_driver, max_age_s=definition.max_age_s, clock=clock)
        def replan(request):
            self.adapter.replan('strict_bound_recovery:' + request.operation)
            self.event({'kind': 'recovery_queue_cleared', 'observation_id': request.observation_id,
                        'history_reset': False, 'recovery_success_verified': None})
            return True
        self.supervisor = TaskSupervisor(contract=contract, scope=scope,
            required_predicates=definition.predicates, max_age_s=definition.max_age_s,
            ask_timeout_s=definition.max_age_s, grounder=grounder, verifier=verifier,
            memory=memory, max_questions=0, max_recoveries=definition.max_recoveries,
            request_reobserve=replan, request_replan=replan, clock=clock)
        self.bridge = FullMiddleBridge(self.adapter, self.supervisor, robot=robot,
                                      visible_tracks=visible_tracks, event=event, clock=clock)

    def reset(self, episode, seed, execution_candidate):
        self.prepared, self._dispatch_bound = None, None
        if (not isinstance(execution_candidate, CapabilityCandidate)
                or execution_candidate.capability != self.definition.task
                or execution_candidate.object_id.startswith('task-target-set:')):
            raise ValueError('real sensor-bound task anchor required, never a semantic target-set alias')
        # A real visible container track can anchor the whole-task contract.
        # All three bottle identities still require separate sensor grounding
        # and independent complete per-target verification by the providers.
        return self.bridge.reset(episode, seed, execution_candidate)

    def prepare(self, episode, step, observation_id, observed_at, observation):
        try:
            reply = self.bridge.prepare(episode, step, observation_id, observed_at, observation)
        except SupervisorStop as exc:
            directive = exc.directive
            return CycleResult(directive.mode, directive.reason, verified_outcome=directive.verified_outcome,
                               memory_record_id=directive.memory_record_id)
        try:
            bounded = list(self.bound_command(reply['raw_action']))
            command_hash(bounded)
            sensor_frame = deepcopy(self.supervisor._frame)
            self.risk_binding.validate(self.supervisor.scope)
            evidence = self.risk_binding.provider.callback(tuple(bounded), sensor_frame)
            if not isinstance(evidence, ActionRiskEvidence) or evidence.source != self.risk_binding.provider.provider_id:
                raise ValueError('risk evidence source differs from admitted provider')
            mode, reason = risk_decision(evidence, action=bounded, observation_id=observation_id,
                                        observed_at=observed_at, calibration_id=self.calibration_id, max_risk=self.max_risk)
        except Exception:
            mode, reason = 'HOLD', 'command_or_risk_provider_error'
            bounded = None
        self.event({'kind': 'single_action_risk_decision', 'observation_id': observation_id,
                    'action_sha256': command_hash(bounded) if bounded is not None else None, 'mode': mode, 'reason': reason,
                    'physical_success_verified': None})
        if mode != 'EXECUTE':
            # prepare consumed one RPC but driver has not run. Burn the pending
            # transaction and end this episode; do not resume with shifted step IDs.
            self.adapter.pending = None
            self.bridge._pending = None
            try:
                hold_measured(self.robot())
                self.adapter.replan('risk_gate_terminal_revoke')
            finally:
                self.adapter.failed = True
            return CycleResult('HOLD', reason, transaction_revoked_terminal=True)
        self.prepared = (episode, step, observation_id, tuple(bounded))
        return CycleResult('EXECUTE', reason, reply=reply)

    def execute(self, episode, step, observation_id):
        if self.prepared is None or self.prepared[:3] != (episode, step, observation_id):
            raise ValueError('no acknowledged prepared command')
        self._dispatch_bound = self.prepared[3]
        self.prepared = None
        try:
            acknowledged = self.bridge.execute(episode, step, observation_id)
            return CycleResult('EXECUTE', 'driver_acknowledged' if acknowledged else 'driver_not_acknowledged',
                               dispatch_ack=acknowledged is True)
        except SupervisorStop as exc:
            return CycleResult(exc.directive.mode, exc.directive.reason,
                               verified_outcome=exc.directive.verified_outcome,
                               transaction_revoked_terminal=self.adapter.failed)
        finally:
            self._dispatch_bound = None

    def request_recovery(self, operation):
        return self.bridge.request_recovery(operation)
