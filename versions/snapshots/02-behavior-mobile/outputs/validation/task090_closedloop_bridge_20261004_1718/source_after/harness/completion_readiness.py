"""Reject known structurally unadmitted completion gates before expensive execution.

This is not a physical precondition or a comprehensive capability/safety check.
Only implementations exposing a checked admission status are inspected; unchecked
legacy/pure-policy paths must not be described as admitted by this report.
"""
from collections.abc import Mapping

PURPOSES = ('capability_validation', 'diagnostic_only')


def completion_readiness(task_id, config, *, purpose=None):
    if config is not None and not isinstance(config, Mapping):
        raise ValueError('orchestrate must be a mapping')
    is_orchestrated = config is not None
    config = config or {}
    declared = config.get('run_purpose')
    purpose = purpose if purpose is not None else declared or 'capability_validation'
    if purpose not in PURPOSES or declared is not None and declared != purpose:
        raise ValueError('invalid or conflicting run_purpose')
    report = dict(run_purpose=purpose, scope='known_unadmitted_completion_gates_only',
        status='not_checked', gate_admitted=None, launch_allowed=True,
        cannot_complete_by_design=None, blockers=[], proves_current_preconditions=False,
        proves_task_capability=False)
    mode = config.get('postconditions')
    if mode != 'task090_observed' and (task_id != 90 or not is_orchestrated):
        return report
    from harness.task090_predicates import Task090Postconditions, validate_mode
    validate_mode(task_id, config)
    gate = Task090Postconditions(config.get('task090_observation_calibration'))
    status = gate.admission_status()
    blockers = list(status.get('admission_blockers', ()))
    if status.get('calibration_blocker'):
        blockers.append(status['calibration_blocker'])
    admitted = status.get('admitted') is True and not blockers
    if not admitted and not blockers:
        blockers.append('completion_gate_not_admitted')
    report.update(status='gate_admitted' if admitted else 'gate_unadmitted',
        gate_admitted=admitted, launch_allowed=admitted or purpose == 'diagnostic_only',
        cannot_complete_by_design=not admitted, blockers=blockers,
        gate_source='harness.task090_predicates.Task090Postconditions', admission_status=status)
    return report


def require_completion_readiness(task_id, config, *, purpose=None):
    report = completion_readiness(task_id, config, purpose=purpose)
    if not report['launch_allowed']:
        raise ValueError('unadmitted completion gate: ' + ', '.join(report['blockers'])
            + '; use explicit diagnostic_only only for observation/interface diagnostics')
    return report
