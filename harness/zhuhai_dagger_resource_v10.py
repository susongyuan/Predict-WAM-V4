"""Conservative admission and exact evidence for the step-zero resource stop."""
from pathlib import Path
import json
import time
from harness.zhuhai_dagger_parallel import GIB, ResourcePressure, can_expand
from harness.zhuhai_dagger_support_admission_v8 import digest, require


def outer_failure_fields(exc, step):
    resource = isinstance(exc, ResourcePressure)
    fields = dict(status='resource_paused' if resource else 'infrastructure_error',
                  error=f'{type(exc).__name__}: {exc}', accepted_for_training=False,
                  control_steps=step)
    if resource:
        fields.update(resource_failure=fields['error'], termination='resource_pause_before_rollout')
    return fields, 0 if resource else 2


class ResourceAdmission:
    """Hard guards remain unchanged; pressure reduces optional concurrency."""
    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.cooldown_until = 0.
        self.serial_until = 0.

    def pressure(self):
        now = self.clock()
        self.cooldown_until = now + 180.
        self.serial_until = now + 1800.

    def allowed(self, row, active, maximum=4):
        now = self.clock()
        if now < self.cooldown_until or not can_expand(row, active, maximum=maximum):
            return False
        # One environment can proceed with old swapped desktop pages; reserve
        # three percentage points before the unchanged 95% runtime stop.
        if not active:
            return row['swap_fraction'] < .92
        if now < self.serial_until or row['swap_fraction'] >= .85:
            return False
        floor = max(2*GIB, .10*row['total'])
        # Seven GiB environment + three GiB model CPU peak + two GiB cushion.
        return row['available'] >= 12*GIB + floor


def audited_resource_pause(plan, raw, path, adopted):
    if not adopted or raw.get('status') != 'infrastructure_error':
        return False
    audit = json.loads(Path(plan['prior_resource_audit']).read_text())
    if Path(path).resolve() != Path(audit['result_path']).resolve():
        return False
    require(audit['passed'] is True and raw['seed'] == audit['seed']
            and raw['error'] == audit['error'] and raw['control_steps'] == 0
            and raw['accepted_for_training'] is False, 'Step-zero resource audit identity differs')
    for item in audit['evidence']:
        require(digest(item['path']) == item['sha256'], 'Original resource evidence changed')
    gates = raw['admission_gates']
    require(gates['camera']['passed'] is True and gates['mechanical']['passed'] is True
            and not raw.get('physics_safety_at_failure'), 'Resource audit cannot exempt unsafe physics')
    directory = Path(path).parent
    require(not (directory/'actions.jsonl').exists(), 'Step-zero resource audit contains actions')
    rules = [json.loads(line) for line in (directory/'rule_evidence.jsonl').read_text().splitlines()]
    require(len(rules) == 1 and rules[0]['control_step'] == 0
            and rules[0]['physics_safety']['passed'] is True, 'Initial resource-stop physics evidence differs')
    return True
