"""Keep unsafe policy rollouts out of training and avoid large live JSON copies."""
from pathlib import Path
import json
from harness.zhuhai_dagger_resource_v10 import outer_failure_fields as resource_fields
from harness.zhuhai_dagger_parallel import GIB, can_expand
from harness.zhuhai_dagger_resource_v10 import ResourceAdmission as PreviousAdmission
from harness.zhuhai_dagger_support_admission_v8 import digest, require


def failure_fields(exc, step, authority, safety, infrastructure_type):
    fields, code = resource_fields(exc, step)
    if (isinstance(exc, infrastructure_type) and authority == 0 and step > 0
            and isinstance(safety, dict) and safety.get('passed') is False
            and isinstance(safety.get('events'), list) and safety['events']):
        fields.update(status='unsafe_policy_episode', failure_authority='policy',
                      failure_control_step=step, all_observations_safe=False,
                      quota_credit=0, termination='actual_policy_physics_violation')
        code = 0
    return fields, code


def write_result(out, row, atomic_json, terminal=False):
    out = Path(out)
    if 'scene_manifest' in row and 'scene_manifest_path' not in row:
        path = out/'scene_manifest.json'
        require(not path.exists(), 'Scene evidence must be written once')
        atomic_json(path, row['scene_manifest'])
        row.update(scene_manifest_path=str(path.resolve()), scene_manifest_sha256=digest(path))
    live = dict(row)
    if not terminal and 'scene_manifest' in live:
        del live['scene_manifest']
        live['live_scene_manifest_in_sidecar'] = True
    atomic_json(out/'result.json', live)


class ResourceAdmission(PreviousAdmission):
    def allowed(self, row, active, maximum=4):
        now = self.clock()
        if now < self.cooldown_until or not can_expand(row, active, maximum=maximum):
            return False
        if not active:
            return row['swap_fraction'] < .92
        if now < self.serial_until or row['swap_fraction'] >= (.85 if active >= 3 else .88):
            return False
        floor = max(2*GIB, .10*row['total'])
        # Two-worker v10 evidence: aggregate peak delta ~11.5 GiB, 81.84%
        # swap; original 12 GiB cold reserve denied worker three altogether.
        # Keep 8.5 GiB plus floor: ~6 GiB Isaac, ~2.5 GiB single WAM CPU peak.
        return row['available'] >= 8.5*GIB + floor


def audited_prior_policy_failure(plan, raw, path, adopted):
    if not adopted or raw.get('status') != 'infrastructure_error':
        return False
    audit = json.loads(Path(plan['prior_policy_failure_audit']).read_text())
    if Path(path).resolve() != Path(audit['result_path']).resolve():
        return False
    require(audit['passed'] is True and raw['seed'] == audit['seed']
            and raw['error'] == audit['error'] and raw['control_steps'] == 262
            and raw['expert_control_steps'] == 0 and raw['accepted_for_training'] is False,
            'Original policy failure identity differs')
    for item in audit['evidence']:
        require(digest(item['path']) == item['sha256'], 'Original policy failure evidence changed')
    rules = [json.loads(line) for line in (Path(path).parent/'rule_evidence.jsonl').read_text().splitlines()]
    require(len(rules) == 263 and rules[-1]['authority'] == 'policy'
            and rules[-1]['control_step'] == 262
            and rules[-1]['physics_safety'] == raw['physics_safety_at_failure']
            and rules[-1]['physics_safety']['passed'] is False,
            'Policy collision must retain its actual unsafe terminal evidence')
    require(all(row['physics_safety']['passed'] is True for row in rules[:-1]),
            'Earlier unknown unsafe frame cannot use the policy failure audit')
    return True
