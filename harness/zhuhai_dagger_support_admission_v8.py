"""Verify separately retained unsafe-path rejection and reachable-goal evidence."""
import hashlib
import json
from pathlib import Path
import h5py
from harness.isaac_zhuhai_support_collision_v8 import VERSION


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def probe_evidence(path):
    path = Path(path)
    result = json.loads(path.read_text())
    plan = json.loads((path.parent/'plan.json').read_text())
    require(result.get('production') is False and result.get('quota_credit') == 0
            and result.get('independent_of_production_quota') is True
            and result.get('scene_reset_after_initial_setup') is False
            and result.get('policy_inference') is False
            and result.get('record_video') is False and result.get('third_view') is False,
            'Only independently archived physical verification can admit the repair')
    require(result.get('prefix_replay_measured_state_max_difference') == 0.,
            'Original safe prefix must recreate the exact measured intervention state')
    for source, expected in plan['sources'].items():
        require(digest(source) == expected, 'Probe source/input drift: '+source)
    screen = result['expert_result']['expert_path_screening']
    require(screen['version'] == VERSION and screen['static_collider_count'] == 16
            and screen['guarded_commands'] > 0 and screen['support_clearance_m'] >= .012,
            'Actual support-aware teacher commands and unchanged clearance required')
    require(result['final_physics_safety']['passed'] is True
            and result['final_rule_evidence']['physics_safety']['passed'] is True,
            'Actual probe execution must remain physically safe')
    require(digest(result['data_path']) == result['data_sha256'], 'Probe archive changed')
    with h5py.File(result['data_path'], 'r') as archive:
        require(bool(archive.attrs['physics_safety_all_observations_passed'])
                and int(archive.attrs['source_observation_count']) == result['control_steps']+1,
                'Every real observation, including terminal, must pass safety')
        require(all(json.loads(record)['physics_safety']['passed'] is True
                    for record in archive['rule_evidence']), 'Unsafe observed frame in probe')
    return result, plan


def verify_pair(rejection_path, reachable_path):
    rejection, first = probe_evidence(rejection_path)
    reachable, second = probe_evidence(reachable_path)
    require(first['scenario'] == second['scenario'] and first['sources'] == second['sources'],
            'Repair checks must use the same scene and frozen teacher/physics sources')
    require(first['scenario']['seed'] == 1014003 and first['replay_until_control_step'] == 1410
            and first['goal'] == 'grasp' and first['key'] == 'spoon',
            'Archived support fault intervention must be tested')
    require(rejection['passed'] is False and rejection['status'] == 'safe_rejection_or_local_goal_failure'
            and rejection['expert_result']['passed'] is False
            and rejection['expert_result']['expert_path_screening']['support_rejections'] > 0,
            'Unsafe recovery must remain an explicit rejected local goal')
    require(second['replay_until_control_step'] == 300 and second['goal'] == 'grasp'
            and second['key'] == 'bowl' and reachable['passed'] is True
            and reachable['status'] == 'passed' and reachable['expert_result']['passed'] is True,
            'An independently archived reachable local goal must actually pass')
    require(rejection['expert_result']['expert_path_screening']['support_geometry_sha256']
            == reachable['expert_result']['expert_path_screening']['support_geometry_sha256'],
            'Fixed support geometry must be identical across checks')
    return dict(passed=True, production=False, quota_credit=0,
                scope='known unsafe spoon recovery safely rejected; reachable bowl grasp physically verified; full-task and model-handback admission remain separate',
                unsafe_case_recovered=False, support_screening_protocol=VERSION,
                probes=[dict(path=str(Path(path).resolve()), sha256=digest(path))
                        for path in (rejection_path, reachable_path)])


def verify_admission(path):
    receipt = json.loads(Path(path).read_text())
    require(receipt.get('passed') is True and len(receipt['probes']) == 2,
            'Both support repair checks are required')
    for probe in receipt['probes']:
        require(digest(probe['path']) == probe['sha256'], 'Bound physical check changed')
    actual = verify_pair(*(probe['path'] for probe in receipt['probes']))
    require(receipt == actual, 'Support repair scope or evidence differs')
    return receipt
