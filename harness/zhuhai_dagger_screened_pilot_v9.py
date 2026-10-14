"""Bind successful teacher sidecars to the actual archived local spans."""
from pathlib import Path
import json
import h5py
from harness.zhuhai_dagger_segmented import SegmentedEpisode
from harness.zhuhai_dagger_segmented_admission import pilot_eligible
from harness.zhuhai_dagger_support_admission_v8 import digest, require
from harness.isaac_zhuhai_support_collision_v8 import VERSION


def screened_pilot(raw, result_path, row):
    if raw.get('expert_path_screening_protocol') != VERSION:
        return None
    episode = SegmentedEpisode(row)
    if not pilot_eligible(episode, raw):
        return None
    result_path = Path(result_path).resolve()
    old_plan_path = result_path.parents[2]/'plan.json'
    old_plan = json.loads(old_plan_path.read_text())
    for relative in ('harness/isaac_zhuhai_segmented_expert_v8.py',
                     'harness/isaac_zhuhai_support_collision_v8.py',
                     'scripts/collect_precision_dagger_segmented_v8.py',
                     'scripts/precision_segmented_policy_worker.py'):
        source = Path(__file__).resolve().parents[1]/relative
        require(old_plan['extension_sources'][str(source)] == digest(source),
                'Pilot teacher/collector/policy implementation differs')
    with h5py.File(row['path'], 'r') as archive:
        metadata = json.loads(archive.attrs['metadata'])
        require(metadata['plan_sha256'] == digest(old_plan_path)
                and metadata['expert_path_screening_protocol'] == VERSION,
                'Pilot archive does not bind the original screened collection plan')
    geometry = None
    records = []
    for index, span in enumerate(raw['segments']):
        path = result_path.parent/f'expert_segment_{index:02d}.json'
        info = json.loads(path.read_text())
        require(all(info[key] == span[key] for key in ('goal', 'key', 'passed', 'failure')),
                'Teacher outcome differs from the actual local span')
        real_commands = span['end']-span['start']
        # A terminal budget callback can raise after the last physical command,
        # before SmoothExpert appends that command to its private trace list.
        require(info['command_count'] == real_commands if span['passed'] else
                info['command_count'] in (real_commands, real_commands-1),
                'Teacher sidecar command count differs from real archive')
        screen = info['expert_path_screening']
        require(screen['version'] == VERSION and screen['static_collider_count'] == 16,
                'Teacher support screening sidecar missing or incompatible')
        if geometry is None:
            geometry = screen['support_geometry_sha256']
        require(screen['support_geometry_sha256'] == geometry, 'Support geometry changed between spans')
        if span['passed']:
            require(screen['guarded_commands'] == info['command_count']
                    and screen['rejected_commands'] == 0
                    and info['goal_evidence']['physics_safety']['passed'] is True,
                    'Every supervised command must be actually screened and physically safe')
        records.append(dict(path=str(path), sha256=digest(path), passed=span['passed']))
    return dict(seed=row['seed'], actual_verified_segments=row['verified_segment_count'],
                actual_model_handbacks=row['handback_count'], archive_validated=True,
                raw_result_path=str(result_path), raw_result_sha256=digest(result_path),
                data_sha256=row['sha256'], support_screening_protocol=VERSION,
                new_repaired_expert_pilot=True, original_plan_path=str(old_plan_path),
                original_plan_sha256=digest(old_plan_path), local_vs_full_success_separate=True,
                expert_segment_receipts=records, later_expert_failure=raw.get('expert_failure'),
                failed_spans_excluded=True)
