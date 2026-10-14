#!/usr/bin/env python3
"""Interpret saved static audit only. No USD open, decryption, simulation or sensor replay."""
import hashlib
import json
import math
from pathlib import Path

OUT = Path(__file__).resolve().parent
SOURCE = OUT/'scope.json'
DEST = OUT/'assessment.json'
assert not DEST.exists(), 'Do not overwrite completed assessment'
def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()
d = json.loads(SOURCE.read_text())
s = d['static_usd']
assert s and d['input_assets_unchanged'] and d['resources']['decrypted_temporary_files_removed']
assert not list(OUT.glob('.static_decrypt_*'))
links = {l['name']: l for l in s['links']}
base, movable = links['base_link'], links['link_0']
j = next(j for j in s['joints'] if j['body1'] == ['/ifzxzj/link_0'])
assert j['type'] == 'PhysicsRevoluteJoint' and j['body0'] == ['/ifzxzj/base_link']
assert j['axis_token'] == 'Y' and j['enabled'] and j['lower_limit'] < j['upper_limit']
assert j['authored_attributes']['physics:localRot0'] == [1., 0., 0., 0.]
assert all(l['metadata_collision_aabb_max_abs_difference_m'] < 1e-6 for l in (base, movable))
shapes = [p for l in s['links'] for p in l['shapes']]
assert len(shapes) == 12 and all(p['enabled'] and p['attributes']['physics:approximation'] == 'convexHull' for p in shapes)
assert sum(p['native_root']['vertex_count'] for p in shapes) == 529
assert all(p['link'] == 'base_link' for p in s['lowest_native_authored_configuration'])
prior_diameter = d['rigid_height_diameter_m']
assert abs(prior_diameter - math.sqrt(sum(v*v for v in d['prior']['native_bbox_m']))) < 1e-12
result = {
    'status': 'static_articulation_confirmed_runtime_rigidity_unknown_not_admitted',
    'source_files': {str(p): sha(p) for p in (SOURCE, Path(__file__), OUT/'audit_static_asset.py')},
    'facts': {
        'material_geometry_links': ['base_link', 'link_0'],
        'auxiliary_rigid_prim': 'meta__base_link_fillable_0_0_link; fixed to base_link; no CollisionAPI mesh',
        'link_hierarchy': 'root has sibling link prims; physical joint graph is base_link -> link_0 (revolute), base_link -> meta__base_link_fillable_0_0_link (fixed)',
        'joint_axis': 'Y in joint frame, aligned with base_link local Y because authored localRot0 is identity; not runtime world Y',
        'joint_limits_degrees': [j['lower_limit'], j['upper_limit']],
        'joint_angular_stiffness': j['authored_attributes']['drive:angular:physics:stiffness'],
        'joint_angular_damping': j['authored_attributes']['drive:angular:physics:damping'],
        'collision_geometry': '12 enabled convexHull meshes (base_link 8 / link_0 4), 529 vertices, static declarations only; not cooked PhysX contact geometry',
        'base_collision_extent_native_m': base['link_local_collision_union']['extent_m'],
        'link_0_collision_extent_native_m': movable['link_local_collision_union']['extent_m'],
        'whole_metadata_bbox_native_m': d['prior']['native_bbox_m'],
        'whole_collision_extent_authored_configuration_m': s['whole_declared_collision_native_root']['extent_m'],
        'rigid_height_prior_diameter_m': prior_diameter,
        'lowest_geometry_native_authored_configuration': s['lowest_native_authored_configuration'],
        'link_0_min_native_root_z_m': movable['native_root_collision_union']['min_m'][2],
        'metadata_collision_aabb_max_difference_m': max(l['metadata_collision_aabb_max_abs_difference_m'] for l in (base, movable))
    },
    'verdict': {
        'static_asset_is_one_rigid_body': False,
        'single_whole_object_transform_is_structurally_guaranteed': False,
        'runtime_relative_link_transform_constant': None,
        'runtime_instance_scale': None,
        'current_point_binding_error_found': False,
        'current_point_binding_correctness_verified': False,
        'actual_link_0_or_lid_motion_found': False,
        'link_0_semantic_name': 'not specified by metadata tags; retain link_0 rather than infer a named lid',
        'reason': 'The authored revolute joint permits relative motion; this disproves a structural single-rigid-body guarantee, not constant relative pose in a particular recording. Runtime lock, angle and motion were not read.',
        'diameter_scope': '0.944907m is the native metadata bbox diagonal used by rigid height, not the 1.133889m proposal gate. Unknown scale/articulation and differing collision bounds prevent certification. A larger collision AABB alone does NOT prove actual pairwise diameter exceeds this prior.',
        'support_scope': 'base_link is lowest only in the native authored configuration under +Z. These coordinates are not scene ground heights or current support/contact labels; tilted/articulated configurations were not evaluated.',
        'prior_parse_failures_resolved': 'The parser initially required every RigidBodyAPI prim to own collider meshes; the fixed fillable helper has none. Both real links have enabled colliders. Two incomplete CPU parse attempts are preserved in scope.json; no disabled physical collider was discovered.',
        'body_only_status': 'Potentially valid only after sensor-supported same-base_link membership, independently validated size/error bound and initial/current support relation. Do not automatically shrink the diameter or rebind points.',
        'whole_object_clearance_status': 'Unknown until all relevant links/scale/joint uncertainty and all support surfaces are conservatively covered; body pose alone is insufficient without justified relative-link constraints.',
        'observed_bank_comparison': 'TRAIN18000 reference z 0.199..0.672m (40 points), TRAIN18001 0.017..0.671m (22 points). These are bank coordinates, not asset-local/floor-relative measurements; spans and earlier RGB crops do not identify links or prove support coverage.'
    },
    'checkable_conditions': d['checkable_conditions'],
    'guardrails': {'runtime_changed': False, 'whole_enclosure_producer_created': False, 'admission': False,
                   'physical_label': None, 'simulation_gpu_downloads': False, 'current_dev_state_read': False,
                   'new_agents': 0, 'decrypted_temporary_files_removed': True},
    'reproduction': {
        'static_parse_command': '/home/user/Robotic/project2/.conda-envs/behavior6/bin/python -B '+str(OUT/'audit_static_asset.py'),
        'interpretation_command': 'python3 -B '+str(Path(__file__)),
        'output_guard': 'Scripts refuse replacing completed evidence. To rerun in this sole output directory, explicitly archive the two completed JSONs here first; do not change original assets.'}
}
with DEST.open('x') as f:
    json.dump(result, f, indent=2, ensure_ascii=False, allow_nan=False)
    f.write('\n')
print(json.dumps({'assessment': str(DEST), 'sha256': sha(DEST), 'checks': 'passed',
                  'parse_attempt_count': 1+len(d.get('prior_parse_attempts', [])),
                  'aggregate_parse_elapsed_s': d['resources']['elapsed_s']+sum(p['resources']['elapsed_s'] for p in d.get('prior_parse_attempts', [])),
                  'max_parse_rss_mib': max([d['resources']['max_rss_mib']]+[p['resources']['max_rss_mib'] for p in d.get('prior_parse_attempts', [])])}))
