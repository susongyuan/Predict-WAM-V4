#!/usr/bin/python3
"""Generate one editable SVG and render its exact PNG with system librsvg.

No image model or diagram dependency; explicit ports and orthogonal routes.
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/architecture'
BACKUP = ROOT / 'docs/archive/architecture_before_system_20261006'
VISUAL_REVIEW = {'svg_sha256': '5113ea68c0f5ff35f206afb08347353a4214e18f197ffa1649e1d7bb67fe239f', 'png_sha256': 'a2c1463c718b6a97d321330419d17b0d5963f128036a2cbbcee7dfad5ee9356d', 'method': 'Visual inspection of final 3072x1830 PNG via image viewer, plus SVG route/ink checks', 'checks': ['Nine original boxes and seventeen arrows retained; direction and two bidirectional paths inspected', 'No clipped text, overlapping labels, crossed routes or hidden arrowheads observed', 'Pure/transparent live equivalence separated from CPU41 and unadmitted GPU/visible-provider interfaces', 'Controller-to-visible-evidence arrow is dashed raw-RGB admission, not GT feedback', '10 paired / conditional20 authorization visibly NOT STARTED; no system gain or visual-pass claim']}  # Invalidated automatically when SVG/PNG changes.
NODES = {
    'world': (50, 200, 300, 650),
    'vlm': (420, 200, 280, 190),
    'orch': (780, 200, 330, 250),
    'nav': (1180, 200, 340, 250),
    'controller': (1600, 200, 380, 650),
    'human': (420, 650, 280, 200),
    'bank': (780, 650, 330, 200),
    'wam': (1180, 650, 340, 200),
    'predicates': (780, 930, 740, 170),
}
# Preserve the nine node boxes and all seventeen orthogonal routes. They are
# functional views, not nine separately deployed services. Solid means an
# explicit contract or pure/transparent native interface; dashed means no live admission.
EDGES = [
    ('world', 'vlm', 'rgb_for_optional_grounding', [(350, 285), (420, 285)], 'data'),
    ('vlm', 'orch', 'optional_goal_programs', [(700, 285), (780, 285)], 'data'),
    ('orch', 'nav', 'fixed_task_and_embodiment_config', [(1110, 300), (1180, 300)], 'data'),
    ('orch', 'controller', 'runtime_execute_hold_reset', [(1040, 200), (1040, 180), (1680, 180), (1680, 200)], 'action'),
    ('controller', 'world', 'live_14d_physical_command', [(1790, 200), (1790, 100), (200, 100), (200, 200)], 'action'),
    ('world', 'controller', 'offline_rgb_t_target_state_t_command_t1', [(250, 200), (250, 140), (1750, 140), (1750, 200)], 'data'),
    ('nav', 'controller', 'native_14d_schema', [(1520, 280), (1600, 280)], 'data'),
    ('controller', 'nav', 'live_timing_and_joint_feedback', [(1600, 380), (1520, 380)], 'feedback'),
    ('nav', 'orch', 'physical_readiness_status', [(1180, 400), (1110, 400)], 'feedback'),
    ('orch', 'wam', 'zero_single_task_condition', [(1110, 420), (1150, 420), (1150, 750), (1180, 750)], 'data'),
    ('controller', 'wam', 'three_view_features_and_14d_history', [(1600, 700), (1520, 700)], 'data'),
    ('wam', 'controller', 'sampled_32x14d_chunks', [(1520, 780), (1600, 780)], 'action'),
    ('predicates', 'orch', 'independent_sensor_completion_status', [(780, 1000), (750, 1000), (750, 420), (780, 420)], 'feedback'),
    ('bank', 'orch', 'scoped_experience_read_write', [(945, 650), (945, 450)], 'feedback'),
    ('human', 'orch', 'optional_ask_and_bound_reply', [(700, 750), (720, 750), (720, 400), (780, 400)], 'feedback'),
    ('orch', 'vlm', 'optional_recovery_replan', [(780, 365), (700, 365)], 'feedback'),
    ('controller', 'predicates', 'sensor_bound_outcome_evidence', [(1790, 850), (1790, 1000), (1520, 1000)], 'data'),
]
PENDING = {
    'rgb_for_optional_grounding', 'optional_goal_programs',
    'runtime_execute_hold_reset',
    'live_timing_and_joint_feedback', 'physical_readiness_status',
    'independent_sensor_completion_status', 'scoped_experience_read_write',
    'optional_ask_and_bound_reply', 'optional_recovery_replan', 'sensor_bound_outcome_evidence',
}
CONNECTED = {'live_14d_physical_command', 'three_view_features_and_14d_history',
             'sampled_32x14d_chunks', 'offline_rgb_t_target_state_t_command_t1'}

def edge_status(name):
    return ('CPU/probe interface only; live provider not admitted' if name in PENDING else
            'connected in pure/transparent native execution; no system uplift claim' if name in CONNECTED else
            'offline contract implemented')

BIDIRECTIONAL = {'scoped_experience_read_write', 'optional_ask_and_bound_reply'}


def on_boundary(point, box):
    px, py = point
    x, y, w, h = box
    return ((px in (x, x+w) and y <= py <= y+h)
            or (py in (y, y+h) and x <= px <= x+w))


def validate():
    """Fail on miswired endpoints, box penetration, or crossing arrow routes."""
    assert len(NODES) == 9 and len(EDGES) == 17
    assert len({e[2] for e in EDGES}) == len(EDGES), 'duplicate edge meaning'
    assert PENDING | BIDIRECTIONAL <= {e[2] for e in EDGES}
    for x, y, w, h in NODES.values():
        assert 0 <= x < x+w <= 2048 and 0 <= y < y+h <= 1220, 'node outside canvas'
    segments = []
    for source, target, name, points, _ in EDGES:
        assert all(0 <= x <= 2048 and 0 <= y <= 1220 for x, y in points), (name, 'canvas')
        assert on_boundary(points[0], NODES[source]), (name, 'source')
        assert on_boundary(points[-1], NODES[target]), (name, 'target')
        for a, b in zip(points, points[1:]):
            assert (a[0] == b[0]) != (a[1] == b[1]), (name, 'non-orthogonal')
            segments.append((name, a, b))
            for node, (x, y, w, h) in NODES.items():
                if a[0] == b[0]:
                    inside = x < a[0] < x+w and max(min(a[1], b[1]), y) < min(max(a[1], b[1]), y+h)
                else:
                    inside = y < a[1] < y+h and max(min(a[0], b[0]), x) < min(max(a[0], b[0]), x+w)
                assert not inside, (name, 'penetrates', node)
    for i, (name, a, b) in enumerate(segments):
        for other, c, d in segments[i+1:]:
            if name == other:
                continue
            av, cv = a[0] == b[0], c[0] == d[0]
            if av != cv:
                v1, v2, h1, h2 = (a, b, c, d) if av else (c, d, a, b)
                crossing = (min(v1[1], v2[1]) <= h1[1] <= max(v1[1], v2[1])
                            and min(h1[0], h2[0]) <= v1[0] <= max(h1[0], h2[0]))
            elif av:
                crossing = a[0] == c[0] and max(min(a[1], b[1]), min(c[1], d[1])) <= min(max(a[1], b[1]), max(c[1], d[1]))
            else:
                crossing = a[1] == c[1] and max(min(a[0], b[0]), min(c[0], d[0])) <= min(max(a[0], b[0]), max(c[0], d[0]))
            assert not crossing, (name, 'crosses', other)


def read_snapshot():
    base = ROOT / 'outputs/runs/put_object_cabinet'
    pure = base / 'dev_30k_isolated700_seed10002_20261005_2041'
    transparent = base / 'ckpt30k_seed10002_transparent_20261006_0440'
    retry = base / 'ckpt20k_seed10002_onboard_retry_20261006_0310'
    paths = {
        'equivalence': ROOT / 'outputs/validation/checkpoint_sweep_analysis_20261005/transparent_real_equivalence.json',
        'onboard': transparent / 'episode_000_seed10002/onboard/manifest.json',
        'gpu': ROOT / 'outputs/validation/robotwin_visible_probe_gpu_20261006_0420/result.json',
        'pure': pure / 'result.json', 'transparent': transparent / 'result.json', 'retry': retry / 'result.json',
        'queue': ROOT / 'outputs/validation/checkpoint_sweep_20261005_2200/queue_status.json',
        'training': base / 'wam_30k_20261005/status.json',
    }
    data = {k: json.loads(p.read_text()) for k, p in paths.items()}
    eq, onboard, records = data['equivalence'], data['onboard'], data['gpu']['records']
    assert eq['baseline_steps'] == eq['transparent_steps'] == 304 and eq['raw_actions_bitwise_equal'] is True
    assert eq['max_raw_abs_difference'] == 0 and data['training']['step'] == 30000
    assert data['training']['status'] == 'bounded_complete'
    for mode in ('pure', 'transparent'):
        ep = data[mode]['episodes'][0]
        assert ep['seed'] == 10002 and ep['status'] == 'native_success' and ep['control_steps'] == 304
    assert onboard['complete'] and onboard['lossless'] and onboard['policy_inputs_unchanged'] and onboard['frames'] == 425
    assert onboard['fields'] == ['head_camera', 'left_camera', 'right_camera', 'commanded_joint_gripper_state']
    assert len(records) == 6 and all(r['response']['outcome'] == 'ok' for r in records)
    states = [v['state'] for r in records for v in r['response']['parsed'].values()]
    assert len(states) == 18 and set(states) == {'unknown'} and data['gpu']['status'] == 'advisory_unadmitted'
    counts = {}
    for step in (15000, 20000, 25000, 30000):
        jobs = [j for j in data['queue']['jobs'] if j['checkpoint_step'] == step]
        valid = [j for j in jobs if j['status'] == 'valid_terminal']
        counts[str(step)] = dict(native_successes=sum(j['native_successes'] for j in valid),
                                valid_terminal=len(valid), infrastructure_incomplete=len(jobs)-len(valid))
    counts['30000']['native_successes'] += 1
    counts['30000']['valid_terminal'] += 1  # Separate registered reference run, not an extra seed.
    assert counts == {'15000': {'native_successes': 0, 'valid_terminal': 3, 'infrastructure_incomplete': 0},
                      '20000': {'native_successes': 0, 'valid_terminal': 2, 'infrastructure_incomplete': 1},
                      '25000': {'native_successes': 0, 'valid_terminal': 3, 'infrastructure_incomplete': 0},
                      '30000': {'native_successes': 1, 'valid_terminal': 3, 'infrastructure_incomplete': 0}}
    ep = data['retry']['episodes'][0]
    assert ep['seed'] == 10002 and ep['status'] == 'step_limit' and ep['control_steps'] == 700
    frozen = json.loads((BACKUP / 'backup_manifest.json').read_text())['supervisor_frozen']
    assert all(hashlib.sha256((ROOT / n).read_bytes()).hexdigest() == sha for n, sha in frozen.items()), 'Supervisor changed after freeze'
    return {
        'as_of': '2026-10-06 architecture handoff; artifact snapshots, not live telemetry',
        'training': {'status': 'bounded_complete', 'step': 30000, 'heldout_used': False},
        'original_dev_sweep': counts,
        'separate_20k_retry': {'run': str(retry.relative_to(ROOT)), 'seed': 10002, 'status': 'step_limit',
                             'control_steps': 700, 'native_success': False, 'not_merged_into_original_sweep': True},
        'equivalence': {**eq, 'seed': 10002, 'baseline_run': str(pure.relative_to(ROOT)),
                        'transparent_run': str(transparent.relative_to(ROOT)), 'system_uplift_demonstrated': False},
        'onboard': onboard,
        'supervisor': {'cpu_tests_passed': 41, 'frozen_source_sha256': frozen, 'real_provider_admitted': False,
                       'test_provenance': 'Final CPU41 delivery and main-thread confirmation; not rerun for diagram'},
        'gpu_visual_probe': {'model': 'Qwen2.5-VL-3B', 'split': 'TRAIN', 'cases': 6, 'parsed': 6,
                             'predicate_states': {'unknown': 18, 'true': 0, 'false': 0},
                             'status': 'advisory_unadmitted', 'visual_validation_pass': False},
        'new_system_evaluation_authorization': {'source': '2026-10-06 user instruction', 'paired_instances': 10,
                  'conditional_extension_to': 20, 'started': False, 'provider_admission_required': True},
        'native_vs_strict': {mode: {
            'native_success': data[mode]['episodes'][0]['evidence']['native_benchmark_success'],
            'strict_stable': data[mode]['episodes'][0]['evidence']['stable_after_settle'],
            'full_containment': data[mode]['episodes'][0]['evidence']['full_containment'],
            'system_success': None} for mode in ('pure', 'transparent')},
    }, list(paths.values())


def main():
    validate()
    snapshot, evidence_files = read_snapshot()
    old = json.loads((BACKUP / 'outputs/architecture/topology.json').read_text())
    assert old['nodes'] == {k: list(v) for k, v in NODES.items()}, 'Node coordinates changed'
    assert [(e['source'], e['target'], e['points']) for e in old['edges']] == [
        (s, t, [list(p) for p in points]) for s, t, _, points, _ in EDGES], 'Routes changed'
    OUT.mkdir(parents=True, exist_ok=True)
    chunks = ['''<svg xmlns="http://www.w3.org/2000/svg" width="2048" height="1220" viewBox="0 0 2048 1220">
<title>Fixed Bimanual WAM</title>
<desc>2026-10-06 architecture snapshot. Nine nodes and seventeen route coordinates are unchanged. Modes are pure and transparent, not a tested full system. Same seed10002: pure and transparent each achieved native success in 304 steps, raw actions bitwise equal. Transparent retains 425 lossless onboard frames of original head/left/right RGB and 14D commanded state; these are not measured joint positions. Whole-task WAM has a fixed 1024D zero condition, chunk32/exec8 and unchanged temporal history. Training completed at 30k. Original fixed-dev sweep: 30k 1/3, 15k 0/3, 25k 0/3; 20k two failures plus one infrastructure-incomplete run; the failed20k retry is separate. Task eligibility, bound external ASK replies, three-valued visible outcomes, bounded recovery callbacks and scoped Memory passed 41 CPU tests; actual independent providers remain unadmitted. GPU Qwen2.5-VL-3B parsed six TRAIN cases, all eighteen predicates unknown: parser completion is not visual validation or admission. Full-system ten paired instances, conditionally twenty, are newly authorized but not started; admission remains to be supplied. Native/GT scoring is isolated in the world/evaluator boundary, never fed into policy, verification, recovery or memory. Native success is not strict stability, complete containment or system benefit. Solid routes indicate explicit contracts or connected native pure/transparent execution; dashed routes indicate CPU/probe-only supervision or unadmitted provider links.</desc>
<defs>
<linearGradient id="physical" x2="0.9" y2="1"><stop stop-color="#eef8fa"/><stop offset="1" stop-color="#dae9f0"/></linearGradient>
<linearGradient id="latent" x2="1" y2="1"><stop stop-color="#f2effc"/><stop offset="1" stop-color="#e0e5fa"/></linearGradient>
''']
    # First three documented dataviz slots; all-pairs validator passes on white.
    # Feedback has a sub-3:1 contrast warning, relieved by labels and the table
    # in system_design.md. These are route identities, never success colors.
    colors = {'data': '#2a78d6', 'action': '#eb6834', 'feedback': '#1baf7a'}
    for kind, color in colors.items():
        chunks.append(f'<marker id="arrow-{kind}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M 0 1 L 9 5 L 0 9 z" fill="{color}"/></marker>')
    chunks.append('</defs><style>text{font-family:DejaVu Sans,sans-serif;fill:#263544}.small{font-size:15px}.body{font-size:16px}.title{font-size:23px;font-weight:600}.edge-label{font-size:15px}</style><rect width="2048" height="1220" fill="white"/>')

    def text(x, y, value, cls='body', extra=''):
        chunks.append(f'<text x="{x}" y="{y}" class="{cls}" {extra}>{escape(value)}</text>')

    text(50, 48, 'WAM V3  |  Fixed bimanual cabinet task', extra='style="font-size:32px;font-weight:600"')
    text(50, 80, 'put_object_cabinet: grasp object + open / hold drawer + place inside | continuous learned control, not a staged runtime', 'body')
    text(1980, 48, '2026-10-06 · PURE / TRANSPARENT', 'small', 'text-anchor="end"')
    for source, target, name, points, kind in EDGES:
        path = 'M ' + ' L '.join(f'{x},{y}' for x, y in points)
        dash = ' stroke-dasharray="4 7"' if name in PENDING else ''
        start = f' marker-start="url(#arrow-{kind})"' if name in BIDIRECTIONAL else ''
        status = edge_status(name)
        chunks.append(f'<g id="edge-{name}" data-source="{source}" data-target="{target}"><title>{escape(name + ": " + status)}</title><path d="{path}" fill="none" stroke="{colors[kind]}" stroke-width="2" stroke-linejoin="round" marker-end="url(#arrow-{kind})"{start}{dash}/></g>')

    def card(node, title, badge, lines, fill, accent):
        x, y, w, h = NODES[node]
        chunks.append(f'<g id="{node}"><rect x="{x}" y="{y+4}" width="{w}" height="{h}" rx="16" fill="#263544" opacity="0.045"/><rect x="{x}" y="{y}" width="{w}" height="{h}" rx="16" fill="{fill}" stroke="{accent}" stroke-width="1.3"/>')
        text(x+21, y+30, badge, 'small', 'style="font-weight:600;font-size:12px;letter-spacing:1px"')
        text(x+21, y+67, title, 'title', 'style="font-size:22px"')
        for i, line in enumerate(lines):
            text(x+21, y+102+i*27, line, 'body')
        chunks.append('</g>')

    card('world', 'RoboTwin cabinet', 'FIXED ALOHA · RIGID DRAWER',
         ['One arm: grasp + place object', 'Other arm: open + hold drawer', '500 episodes · 400 / 50 / 50'], 'url(#physical)', '#2a78d6')
    # Local schematic only: reuse the physical-anchor slot, but remove the old
    # mobile base. Two fixed arm mounts face a sliding drawer, not a hinged lid.
    chunks.append('''<g id="robot-schematic" stroke="#587681" stroke-width="3" stroke-linejoin="round">
<title>Conceptual fixed dual-arm tabletop and open drawer; not a simulator rendering or success record</title>
<path d="M80 534 L174 480 L323 524 L230 585 Z" fill="#dae9f0" stroke-width="1.2"/>
<path d="M97 543 L97 609 M300 539 L300 605" fill="none"/>
<path d="M160 437 L183 420 L291 420 L268 437 Z" fill="#d5e3e9"/>
<path d="M268 437 L291 420 L291 479 L268 495 Z" fill="#91acb8"/>
<rect x="160" y="437" width="108" height="58" rx="3" fill="#eef5f7"/>
<rect x="166" y="449" width="96" height="36" rx="2" fill="#829da8"/>
<path d="M165 464 L254 464 L244 516 L151 516 Z" fill="#b9d1da"/>
<path d="M151 516 L244 516 L244 530 L151 530 Z" fill="#91acb8"/>
<path d="M186 523 L209 523" fill="none"/>
<rect x="90" y="554" width="34" height="14" rx="3" fill="#829da8"/>
<rect x="274" y="559" width="34" height="14" rx="3" fill="#829da8"/>
<path d="M106 554 L114 505 L146 465 M291 559 L301 507 L233 524" fill="none" stroke-width="10"/>
<path d="M140 460 L150 449 M151 468 L162 457 M231 517 L218 520 M234 530 L221 533" fill="none" stroke-width="4"/>
<rect x="152" y="440" width="14" height="14" rx="2" fill="#b9d1da"/>
</g>''')
    chunks.append('<g id="evaluation-audit" data-node="world">')
    text(71, 646, 'Concept schematic, not a rollout', 'small')
    text(71, 700, 'ISOLATED NATIVE SCORER', 'small', 'style="font-weight:600"')
    text(71, 730, 'GT never enters policy / memory', 'small')
    text(71, 758, '30k: 1/3 · 15k, 25k: 0/3', 'small')
    text(71, 786, '20k: 2 fail + 1 infra', 'small')
    text(71, 817, '20k retry: failed (separate)', 'small')
    chunks.append('</g>')
    card('vlm', 'Goal grounding', 'GPU PROBE · NOT ADMITTED',
         ['3B · 6 TRAIN cases parsed', 'All six cases: UNKNOWN', 'Parsing ≠ visual validation'], '#f5f5f6', '#9098a0')
    card('orch', 'Task supervisor', 'CPU 41 PASS · LIVE PENDING',
         ['Whole-task capability gate', 'ASK / three-valued outcomes', 'Recovery + scoped ledger', 'Zero cond; no skill stages'], '#eef8fa', '#2a78d6')
    card('nav', 'Aloha I/O contract', 'NATIVE 14D · FIXED BASE',
         ['2 × (6 arm joints + 1 gripper)', 'State = controller targets', 'Not measured joint qpos', 'History uses observation indices'], '#eef8fa', '#2a78d6')
    card('wam', 'WAM V3 · native 14D', '30K COMPLETE · LIVE INFERENCE',
         ['3-view DINO · chunk32 / exec8', 'Temporal history + future head', 'No VTT / language condition'], 'url(#latent)', '#2a78d6')
    chunks.append('<g id="pending-navigation" data-node="nav"><rect x="1194" y="400" width="312" height="38" rx="8" fill="none" stroke="#9098a0" stroke-dasharray="4 7"/>')
    text(1204, 424, '15 Hz nominal ≠ measured clock', 'small')
    chunks.append('</g><g id="pending-manip-pilot" data-node="wam"><rect x="1194" y="817" width="312" height="27" rx="8" fill="none" stroke="#9098a0" stroke-dasharray="4 7"/>')
    text(1204, 836, 'Native ≠ full-system success', 'small')
    chunks.append('</g>')
    card('controller', 'Pure / transparent', 'REAL · SAME-SEED NATIVE SUCCESS',
         ['Same seed: 304 steps in both', 'Raw actions: bitwise identical', 'Onboard raw 3 RGB retained', '14D command-state; no policy GT'], '#fbf1e4', '#eb6834')
    chunks.append('''<g id="control-ownership" data-node="controller">
<rect x="1622" y="475" width="336" height="137" rx="12" fill="#fffaf3" stroke="#eb6834" stroke-width="1.2"/>
<text x="1642" y="502" class="small" style="font-weight:600">TRANSACTION · TRANSPARENT</text>
<text x="1642" y="530" class="body">episode / step / obs-ID binding</text>
<text x="1642" y="558" class="body">Single dispatch ACK; no retry</text>
<text x="1642" y="586" class="small">ACK ≠ task success; no reset trick</text>
<rect x="1622" y="640" width="336" height="132" rx="12" fill="#f7f4fd" stroke="#2a78d6" stroke-width="1.2"/>
<text x="1642" y="670" class="small" style="font-weight:600">UNCHANGED WAM CONTRACT</text>
<text x="1642" y="700" class="body">32 × 14D output; exec8 history</text>
<text x="1642" y="728" class="body">TRAIN stats; zeros(1024)</text>
<text x="1642" y="756" class="small">No actual-qpos substitution / GT</text>
</g>''')
    card('human', 'Bound user reply', 'CPU ONLY · NO LIVE ASK RESULT',
         ['Two visible, distinct choices', 'External reply; no fake user', 'Expired / stale reply → HOLD', 'No WAM target switching'], '#f5f5f6', '#9098a0')
    card('bank', 'Scoped outcome ledger', 'CPU ONLY · NOT V3 HISTORY',
         ['Independent, complete Outcome', 'Scope / version / obs binding', 'Unknown → no memory write', 'No VTT injection / gain claim'], '#f5f5f6', '#9098a0')
    card('predicates', 'Visible evidence admission', 'CPU INTERFACE · REAL PROVIDER NOT ADMITTED',
         ['Trusted + admitted + independent; bound episode / object / obs',
          'Unknown ≠ success; raw VLM stays advisory; no GT-based recovery',
          'Native score is isolated; containment / strict system success unverified'], '#edf5f7', '#9098a0')

    for x, y, label, anchor in [
        (385, 270, 'probe', 'middle'), (740, 270, 'CPU', 'middle'),
        (1145, 285, 'config', 'middle'), (1360, 170, 'CPU gate; live admission pending', 'middle'),
        (1360, 94, 'PURE / TRANSPARENT: native 14D · same seed / 304 steps', 'middle'),
        (1000, 134, 'LIVE: onboard raw 3 RGB + command-state(t) | no GT / no actual-qpos substitution', 'middle'),
        (1560, 265, '14D', 'middle'), (1560, 365, 'clock?', 'middle'),
        (1135, 590, 'zeros(1024)', 'end'),
        (1560, 685, 'live I/O', 'middle'), (1560, 765, 'live out', 'middle'),
        (1145, 385, 'admit?', 'middle'),
        (765, 590, 'CPU · unknown', 'start'),
        (960, 530, 'CPU ledger', 'start'),
        (705, 535, 'CPU ASK', 'end'),
        (740, 350, 'CPU', 'middle'),
        (1660, 980, 'raw RGB; UNADMITTED', 'middle')]:
        text(x, y, label, 'edge-label', f'text-anchor="{anchor}"')
    text(50, 1135, 'Solid: native-connected or explicit contract. Dashed: CPU/probe only; live provider unadmitted. No full-system benefit established.', 'small')
    for x, kind in [(50, 'data'), (240, 'action'), (440, 'feedback')]:
        chunks.append(f'<path d="M{x} 1156 h32" stroke="{colors[kind]}" stroke-width="2"/>')
        text(x+42, 1161, kind, 'small')
    text(660, 1161, 'Modes: pure / transparent · 9 nodes / 17 routes · snapshot, not live telemetry', 'small')
    text(50, 1185, 'AUTHORIZED, NOT STARTED: full-system 10 paired instances; conditional extension to 20. Provider admission remains required.', 'small')
    text(50, 1205, 'Native success ≠ strict/system success. 3B / 6 TRAIN: parsed, all UNKNOWN. No visual pass or system gain claimed.', 'small')
    chunks.append('</svg>')
    svg = '\n'.join(chunks)
    document = ET.fromstring(svg)
    ns = '{http://www.w3.org/2000/svg}'
    assert not any(e.tag in {ns+'image', ns+'feImage', ns+'foreignObject'} for e in document.iter()), 'Vector only'
    for i, label in enumerate(document.iter(ns+'text')):
        label.set('id', f'label-{i}')
    for source, target, name, _, _ in EDGES:
        group = document.find(f"{ns}g[@id='edge-{name}']")
        assert (group.get('data-source'), group.get('data-target')) == (source, target)
        path = group.find(ns+'path')
        assert ('stroke-dasharray' in path.attrib) == (name in PENDING)
        assert ('marker-start' in path.attrib) == (name in BIDIRECTIONAL)
        kind = next(k for _, _, n, _, k in EDGES if n == name)
        assert path.get('marker-end') == f'url(#arrow-{kind})'
        marker = document.find(f"{ns}defs/{ns}marker[@id='arrow-{kind}']")
        assert marker is not None and marker.get('orient') == 'auto-start-reverse'
    assert document.find(f"{ns}g[@id='pending-navigation']/{ns}rect").get('stroke-dasharray')
    assert document.find(f"{ns}g[@id='pending-manip-pilot']/{ns}rect").get('stroke-dasharray')
    ET.register_namespace('', ns[1:-1])
    svg = ET.tostring(document, encoding='unicode')
    svg_path = OUT / 'wam_vlm_architecture.svg'
    svg_path.write_text(svg, encoding='utf-8')
    import gi
    gi.require_version('Rsvg', '2.0')
    from gi.repository import Rsvg
    import cairo
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 3072, 1830)
    context = cairo.Context(surface)
    handle = Rsvg.Handle.new_from_file(str(svg_path))
    viewport = Rsvg.Rectangle()
    viewport.x, viewport.y, viewport.width, viewport.height = 0, 0, 3072, 1830
    assert handle.render_document(context, viewport)
    # Check actual librsvg ink bounds, not guessed character widths. Same renderer as the PNG.
    bounds_view = Rsvg.Rectangle()
    bounds_view.x, bounds_view.y, bounds_view.width, bounds_view.height = 0, 0, 2048, 1220
    owners = {}
    for group in document.iter(ns+'g'):
        node = group.get('data-node', group.get('id'))
        if node in NODES:
            owners.update({t.get('id'): node for t in group.iter(ns+'text')})
    boxes = []
    for label in document.iter(ns+'text'):
        ok, ink, _ = handle.get_geometry_for_layer('#'+label.get('id'), bounds_view)
        assert ok
        x, y, right, bottom = ink.x, ink.y, ink.x+ink.width, ink.y+ink.height
        detail = ''.join(label.itertext())
        assert 0 <= x <= right <= 2048 and 0 <= y <= bottom <= 1220, ('text outside canvas', detail)
        owner = owners.get(label.get('id'))
        for node, (nx, ny, w, h) in NODES.items():
            if node == owner:
                assert nx+8 <= x and right <= nx+w-8 and ny+8 <= y and bottom <= ny+h-8, ('text overflow', node, detail)
            else:
                assert right <= nx or nx+w <= x or bottom <= ny or ny+h <= y, ('label overlaps node', node, detail)
        for ox, oy, ore, ob, other in boxes:
            assert right <= ox or ore <= x or bottom <= oy or ob <= y, ('text overlap', detail, other)
        for _, _, name, points, _ in EDGES:
            for a, b in zip(points, points[1:]):
                assert (max(a[0], b[0]) < x or right < min(a[0], b[0])
                        or max(a[1], b[1]) < y or bottom < min(a[1], b[1])), ('text crosses route', detail, name)
        boxes.append((x, y, right, bottom, detail))
    png_path = OUT / 'wam_vlm_architecture.png'
    surface.write_to_png(str(png_path))
    png = cairo.ImageSurface.create_from_png(str(png_path))
    assert (png.get_width(), png.get_height()) == (3072, 1830)
    assert bytes(png.get_data()) == bytes(surface.get_data()), 'PNG differs from current SVG render'
    topology_path = OUT / 'topology.json'
    evidence = {
        'rgb_for_optional_grounding': 'GPU advisory probe only: 3B parsed 6 TRAIN cases, all 18 judgments unknown. Recorded onboard raw 3RGB exists, but no admitted live visual grounding provider.',
        'optional_goal_programs': 'TaskSupervisor.ground: CPU-tested GroundingResult execute/ask/unsupported with freshness and scope checks; whole-task capability only, no skill program or live-admitted VLM.',
        'fixed_task_and_embodiment_config': 'Whole-task zero_single_task, native state/action14, three RGB, chunk32/exec8. ModelContract and explicit configuration; fixed base, not a navigation process.',
        'runtime_execute_hold_reset': 'Legacy route ID retained, NOT a reset command: CPU task-eligibility/HOLD directives only. Transparent transactions have live evidence; high-level supervisor is not live-admitted and has no policy reset handle.',
        'live_14d_physical_command': 'CONNECTED pure/transparent: real same-seed native success at 304 steps in each; raw actions bitwise equal. Existing bounds/native TOPP retained. No system uplift or safety-HOLD claim.',
        'offline_rgb_t_target_state_t_command_t1': 'Legacy route ID now includes LIVE sensor ingress: same onboard head/left/right raw RGB plus 14D commanded state, not measured qpos or GT. Transparent lossless onboard manifest:425 frames. Offline training alignment remains RGB(t)/command-state(t)/command(t+1).',
        'native_14d_schema': 'Explicit Aloha order: left6 + gripper + right6 + gripper; validate/bounds/native driver path. No old61D/23D padding or command-to-measured-state substitution.',
        'live_timing_and_joint_feedback': 'NOT ADMITTED as a physical timing/feedback guarantee: onboard manifest has wall-monotonic capture time, not simulation time; 15Hz nominal and command-state do not establish actual qpos or fixed-rate control.',
        'physical_readiness_status': 'CPU eligibility/scope checks are not physical readiness. No fake confidence/calibration, GT stage or measured-state claim.',
        'zero_single_task_condition': 'WAM always receives zeros(1024) in both modes; transparent adapter and supervisor do not inject VTT, language, stage, user-reply or memory condition vectors.',
        'three_view_features_and_14d_history': 'CONNECTED in both modes: DINO three views, native14 command history, current/lags/episode-start; every policy observation enters history even while queued. Transparent same-seed equivalence does not change exec8.',
        'sampled_32x14d_chunks': 'CONNECTED: real frozen30k WAM chunk32/exec8 sampled actions, baseline and transparent raw bitwise equality across304 steps. Not replayed demonstrations; not a measured30Hz or system-improvement result.',
        'independent_sensor_completion_status': 'CPU evaluate_visible requires trusted/admitted/independent provider, matching version/scope/episode/object/obs and a complete Outcome. Unknown never completes. Actual providers not admitted.',
        'scoped_experience_read_write': 'CPU core.Memory boundary only: independent complete Outcome, exact scope/version; unknown not written. No demonstrated online memory benefit, no VTT promotion, no GT retrieval into policy.',
        'optional_ask_and_bound_reply': 'CPU only: two visible candidates must imply distinct eligibility decisions; externally supplied UserReply binds question/episode/obs/deadline. Stale reply HOLD, no fabricated user and no unsupported object retargeting.',
        'optional_recovery_replan': 'CPU callback contract only: independent visible failure/permission; passive reobserve or replan request never resets WAM history. Missing capability/failed ACK HOLD; no blind retry or live recovery claim.',
        'sensor_bound_outcome_evidence': 'DASHED admission boundary: raw onboard3RGB capture exists, but live independent sensor Outcome provider is not admitted. This route is NOT evaluator GT; native/GT scoring stays isolated in world.',
    }
    assert evidence.keys() == {e[2] for e in EDGES}
    roles = {
        'world': 'fixed-Aloha task/data plus isolated native/GT evaluation; sweep and separate retry results',
        'vlm': '3B GPU6-TRAIN parser probe: all unknown, advisory/unadmitted, not visual validation',
        'orch': 'frozen CPU whole-task TaskSupervisor: eligibility/ASK/three-valued verification/recovery/Memory; no admitted real providers',
        'nav': 'native14 commanded-state I/O contract; former navigation slot, no mobile policy',
        'controller': 'pure or transparent native execution; same-seed304-step raw-action equivalence and raw onboard3RGB',
        'human': 'CPU external bound UserReply interface, not autonomous user generation or measured interaction benefit',
        'bank': 'CPU scoped independent-outcome ledger, distinct from V3 temporal history; no memory benefit claim',
        'wam': '30k complete; frozen whole-task native14/3RGB/zero1024/chunk32-exec8',
        'predicates': 'CPU visible-evidence admission gate, providers unadmitted; never native evaluator GT',
    }
    cpu_interfaces = {'optional_goal_programs', 'runtime_execute_hold_reset', 'physical_readiness_status',
                      'independent_sensor_completion_status', 'scoped_experience_read_write',
                      'optional_ask_and_bound_reply', 'optional_recovery_replan'}
    levels = {n: ('cpu_validated_not_live_admitted' if n in cpu_interfaces else
                  'gpu_probe_advisory' if n == 'rgb_for_optional_grounding' else
                  'live_capture_provider_unadmitted' if n == 'sensor_bound_outcome_evidence' else
                  'timing_unverified' if n == 'live_timing_and_joint_feedback' else
                  'connected_native' if n in CONNECTED else 'explicit_contract') for _, _, n, _, _ in EDGES}
    modes = {
        'pure': {'supervisor': 'off', 'transaction_wrapper': 'off', 'condition': 'zeros(1024)',
                 'live_native_reference': snapshot['equivalence']['baseline_run'], 'system_uplift_claim': False},
        'transparent': {'supervisor': 'off / not admitted', 'transaction_wrapper': 'on', 'condition': 'zeros(1024)',
                        'live_native_reference': snapshot['equivalence']['transparent_run'],
                        'onboard_raw_three_rgb': True, 'system_uplift_claim': False},
    }
    topology_path.write_text(json.dumps({
        'nodes': NODES,
        'edges': [dict(source=src, target=dst, meaning=name, points=points, kind=kind,
                       implemented=name not in PENDING, cpu_interface_implemented=name in cpu_interfaces,
                       implementation_level=levels[name], status=edge_status(name),
                       bidirectional=name in BIDIRECTIONAL, semantic_evidence=evidence[name],
                       modes={mode: ('not_active_or_not_admitted' if name in PENDING else
                                      'connected_native' if name in CONNECTED else 'explicit_contract') for mode in modes})
                  for src, dst, name, points, kind in EDGES],
        'scope': '2026-10-06 fixed layout; only pure/transparent native modes validated. CPU/probe interfaces are not live system benefits.',
        'mode_definitions': modes, 'node_roles': roles, 'snapshot': snapshot,
        'legacy_route_ids': {'runtime_execute_hold_reset': 'eligibility/HOLD interface; NO policy reset command',
                             'offline_rgb_t_target_state_t_command_t1': 'live sensor ingress plus unchanged offline alignment'},
        'geometry_reference': str(BACKUP.relative_to(ROOT) / 'outputs/architecture/topology.json'),
        'validation': 'unchanged boxes/routes; ports, orthogonal paths, no penetration/crossings, actual text ink, SVG/PNG equality',
    }, indent=2) + '\n')

    def digest(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    hashes = {p.name: digest(p) for p in (svg_path, png_path, topology_path)}
    visual_matches = bool(VISUAL_REVIEW and VISUAL_REVIEW.get('svg_sha256') == hashes[svg_path.name]
                          and VISUAL_REVIEW.get('png_sha256') == hashes[png_path.name])
    (OUT / 'validation.json').write_text(json.dumps({
        'checked_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'Architecture-only CPU rendering/QA; no new model, simulator or system-capability experiment',
        'nodes': len(NODES), 'edges': len(EDGES),
        'checks': {
            'same_node_coordinates_as_archive': True, 'same_edge_coordinates_as_archive': True,
            'source_destination_ports': 'pass', 'orthogonal_routes': 'pass',
            'node_penetration': 'none', 'arrow_crossings_or_overlaps': 'none',
            'arrowheads_and_bidirectional_orientation': 'pass',
            'svg_xml': 'pass', 'raster_layers_in_publication_svg': 0,
            'canvas_and_text_bounds': 'pass', 'text_overlaps_or_route_crossings': 'none',
            'rendered_text_labels_checked': len(boxes), 'implemented_pending_arrow_styles': 'pass',
            'png_size': [3072, 1830], 'png_rendered_from_current_svg': True,
            'manual_image_inspection': {'status': 'pass' if visual_matches else 'pending',
                'review': VISUAL_REVIEW if visual_matches else None,
                'note': 'Hash-bound visual review; changing the SVG/PNG invalidates the review.'},
        },
        'routes': {n: dict(source=src, target=dst, source_port=pts[0], target_port=pts[-1], result='pass')
                   for src, dst, n, pts, _ in EDGES},
        'semantic_edge_evidence': evidence, 'implementation_levels': levels,
        'pending_interfaces': sorted(PENDING), 'mode_definitions': modes, 'snapshot': snapshot,
        'pending_pilot': [
            'Actual independent grounding/visible-outcome providers remain unadmitted; CPU41 is not physical verification.',
            'GPU3B six TRAIN cases parsed, all eighteen predicates unknown: no visual admission or success accuracy claim.',
            'Authorized full-system ten paired instances, conditionally twenty, are not started; admission prerequisites remain.',
            'Strict containment, release/stability, timing and live recovery/asking/memory benefits remain unverified.',
        ],
        'runtime_limits': [
            'Native success and same-seed transparent equivalence are not full-system uplift or generalization.',
            'GT stays evaluator-only; the controller-to-predicates route now denotes only unadmitted visible sensor input.',
            'Commanded state is not measured qpos. Onboard capture wall time is not simulation time.',
            'Condition remains zero; no dynamic skill stage, VTT/memory condition or fabricated user reply.',
            '20k original infrastructure-incomplete instance and its failed retry remain separate records.',
        ],
        'palette': {'mode': 'unchanged light SVG/PNG', 'route_colors': colors,
                    'note': 'Status also encoded by dashes and explicit labels, not color alone; no new CVD certification claimed.'},
        'backup_manifest': str((BACKUP / 'backup_manifest.json').relative_to(ROOT)),
        'evidence_sha256': {str(path.relative_to(ROOT)): digest(path) for path in evidence_files + [
            ROOT / 'scripts/draw_architecture.py', ROOT / 'harness/robotwin_task_supervisor.py',
            ROOT / 'scripts/test_robotwin_task_supervisor.py']},
        'sha256': hashes,
    }, indent=2) + '\n')
    print(f'PASS: {len(NODES)} modules, {len(EDGES)} unchanged routes; SVG + 3072×1830 PNG; visual review={visual_matches}')


if __name__ == '__main__':
    main()
