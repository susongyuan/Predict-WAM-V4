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
# offline contract or connected simulator interface; dashed means OFF/unverified.
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
    'optional_ask_and_bound_reply', 'optional_recovery_replan',
}
CONNECTED = {'live_14d_physical_command', 'three_view_features_and_14d_history',
             'sampled_32x14d_chunks', 'sensor_bound_outcome_evidence'}

def edge_status(name):
    return ('inactive / unverified' if name in PENDING else
            'connected in real simulator; native success is not system validation' if name in CONNECTED else
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


def main():
    validate()
    OUT.mkdir(parents=True, exist_ok=True)
    chunks = ['''<svg xmlns="http://www.w3.org/2000/svg" width="2048" height="1220" viewBox="0 0 2048 1220">
<title>Fixed Bimanual WAM</title>
<desc>2026-10-05 latest user-reported outcome snapshot, not live telemetry or an independent audit of run artifacts. Existing nine nodes and seventeen routes retained. 30k training complete. 30k seed10002 achieved native success in 304 steps / 340.1 seconds, with 425 frames at 1080p and 425 matching independent predicate recomputations. After settling native=true and contact separation=true; strict stable=false and full containment unknown. This is native benchmark success, not full-system or strict physical success. Earlier dev10000 failed within 500 steps, not the complete native700 budget; earlier seed10001 blocked after action141 and was externally terminated. Other seeds and 30k/15k/20k/25k development queues are planned to start, not completed or passed. Policy child and sim process are separate; per-call journal plus per-seed external hard timeout are connected. Only 3 RGB views and 14D commanded state enter policy, not GT or measured qpos; zero task condition. VLM/asking/recovery/experience remain OFF. Native predicate and contact separation do not prove strict containment, release or five-second hold. Offline split: 400 train / 50 val / 50 heldout sealed. Solid routes distinguish connected and offline contracts by labels; dashed routes are OFF/unverified.</desc>
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
    text(1980, 48, '2026-10-05 · LATEST REPORTED RESULT', 'small', 'text-anchor="end"')
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
    text(71, 700, 'INDEPENDENT GT EVALUATION', 'small', 'style="font-weight:600"')
    text(71, 730, 'GT scorer connected; no policy GT', 'small')
    text(71, 758, 'No GT → state / cond / train split', 'small')
    text(71, 786, '30k / 10002: NATIVE SUCCESS', 'small')
    text(71, 817, 'Not hinged-lid / deformable task', 'small')
    chunks.append('</g>')
    card('vlm', 'Goal grounding', 'OFF · EXTERNAL VLM',
         ['Not used by cabinet training', 'No language-to-skill dispatch', 'Future integration only'], '#f5f5f6', '#9098a0')
    card('orch', 'Single-task context', 'CURRENT · ZERO CONDITION',
         ['One continuous cabinet task', 'cond = zeros(1024)', 'No runtime skill-stage switch', 'Not a GT phase controller'], '#eef8fa', '#2a78d6')
    card('nav', 'Aloha I/O contract', 'NATIVE 14D · FIXED BASE',
         ['2 × (6 arm joints + 1 gripper)', 'State = controller targets', 'Not measured joint qpos', 'History uses observation indices'], '#eef8fa', '#2a78d6')
    card('wam', 'WAM V3 · native 14D', 'OFFLINE · 30K COMPLETE',
         ['3-view frozen DINO · chunk 32', 'Built-in history + future head', '318 reused / 6 reset entries'], 'url(#latent)', '#2a78d6')
    chunks.append('<g id="pending-navigation" data-node="nav"><rect x="1194" y="400" width="312" height="38" rx="8" fill="none" stroke="#9098a0" stroke-dasharray="4 7"/>')
    text(1204, 424, '15 Hz nominal; live timing pending', 'small')
    chunks.append('</g><g id="pending-manip-pilot" data-node="wam"><rect x="1194" y="817" width="312" height="27" rx="8" fill="none" stroke="#9098a0" stroke-dasharray="4 7"/>')
    text(1204, 836, 'Native success ≠ system pass', 'small')
    chunks.append('</g>')
    card('controller', 'Data / control adapter', 'CONNECTED · PURE WAM ROLLOUT',
         ['Input: 3 RGB + 14D command', 'Policy child / sim process split', 'No measured qpos input', 'No evaluator GT in policy fields'], '#fbf1e4', '#eb6834')
    chunks.append('''<g id="control-ownership" data-node="controller">
<rect x="1622" y="475" width="336" height="137" rx="12" fill="#fffaf3" stroke="#eb6834" stroke-width="1.2"/>
<text x="1642" y="502" class="small" style="font-weight:600">CONNECTED · ISOLATED EXECUTION</text>
<text x="1642" y="530" class="body">Per-call atomic phase journal</text>
<text x="1642" y="558" class="body">Per-seed unit: external hard timeout</text>
<text x="1642" y="586" class="small">Block root cause remains unverified</text>
<rect x="1622" y="640" width="336" height="132" rx="12" fill="#f7f4fd" stroke="#2a78d6" stroke-width="1.2"/>
<text x="1642" y="670" class="small" style="font-weight:600">OFFLINE CONTRACT / LIVE SAMPLE</text>
<text x="1642" y="700" class="body">3 RGB + 14D command history</text>
<text x="1642" y="728" class="body">TRAIN stats; action / future masks</text>
<text x="1642" y="756" class="small">32 × 14D output; no base command</text>
</g>''')
    card('human', 'Intent clarification', 'OFF · NO ASKING',
         ['No user-query runtime enabled', 'No interaction success claim', 'Later: bind reply to episode', 'Not needed for current training'], '#f5f5f6', '#9098a0')
    card('bank', 'Experience memory', 'OFF · NOT V3 HISTORY',
         ['No online experience writes', 'No retrieval-driven improvement', 'Internal temporal memory ≠ this', 'No GT or unverified promotion'], '#f5f5f6', '#9098a0')
    card('predicates', 'Evaluator-only outcome evidence', 'CONNECTED SCORER · STRICT SUCCESS UNVERIFIED · RECOVERY OFF',
         ['425 / 425 independent predicate recomputations match', 'Post-settle native=true; contact separation=true', 'Strict stable=false; full containment UNKNOWN; no policy GT'], '#edf5f7', '#9098a0')

    for x, y, label, anchor in [
        (385, 270, 'RGB off', 'middle'), (740, 270, 'off', 'middle'),
        (1145, 285, 'config', 'middle'), (1360, 170, 'runtime EXECUTE / HOLD pending', 'middle'),
        (1360, 94, 'CONNECTED: native 14D command · seed10002 native success', 'middle'),
        (1000, 134, 'OFFLINE: RGB(t) + target-state(t) → command(t+1) | no JPEG recompression', 'middle'),
        (1560, 265, 'schema', 'middle'), (1560, 365, 'timing?', 'middle'),
        (1135, 590, 'zero task condition', 'end'),
        (1560, 685, 'live I/O', 'middle'), (1560, 765, 'live out', 'middle'),
        (1145, 385, 'ready?', 'middle'),
        (765, 590, 'verdict pending', 'start'),
        (960, 530, 'experience off', 'start'),
        (705, 535, 'ASK off', 'end'),
        (740, 350, 'off', 'middle'),
        (1660, 980, 'GT evidence · connected', 'middle')]:
        text(x, y, label, 'edge-label', f'text-anchor="{anchor}"')
    # Redundant line-style + direct-label encodings; readable without color.
    text(50, 1135, 'Solid = connected or offline (labelled); dashed = OFF / unverified. Native success ≠ full-system pass. Route colors:', 'small')
    for x, kind in [(50, 'data'), (240, 'action'), (440, 'feedback')]:
        chunks.append(f'<path d="M{x} 1156 h32" stroke="{colors[kind]}" stroke-width="2"/>')
        text(x+42, 1161, kind, 'small')
    text(660, 1161, '17 routes + status evidence: topology.json · fixed layout, not live telemetry', 'small')
    text(50, 1185, '30k seed10002: NATIVE SUCCESS · 304 steps / 340.1 s · 425 frames, 1080p · native success does not establish strict or full-system success.', 'small')
    text(50, 1205, 'Earlier: 10000 failed at 500 (<700); 10001 blocked, externally stopped. Next: other seeds + 30k / 15k / 20k / 25k dev queues PLANNED, not passed.', 'small')
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
        'rgb_for_optional_grounding': 'OFF: external VLM is not part of the current cabinet training or sample path.',
        'optional_goal_programs': 'OFF: no VLM program is consumed; cabinet is one continuous task, not a runtime skill sequence.',
        'fixed_task_and_embodiment_config': 'configs/fixed_bimanual/robotwin_cabinet_500_wam.yaml and scripts/train_robotwin_wam.py load_config: fixed task, native state/action 14, chunk 32; functional config view, not a separate navigation process.',
        'runtime_execute_hold_reset': 'PENDING: live cabinet dispatch, queue reset and HOLD/safety behavior require local closed-loop validation; old mobile policy is not evidence of this wiring.',
        'live_14d_physical_command': 'PENDING: sampled 14D chunks have not established end-to-end physical controller order, units, timing, limits or task success.',
        'offline_rgb_t_target_state_t_command_t1': 'scripts/convert_robotwin_legacy.py and cabinet SOURCE.json: RGB(t), controller-target state(t), command(t+1); omit final observation and copy original JPEG buffers without recompression. This solid route is offline data, not live sensors.',
        'native_14d_schema': 'harness/robotwin_dataset.py, harness/robotwin_training_data.py and cabinet config: left arm 6, left gripper 1, right arm 6, right gripper 1; no base dimensions, no 61D/23D truncation/padding.',
        'live_timing_and_joint_feedback': 'PENDING: measured joint feedback and live capture/control timing require simulator/hardware I/O validation. Legacy state is controller targets; nominal 15 Hz is not a measured clock.',
        'physical_readiness_status': 'PENDING: no physical readiness or high-level phase transitions are implemented for this cabinet baseline.',
        'zero_single_task_condition': 'harness/robotwin_training_data.py RobotwinCachedDataset: task_cond=zeros(1024); no GT phase, future image or language skill condition.',
        'three_view_features_and_14d_history': 'RobotwinCachedDataset -> WamV3Wrapper: frozen cached 3-view DINO, 7x14 state history and temporal visual pool; current+lags+episode start. All lags are observation indices. TRAIN-only statistics, masked episode tails.',
        'sampled_32x14d_chunks': 'scripts/train_robotwin_wam.py validation and smoke_20261005: native 32x14 sample and model save/reload path tested; this is not a live action-dispatch or physical result.',
        'independent_sensor_completion_status': 'PENDING: target-bound sensor completion gate and recovery are not enabled for cabinet. A future gate must treat Unknown as unverified, not success; evaluator GT cannot drive it.',
        'scoped_experience_read_write': 'OFF: external experience store/retrieval is not enabled. V3 internal temporal memory is observation history, not an independently verified experience-improvement mechanism.',
        'optional_ask_and_bound_reply': 'OFF: no external asking or human-reply loop is enabled; historical mobile prototypes do not establish cabinet capability.',
        'optional_recovery_replan': 'OFF: no external VLM recovery replanning or runtime high-level skill-stage switch is enabled.',
        'sensor_bound_outcome_evidence': 'PENDING: grasp, drawer holding, object-inside, release/stability evidence and independent local evaluator wiring need physical verification. No GT return to policy state, condition, split or experience.',
    }
    evidence.update({
        'live_14d_physical_command': 'CONNECTED: pure WAM native 14D execution; dev10000 consumed 500 steps, failed. Native 700 budget incomplete; order/units audit is not a timing or safety guarantee.',
        'three_view_features_and_14d_history': 'CONNECTED: only 3 RGB + 14D commanded joint/gripper state cross into policy child; no measured qpos or GT. Frozen DINO and index-based history; zero task condition.',
        'sampled_32x14d_chunks': 'CONNECTED: real 30k policy sampling, chunk32 / exec8; command targets passed to native TOPP. 30k seed10002 native success reported; not measured fixed-rate control or full-system success.',
        'sensor_bound_outcome_evidence': 'CONNECTED evaluator-only GT: reported seed10002 has 425/425 predicate matches, post-settle native=true, contact separation=true; strict stable=false, full containment unknown. Native predicate is not strict containment/release/5-second hold. No GT feedback to policy or recovery.',
        'runtime_execute_hold_reset': 'UNVERIFIED high-level EXECUTE/HOLD interface; baseline reset/action dispatch is connected, but no high-level orchestrator or verified safety HOLD.',
    })
    assert evidence.keys() == {e[2] for e in EDGES}
    topology_path.write_text(json.dumps({
        'nodes': NODES, 'edges': [dict(source=s, target=t, meaning=n, points=p, kind=k,
                                     implemented=n not in PENDING, status=edge_status(n), bidirectional=n in BIDIRECTIONAL,
                                     semantic_evidence=evidence[n])
                                for s, t, n, p, k in EDGES],
        'validation': 'ports, orthogonal routes, canvas/text bounds, no node/label penetration or crossings, XML, no raster',
        'scope': 'Existing layout retained. Solid: connected or offline contract (status field); dashed: inactive/unverified. Reported seed10002 native success only; no strict or full-system pass.',
        'node_roles': {'world': 'task/data + isolated evaluation boundary', 'vlm': 'optional goal grounding OFF',
                       'orch': 'single-task zero condition', 'nav': 'fixed Aloha I/O contract (former navigation slot)',
                       'controller': 'offline adapter + connected isolated pure-WAM control', 'human': 'asking OFF',
                       'bank': 'experience memory OFF', 'wam': 'native 14D WAM V3',
                       'predicates': 'evaluator-only native evidence connected; strict success unverified; recovery OFF'},
    }, indent=2) + '\n')

    def digest(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    artifacts = [svg_path, png_path, topology_path]
    for optional in (OUT / 'wam_vlm_hybrid_reference.png', OUT / 'wam_vlm_hybrid_reference_pixelmatch.svg',
                     OUT / 'reference/task_agnostic_system_reference_20260927.png',
                     OUT / 'reference/task_agnostic_system_reference_pixelmatch_20260927.svg'):
        if optional.exists():
            artifacts.append(optional)
    (OUT / 'validation.json').write_text(json.dumps({
        'checked_utc': datetime.now(timezone.utc).isoformat(),
        'scope': '2026-10-05 latest user-reported result: 30k complete, seed10002 native success; strict stability false, full containment unknown, full system unvalidated',
        'nodes': len(NODES), 'edges': len(EDGES),
        'checks': {
            'source_destination_ports': 'pass', 'orthogonal_routes': 'pass',
            'node_penetration': 'none', 'arrow_crossings_or_overlaps': 'none',
            'svg_xml': 'pass', 'raster_layers_in_publication_svg': 0,
            'canvas_and_text_bounds': 'pass', 'text_overlaps_or_route_crossings': 'none',
            'rendered_text_labels_checked': len(boxes),
            'implemented_pending_arrow_styles': 'pass', 'png_size': [3072, 1830],
            'png_rendered_from_current_svg': True,
            'manual_image_inspection': {'status': 'pending', 'note': 'Each new render requires visual inspection; automated geometry checks do not substitute for it.'},
        },
        'routes': {n: dict(source=s, target=t, source_port=p[0], target_port=p[-1], result='pass')
                   for s, t, n, p, _ in EDGES},
        'semantic_edge_evidence': evidence,
        'pending_interfaces': sorted(PENDING),
        'pending_pilot': [
            'Other seeds and 30k/15k/20k/25k development queues will start; no pending-queue outcome or aggregate pass claimed.',
            'Native block after action141 was externally terminated; precise call/root cause unresolved. Isolation/journal/hard timeout do not establish a fix.',
            'Strict containment, release, hold duration and physical timing remain unverified; VLM/asking/recovery/experience remain OFF.',
            'No full three-seed rate or formal competition success; rigid drawer proxy only.',
        ],
        'resume_snapshot': {
            'as_of': '2026-10-05 latest user update (time unspecified)', 'source': 'latest user-provided facts supersede earlier progress/report snapshot; no new run-artifact audit',
            'training': 'bounded_complete / step30000',
            'dev10000': '500-step failure; native700 incomplete',
            'dev10001': 'native block after action141 returned; external unit stop; unscored',
            'dev10002': {'checkpoint_step': 30000, 'native_success': True, 'control_steps': 304,
                         'wall_seconds': 340.1, 'video_frames': 425, 'video_resolution': '1080p',
                         'independent_predicate_matches': 425, 'independent_predicate_total': 425,
                         'post_settle_native': True, 'contact_separation': True,
                         'strict_stable': False, 'full_containment': None,
                         'provenance': 'user-reported latest result'},
            'next_queue': 'other seeds and 30k/15k/20k/25k development evaluations',
            'current_status': 'queues planned to start; not full-system pass',
            'isolation': 'policy child + sim process; per-call journal; per-seed unit external hard timeout',
            'budget_seconds': {'episode': 1200, 'total': 1500, 'external': 1560, 'stop_grace': 30},
        },
        'audited_evidence': {
            'dataset_receipt': 'data/benchmarks/robotwin2/put_object_cabinet_500_v1/SOURCE.json',
            'split': {'train': 400, 'val': 50, 'heldout': 50, 'heldout_sealed': True},
            'alignment': 'RGB(t), controller-target state(t), command(t+1); original JPEG buffers unchanged',
            'time_semantics': 'observation_index; nominal 15Hz is not a measured clock',
            'initialization': {'receipt': 'outputs/validation/robotwin_continuous_task_20261004/model_initialization.json',
                               'loaded_parameter_entries': 318, 'reset_parameter_entries': 6,
                               'reset_projections': ['proprio_proj', 'action_proj', 'output_proj'],
                               'old_optimizer_resumed': False},
            'cache': {'episodes': 450, 'splits': ['train', 'val'],
                      'completed_log': 'outputs/validation/robotwin_cabinet_cache_20261005/cache.log'},
            'cuda_smoke': {'status': 'outputs/runs/put_object_cabinet/smoke_20261005/status.json',
                           'updates': 2, 'physical_evaluation': False},
            'training_plan': {'run': 'outputs/runs/put_object_cabinet/wam_30k_20261005',
                              'unit': 'p2-robotwin-cabinet-train-20261005',
                              'first_target': 30000, 'schedule_and_cap': 50000,
                              'last_every': 1000, 'fixed_checkpoint_and_val_every': 5000,
                              'state': 'bounded_complete / step30000; no resumed training'},
        },
        'runtime_limits': [
            'Physical control and evaluator are connected; 500-step dev failure is not a completed native700 trial or task success.',
            'Ground truth stays evaluator-only: no return to state, condition, training split, stage transitions or experience.',
            'State is controller joint targets plus gripper commands, not measured qpos; 15Hz is nominal, not observed timing.',
            'Single-task condition is zero. No runtime high-level skill-stage switch, VLM, asking, recovery or experience memory is enabled.',
            'V3 temporal history and future-feature prediction are model components, not verified experience improvement or physical verification.',
            'Task is fixed-Aloha rigid drawer cooperation, not hinged-bin opening, deformable manipulation or formal competition validation.',
        ],
        'palette': {'mode': 'explicit light-background local SVG/PNG export', 'route_colors': colors,
                    'validation': 'all-pairs pass: CVD minimum 9.2, normal-vision minimum 24.0; feedback contrast 2.82 warning',
                    'relief': 'all routes directly labelled, line-style status legend, full route table in docs/system_design.md'},
        'evidence_sha256': {str(path.relative_to(ROOT)): digest(path) for path in [
            ROOT / 'progress.md',
            ROOT / 'outputs/validation/resume_20261005_1913/scoring_audit.md',
            ROOT / 'outputs/validation/resume_20261005_1913/contract_audit.md',
            ROOT / 'outputs/validation/resume_20261005_1913/external_stop_receipt.json',
            ROOT / 'scripts/draw_architecture.py', ROOT / 'README.md', ROOT / 'docs/system_design.md',
            ROOT / 'harness/wam_v3.py',
            ROOT / 'harness/robotwin_dataset.py', ROOT / 'harness/robotwin_training_data.py',
            ROOT / 'scripts/convert_robotwin_legacy.py', ROOT / 'scripts/train_robotwin_wam.py',
            ROOT / 'configs/fixed_bimanual/robotwin_cabinet_500_wam.yaml',
            ROOT / 'data/benchmarks/robotwin2/put_object_cabinet_500_v1/SOURCE.json',
            ROOT / 'data/benchmarks/robotwin2/put_object_cabinet_500_v1/split_manifest.json',
            ROOT / 'outputs/validation/robotwin_continuous_task_20261004/model_initialization.json',
            ROOT / 'outputs/runs/put_object_cabinet/smoke_20261005/status.json']},
        'sha256': {Path(path).name: digest(path) for path in artifacts},
    }, indent=2) + '\n')
    print(f'PASS: {len(NODES)} modules, {len(EDGES)} checked arrows; SVG and 3072×1830 PNG in {OUT}')


if __name__ == '__main__':
    main()
