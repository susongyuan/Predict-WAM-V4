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
# Functional views of one policy-server runtime, not separate nav/manip models.
# Reuse the former residual route for the real shared-encoder arrival features.
EDGES = [
    ('world', 'vlm', 'head_rgb_for_grounding', [(350, 285), (420, 285)], 'data'),
    ('vlm', 'orch', 'candidate_skill_programs', [(700, 285), (780, 285)], 'data'),
    ('orch', 'nav', 'docking_target_and_budget', [(1110, 300), (1180, 300)], 'data'),
    ('orch', 'controller', 'execute_hold_and_queue_reset', [(1040, 200), (1040, 180), (1680, 180), (1680, 200)], 'action'),
    ('controller', 'world', 'bounded_23d_physical_action', [(1790, 200), (1790, 100), (200, 100), (200, 200)], 'action'),
    ('world', 'controller', 'rgbd_wrist_rgb_and_61d_proprio', [(250, 200), (250, 140), (1750, 140), (1750, 200)], 'data'),
    ('nav', 'controller', 'docking_base_and_posture_command', [(1520, 280), (1600, 280)], 'action'),
    ('controller', 'nav', 'head_rgbd_campose_and_base_velocity', [(1600, 380), (1520, 380)], 'data'),
    ('nav', 'orch', 'docked_or_search_failed_status', [(1180, 400), (1110, 400)], 'feedback'),
    ('orch', 'wam', 'manipulation_contract_condition', [(1110, 420), (1150, 420), (1150, 750), (1180, 750)], 'data'),
    ('controller', 'wam', 'three_view_rgb_and_masked_61d', [(1600, 700), (1520, 700)], 'data'),
    ('wam', 'controller', 'manipulation_32x23d_chunks', [(1520, 780), (1600, 780)], 'action'),
    ('predicates', 'orch', 'fail_closed_completion_status', [(780, 1000), (750, 1000), (750, 420), (780, 420)], 'feedback'),
    ('bank', 'orch', 'condition_arms_and_reviewed_outcomes', [(945, 650), (945, 450)], 'memory'),
    ('human', 'orch', 'ask_and_bound_reply', [(700, 750), (720, 750), (720, 400), (780, 400)], 'interaction'),
    ('orch', 'vlm', 'replan_after_repeated_failure', [(780, 365), (700, 365)], 'feedback'),
    ('controller', 'predicates', 'finger_and_eef_proprio', [(1790, 850), (1790, 1000), (1520, 1000)], 'data'),
]
# Keep the nine functional modules and seventeen existing routes. The Task90
# onboard producer is not wired/admitted; legacy proprio transport alone is not
# completion evidence. Dotted routes denote pending integration, not capability.
PENDING = {'replan_after_repeated_failure', 'finger_and_eef_proprio'}
BIDIRECTIONAL = {'condition_arms_and_reviewed_outcomes', 'ask_and_bound_reply'}


def on_boundary(point, box):
    px, py = point
    x, y, w, h = box
    return ((px in (x, x+w) and y <= py <= y+h)
            or (py in (y, y+h) and x <= px <= x+w))


def validate():
    """Fail on miswired endpoints, box penetration, or crossing arrow routes."""
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
<title>WAM + VLM: task-agnostic mobile-manipulation runtime</title>
<desc>Task90 snapshot, 2026-10-02. Composting_waste keeps the original two inside goals: a rigid half-banana and half-pomegranate in one trash can; release and stability require separate evidence. At fixed WAM checkpoint 25k, seeds 0/1/2 on the same dev instance311 with base prepositioning and assisted grasp qualified the FIRST BIN grasp in 2/3 runs, with maximum logged AABB-bottom lifts of 20.7/24.4cm. All three had Q=0, task_success=false, evaluator rc=1 and incomplete evaluation. This is not full-task success, generalization, autonomous navigation or friction-only grasp. Training resumed from 25k toward 30k; 30k is not complete at this snapshot. The 50k plan checks every 5k; 200 demonstrations (180 train/20 validation), RGB, train-only VTT and DINO cache are ready. DINO is frozen; the action Transformer and temporal/gating components continue training. The existing world-to-control-to-navigation transport can carry an explicit opt-in nav-only truth side channel for development navigation/prepositioning, isolated from the sensor allowlist. WAM actions, grounding, condition/history encoding and online manipulation postconditions receive no GT fields. Evaluator poses, BDDL, Q and assisted-lift audit cannot confirm manipulation completion, switch manipulation stages or promote memory. Navigation handoff in the assisted diagnostic is not observed manipulation success. The intended online evidence source is onboard RGB-D plus proprioception; its producer is being implemented independently on CPU and is not wired or admitted. Unknown does not confirm success or advance a contract; bounded execution/observation may continue until its budget, then safe stop/recovery, with no unverified memory promotion. VLM, asking and memory benefits remain unverified. R1Pro uses assisted simulation grasping on an Isaac Sim6.1 port; no real-world or deformable-object result is claimed.</desc>
<defs>
<linearGradient id="physical" x2="0.9" y2="1"><stop stop-color="#eef8fa"/><stop offset="1" stop-color="#dae9f0"/></linearGradient>
<linearGradient id="latent" x2="1" y2="1"><stop stop-color="#f2effc"/><stop offset="1" stop-color="#e0e5fa"/></linearGradient>
''']
    colors = {'data': '#287fb5', 'action': '#b77928', 'feedback': '#008c70',
              'interaction': '#8565aa', 'memory': '#008c70'}
    for kind, color in colors.items():
        chunks.append(f'<marker id="arrow-{kind}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M 0 1 L 9 5 L 0 9 z" fill="{color}"/></marker>')
    chunks.append('</defs><style>text{font-family:DejaVu Sans,sans-serif;fill:#263544}.small{font-size:15px}.body{font-size:16px}.title{font-size:23px;font-weight:600}.edge-label{font-size:15px}</style><rect width="2048" height="1220" fill="white"/>')

    def text(x, y, value, cls='body', extra=''):
        chunks.append(f'<text x="{x}" y="{y}" class="{cls}" {extra}>{escape(value)}</text>')

    text(50, 48, 'WAM + VLM  |  General mobile-manipulation runtime', extra='style="font-size:32px;font-weight:600"')
    text(50, 80, 'Task90: two fruit halves into one bin • existing navigation • learned manipulation • independent outcome checks', 'body')
    text(1980, 48, 'TASK 90 · ORDINARY GRASP / PLACE', 'small', 'text-anchor="end"')
    for source, target, name, points, kind in EDGES:
        path = 'M ' + ' L '.join(f'{x},{y}' for x, y in points)
        dash = ' stroke-dasharray="4 7" opacity="0.5"' if name in PENDING else ''
        start = f' marker-start="url(#arrow-{kind})"' if name in BIDIRECTIONAL else ''
        chunks.append(f'<g id="edge-{name}" data-source="{source}" data-target="{target}"><path d="{path}" fill="none" stroke="{colors[kind]}" stroke-width="2.6" stroke-linejoin="round" marker-end="url(#arrow-{kind})"{start}{dash}/></g>')

    def card(node, title, badge, lines, fill, accent):
        x, y, w, h = NODES[node]
        chunks.append(f'<g id="{node}"><rect x="{x}" y="{y+4}" width="{w}" height="{h}" rx="16" fill="#263544" opacity="0.045"/><rect x="{x}" y="{y}" width="{w}" height="{h}" rx="16" fill="{fill}" stroke="{accent}" stroke-width="1.3"/>')
        text(x+21, y+30, badge, 'small', 'style="font-weight:600;font-size:12px;letter-spacing:1px"')
        text(x+21, y+67, title, 'title', 'style="font-size:22px"')
        for i, line in enumerate(lines):
            text(x+21, y+102+i*27, line, 'body')
        chunks.append('</g>')

    card('world', 'BEHAVIOR · Task90', 'RS_INT · ISAAC SIM 6.1 PORT',
         ['Two rigid fruit halves → bin', 'R1Pro: assisted grasping', 'No GT fields to WAM actions'], 'url(#physical)', '#548795')
    # A code-native physical anchor: room floor, mobile base, torso, two arms.
    chunks.append('''<g id="robot-schematic" stroke="#587681" stroke-width="3" stroke-linejoin="round">
<path d="M95 526 L194 472 L337 519 L242 570 Z" fill="#cfdfe6" stroke-width="1.2"/>
<path d="M179 516 L216 494 L255 514 L217 539 Z" fill="#829da8"/>
<path d="M179 516 L179 530 L217 553 L217 539 Z" fill="#66818e"/>
<path d="M217 539 L255 514 L255 530 L217 553 Z" fill="#91acb8"/>
<path d="M209 496 L209 439 L228 433 L228 495 Z" fill="#eef5f7"/>
<path d="M204 439 L238 439 L244 421 L214 412 Z" fill="#b9d1da"/>
<circle cx="222" cy="426" r="4" fill="#48a7ae" stroke="none"/>
<path d="M209 452 L178 462 L155 493 M228 451 L261 458 L285 481" fill="none" stroke-width="10"/>
<path d="M150 489 L145 502 M160 495 L157 507 M280 478 L286 491 M290 477 L298 488" fill="none" stroke-width="4"/>
</g>''')
    chunks.append('<g id="evaluation-audit" data-node="world">')
    text(71, 700, 'TRUTH BOUNDARIES', 'small', 'style="font-weight:600"')
    text(71, 730, 'Opt-in: dev nav / preposition', 'small')
    text(71, 758, 'Audit only: poses / BDDL / Q', 'small')
    text(71, 786, 'Not manip. completion / memory', 'small')
    chunks.append('</g>')
    card('vlm', 'Goal grounding', 'VLM · BENEFIT UNVERIFIED',
         ['Goal + RGB → skill programs', 'Fixed plan for these checks', 'Safety cannot be bypassed'], '#f3eff9', '#8565aa')
    card('orch', 'Skill contracts', 'OBSERVATION-GATED ADVANCE',
         ['No confirmation → no advance', 'Unknown: bounded observation', 'Budget ends → safe stop', 'No GT manipulation transitions'], '#e9f6f0', '#388571')
    card('nav', 'Navigation / docking', 'REUSED · ASSISTANCE DISCLOSED',
         ['RGB-D map + body envelope', 'Opt-in nav-only truth channel', 'Dev preposition ≠ nav success', 'GT isolated from manip. evidence'], '#eaf6f7', '#438b98')
    card('wam', 'WAM V3 · Task90', '25K → 30K TRAINING STARTED',
         ['Head + 2 wrists · K=1 · exec 8', 'DINO cached + frozen · history', 'Action Transformer still trains'], 'url(#latent)', '#727aaf')
    # Existing status panels and all node/route geometry are retained.
    chunks.append('<g id="pending-navigation" data-node="nav"><rect x="1194" y="400" width="312" height="38" rx="8" fill="none" stroke="#438b98"/>')
    text(1204, 424, 'DEV ONLY · first-bin assistance', 'small')
    chunks.append('</g><g id="pending-manip-pilot" data-node="wam"><rect x="1194" y="817" width="312" height="27" rx="8" fill="none" stroke="#727aaf" stroke-dasharray="4 7"/>')
    text(1204, 836, '50k plan · checks every 5k', 'small')
    chunks.append('</g>')
    card('controller', 'Policy / control adapter', 'NAV TRUTH / SENSOR SPLIT',
         ['RGB-D + wrist RGB + 61D input', 'Nav truth split before allowlist', 'WAM / postconditions: sensors only', 'One action owner per step'], '#fbf1e4', '#b77928')
    chunks.append('''<g id="control-ownership" data-node="controller">
<rect x="1622" y="500" width="336" height="112" rx="12" fill="#fffaf3" stroke="#d7a45a" stroke-width="1.2"/>
<text x="1642" y="530" class="small" style="font-weight:600">TRAVEL / DEV ASSISTANCE</text>
<text x="1642" y="560" class="body">Base + posture; nav-only truth</text>
<text x="1642" y="588" class="body">Explicit opt-in; depth shield</text>
<rect x="1622" y="640" width="336" height="132" rx="12" fill="#f7f4fd" stroke="#8576b5" stroke-width="1.2"/>
<text x="1642" y="670" class="small" style="font-weight:600">WAM SENSOR ALLOWLIST</text>
<text x="1642" y="700" class="body">3 RGB views + masked 61D</text>
<text x="1642" y="728" class="body">No GT fields; limits may override</text>
<text x="1642" y="756" class="small">32 × 23D chunks; execute 8 steps</text>
</g>''')
    card('human', 'Intent clarification', 'ASKING · BENEFIT UNVERIFIED',
         ['CPU Qwen3-1.7B / human CLI', 'Action-changing ambiguity', 'Episode / question IDs checked', 'No reply → HOLD'], '#f4eef9', '#8565aa')
    card('bank', 'Condition memory', 'MEMORY · BENEFIT UNVERIFIED',
         ['Task / object / nav-next / version', 'Train targets / reviewed VTT', 'No unverified success promotion', 'No evaluator GT as evidence'], '#edf6f0', '#388571')
    card('predicates', 'Evidence gate + recovery', 'ONBOARD PRODUCER · CPU WORK / NOT WIRED OR ADMITTED',
         ['RGB-D + proprio only; observe target / hand / release / stability', 'Unknown ≠ success: no advance; budget → safe stop / recovery', 'No GT manipulation completion, stage switch or memory promotion'], '#edf5f7', '#548795')

    for x, y, label, anchor in [
        (385, 270, 'RGB', 'middle'), (740, 270, 'programs', 'middle'),
        (1145, 285, 'target', 'middle'), (1360, 170, 'EXECUTE / HOLD', 'middle'),
        (1360, 94, '30 Hz · bounded 23D action', 'middle'),
        (1000, 134, 'RGB-D + wrists + 61D | opt-in nav-only truth → isolated navigation', 'middle'),
        (1560, 265, 'dock cmd', 'middle'), (1560, 365, 'nav truth', 'middle'),
        (1135, 590, 'contract condition', 'end'),
        (1560, 685, 'no GT', 'middle'), (1560, 765, '32×23D', 'middle'),
        (1145, 385, 'status', 'middle'),
        (765, 590, 'verdict / recovery', 'start'),
        (960, 530, 'arms ↕ reviewed VTT', 'start'),
        (705, 535, 'ASK ↕ answer', 'end'),
        (740, 350, 'replan', 'middle'),
        (1660, 980, 'onboard only · pending producer', 'middle')]:
        text(x, y, label, 'edge-label', f'text-anchor="{anchor}"')
    text(50, 1135, 'Routes show interfaces, not proven capability; dotted = pending. GT: opt-in dev navigation / preposition + evaluator audit only; never WAM action input or manipulation verdict.', 'small')
    text(50, 1160, '25k fixed WAM · same dev311 · seeds 0/1/2 · preposition + assisted grasp: FIRST BIN grasp 2/3; max logged lifts 20.7 / 24.4 cm. All Q=0, rc=1; evaluations incomplete.', 'small')
    text(50, 1185, 'Original goals unchanged: both fruit halves inside bin; release / stability need separate evidence. First grasp ≠ full-task success, generalization or autonomous navigation.', 'small')
    text(50, 1205, 'DINO cache ready + frozen; action Transformer training 25k → 30k, not yet 30k complete. Onboard producer not admitted; VLM / asking / memory benefits unverified.', 'small')
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
    assert not document.find(f"{ns}g[@id='pending-navigation']/{ns}rect").get('stroke-dasharray')
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
        'head_rgb_for_grounding': 'WamPolicy._ground/_act_orchestrated: head RGB only; deterministic benchmark gates may use a fixed typed program so grounding quality is not confused with control quality.',
        'candidate_skill_programs': 'harness/grounding.py:ground/compile_programs validates typed skill(objects) candidates against the skill catalog before Orchestrator.reset.',
        'docking_target_and_budget': 'Orchestrator navigation_completion="docking": move-to key and budget (min_budget_steps) -> WamPolicy._act_orchestrated -> DockingController.start(label, phase).',
        'execute_hold_and_queue_reset': 'Orchestrator.tick -> WamPolicy._act_orchestrated: Directive EXECUTE/HOLD, queue invalidation, hold_baseline.',
        'bounded_23d_physical_action': 'WamPolicy._bounded_chunk/_apply_base_control/serve -> official websocket evaluator and R1Pro controllers; action shape 23.',
        'rgbd_wrist_rgb_and_61d_proprio': 'World -> control transport: RGB-D, wrist RGB, 61D proprio/camera poses. ONLY the explicit privileged_base_navigation wrapper opt-in adds the named nav-only truth packet; WamPolicy._split_navigation_observation separates it before the sensor allowlist. PrepositionedBaseRGBDWrapper uses truth for diagnostic base-only reset placement, not autonomous navigation or manipulation success.',
        'docking_base_and_posture_command': 'harness/docking_controller.py DockingController.step -> base velocity + trunk/arm posture target, dispatched by WamPolicy._act_orchestrated during move to.',
        'head_rgbd_campose_and_base_velocity': 'Control -> navigation: head RGB-D, camera pose, base velocity; explicit opt-in nav-only truth -> PrivbaseNavigator.prepare. Its separate odometry/ObjectMap serves first move to(trash_can) only; no truth map is shared with WAM or observed manipulation postconditions. Development navigation/prepositioning assistance is not an autonomous result.',
        'docked_or_search_failed_status': 'DockingController.last_status (docked / search failed / budget) -> Orchestrator._completion with navigation_completion="docking".',
        'manipulation_contract_condition': 'ConditionBank.select(contract): frozen train-only task/skill condition vectors -> WamPolicy._sample for manipulation keys only.',
        'three_view_rgb_and_masked_61d': 'WamPolicy._policy_observation allowlists this robot RGB/depth/proprio/camera pose only; WAM _preprocess/_view_pixels uses head + two wrist RGB and masked 61D (positions_no_gripper). Frozen DINO supplies temporal features. No GT object pose, goal/Q, assisted-hand identity or phase labels enter WAM actions, grounding or history/condition encoding.',
        'manipulation_32x23d_chunks': 'WamPolicy._sample/_bounded_chunk proposes 32x23D chunks. Controller bounds and harness/docking_controller.py geometric safety checks may hold or replace unsafe base motion.',
        'fail_closed_completion_status': 'Orchestrator._completion accepts only verified sensor-derived postconditions for manipulation advance. Unknown returns no completion before budget; candidate satisfaction holds; unverified budget expiry fails/stops safely, not success. No GT manipulation phase switch or unverified success-memory promotion. Task090Postconditions.admission_status remains false; independent onboard producer CPU work is not integrated/admitted.',
        'condition_arms_and_reviewed_outcomes': 'ConditionBank.select/update/add_memory_arm and harness/memory_vtt.py: scoped, independently reviewed evidence; frozen runs do not update; support is not policy success.',
        'ask_and_bound_reply': 'Orchestrator.tick/answer <-> policy_server.control_message: bounded ASK, episode/question IDs, source-labeled human/sim replies.',
        'replan_after_repeated_failure': 'PENDING: VLM recovery replanning is not wired; Task 1 uses WamPolicy._skip_failed_object as the orchestrator replanner (rule-based).',
        'finger_and_eef_proprio': 'PENDING producer integration: only onboard RGB-D + proprio may supply target/hand-relative motion, support/container relation and release/stability evidence. Legacy proprio transport is insufficient. Task090Postconditions ADMISSION_BLOCKERS remain; CPU implementation work is not runtime admission. Evaluator poses, BDDL/Q, assisted-grasp audit and nav truth must never enter this route.',
    }
    assert evidence.keys() == {e[2] for e in EDGES}
    topology_path.write_text(json.dumps({
        'nodes': NODES, 'edges': [dict(source=s, target=t, meaning=n, points=p, kind=k,
                                     implemented=n not in PENDING, bidirectional=n in BIDIRECTIONAL,
                                     semantic_evidence=evidence[n])
                                for s, t, n, p, k in EDGES],
        'validation': 'ports, orthogonal routes, canvas/text bounds, no node/label penetration or crossings, XML, no raster',
        'scope': 'Existing layout retained; implementation flags describe interfaces, not task capability',
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
        'scope': '2026-10-02 Task90 snapshot: fixed 25k prepositioned assisted first-bin grasp 2/3 on same dev311; all Q0/rc1, no full task success; training 25k toward 30k',
        'nodes': len(NODES), 'edges': len(EDGES),
        'checks': {
            'source_destination_ports': 'pass', 'orthogonal_routes': 'pass',
            'node_penetration': 'none', 'arrow_crossings_or_overlaps': 'none',
            'svg_xml': 'pass', 'raster_layers_in_publication_svg': 0,
            'canvas_and_text_bounds': 'pass', 'text_overlaps_or_route_crossings': 'none',
            'rendered_text_labels_checked': len(boxes),
            'implemented_pending_arrow_styles': 'pass', 'png_size': [3072, 1830],
            'png_rendered_from_current_svg': True,
            'manual_image_inspection': {'status': 'pending', 'note': 'New V3 labels require manual inspection after this generated render; automated geometry checks below are separate.'},
        },
        'routes': {n: dict(source=s, target=t, source_port=p[0], target_port=p[-1], result='pass')
                   for s, t, n, p, _ in EDGES},
        'semantic_edge_evidence': evidence,
        'pending_interfaces': sorted(PENDING),
        'pending_pilot': ['Admit independent onboard RGB-D target/hand/relation producer; CPU work is not connected or admitted.',
                          'Verify bounded unknown handling, safe recovery and no GT manipulation phase switching.',
                          'Complete full-system evaluation of both original inside goals, separately verify release/stability, and resolve evaluator rc1.',
                          'Evaluate VLM, asking and memory benefits separately; none is established here.'],
        'audited_evidence': {
            'task90_25k_triplet': {
                'summary': 'outputs/validation/task090_25k_triplet_20261002/state.json',
                'independent_audit': 'outputs/validation/task090_25k_triplet_20261002/independent_lift_check.json',
                'checkpoint_step': 25000, 'instance_id': 311, 'seeds': [0, 1, 2],
                'first_grasp_target': 'ashcan.n.01_1', 'qualified_first_bin_grasps': 2, 'runs': 3,
                'max_logged_aabb_bottom_lift_cm_successful_seeds': [20.7, 24.4],
                'qualification': 'active assisted grasp + reset-relative AABB bottom lift >=3cm for >=15 consecutive control steps',
                'q_scores': [0, 0, 0], 'task_success': [False, False, False],
                'evaluator_returncodes': [1, 1, 1], 'evaluation_complete': [False, False, False],
                'interpretation': 'Prepositioned-base assisted diagnostic only; not full-task success, generalization, friction-only grasp or autonomous navigation'},
            'task90_training': {
                'launch': 'outputs/validation/task090_takeover_20261001/to30k_s030000_launch.json',
                'manifest': 'outputs/training/task090_v3_50k_20260930_s030000/manifest.json',
                'episodes': 200, 'train': 180, 'validation': 20, 'dino_cache': 'ready', 'dino_encoder': 'frozen',
                'action_transformer': 'training', 'segment': '25000 -> 30000 started; not completed at snapshot',
                'plan_steps': 50000, 'check_every_steps': 5000},
            'onboard_evidence': {'status': 'independent CPU implementation work, not wired or admitted',
                                 'gate_source': 'harness/task090_predicates.py',
                                 'admission_blockers': 'object/hand tracks, support/container/sink relations, train/geometry calibration'},
        },
        'runtime_limits': [
            'Truth is opt-in development base navigation/prepositioning or evaluator-only audit; not WAM action fields, manipulation verdict, manipulation stage switching or memory evidence.',
            'Navigation uses a separate truth map in the opt-in diagnostic. Sensor-only deployment remains the intended runtime, not a proven result.',
            'Task90 keeps the original two inside goals and rigid fruit halves. First grasp here targets the bin; it is not fruit placement or deformable manipulation.',
            'Assisted-grasp lift is a simulator-constraint diagnostic, not evidence of real-world/friction-only grasp or absence of other support.',
            '25k same-instance three-seed checks all have Q0, task_success false and rc1; evaluations are incomplete.',
            'Unknown does not always mean immediate HOLD: no unverified advance; bounded execution/observation until budget then safe stop/recovery. Candidate satisfaction may HOLD earlier.',
            'Onboard RGB-D/proprio producer, target-bound relations and safe continuation require independent admission; force/tactile/current evidence is not established.',
            'VLM / asking / memory benefit, full-task success and generalization remain unverified.',
        ],
        'evidence_sha256': {str(path.relative_to(ROOT)): digest(path) for path in [
            ROOT / 'scripts/draw_architecture.py',
            ROOT / 'harness/policy_server.py', ROOT / 'harness/orchestrator.py', ROOT / 'harness/docking_controller.py',
            ROOT / 'harness/task090_predicates.py', ROOT / 'harness/task090_perception.py',
            ROOT / 'harness/privileged_base_navigation.py', ROOT / 'harness/prepositioned_base_rgbd_wrapper.py',
            ROOT / 'harness/wam_v3.py', ROOT / 'scripts/train_wam_task_v3.py',
            ROOT / 'outputs/validation/task090_25k_triplet_20261002/state.json',
            ROOT / 'outputs/validation/task090_25k_triplet_20261002/independent_lift_check.json',
            ROOT / 'outputs/validation/task090_takeover_20261001/to30k_s030000_launch.json',
            ROOT / 'outputs/training/task090_v3_50k_20260930_s030000/manifest.json',
            ROOT / 'outputs/validation/task090_preparation_20260930/dino_v1/completion.json',
            ROOT / 'data/skill_vtt_task090/meta.json']},
        'sha256': {Path(path).name: digest(path) for path in artifacts},
    }, indent=2) + '\n')
    print(f'PASS: {len(NODES)} modules, {len(EDGES)} checked arrows; SVG and 3072×1830 PNG in {OUT}')


if __name__ == '__main__':
    main()
