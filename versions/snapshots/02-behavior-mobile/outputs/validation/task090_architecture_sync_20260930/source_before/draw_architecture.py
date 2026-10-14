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
# RGB-D perception, obstacle points and the dynamic body envelope are wired.  The
# remaining dotted route is semantic VLM recovery replanning; bounded rule-based
# retry / rollback / skip is already implemented in the orchestrator.
PENDING = {'replan_after_repeated_failure'}
BIDIRECTIONAL = {'condition_arms_and_reviewed_outcomes', 'ask_and_bound_reply'}
REPLAY_RUN = 'outputs/validation/task1_demo_replay_20260926/run_ep389_inst290_rgbd'
ADMISSION = 'outputs/validation/task1_manip_admission_20260926'
COPYCAT = 'outputs/validation/task1_gripper_copycat_20260926/last/summary.json'


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
<desc>Code-aligned snapshot, 2026-09-30. WAM V3 completed 30k optimizer steps and six development evaluations; none completed Task 0. The separate 25k repeated-test plan has one failed completed episode out of ten; nine have not run. A contact candidate is not an observed outcome: the default candidate policy now clears queued actions and holds without advancing, retrying an irreversible action, or writing success memory. Place release-height state is scoped to the current contract and episode. These changes passed independent CPU tests, not a new physical episode. An admitted independent outcome verifier and safe post-verification resume remain missing. Current simulator grasping is contact-triggered assisted grasp, not pure friction or real-world grasp validation. The learned WAM remains the nominal short-horizon manipulation policy; task-independent RGB-D and proprioception safety own travel, docking, collision checks and unsafe-command suppression. V3 vision updates on every control frame; exec8 is the action-chunk refresh interval, not the visual observation interval. Three validation anchors were examined offline for exec4; no exec4 closed-loop improvement is established. Task 0 BDDL checks toggled_on only; returning the object requires separate evidence for the user's complete four-stage task. Object poses, BDDL and Q-score are evaluator-only audit and never policy inputs. VLM recovery, an admitted outcome verifier, and force/tactile/current evidence remain pending. No task success or generalization is claimed.</desc>
<defs>
<linearGradient id="physical" x2="0.9" y2="1"><stop stop-color="#eef8fa"/><stop offset="1" stop-color="#dae9f0"/></linearGradient>
<linearGradient id="latent" x2="1" y2="1"><stop stop-color="#f2effc"/><stop offset="1" stop-color="#e0e5fa"/></linearGradient>
''']
    colors = {'data': '#64778b', 'action': '#b77928', 'feedback': '#217e73',
              'interaction': '#8565aa', 'memory': '#217e73'}
    for kind, color in colors.items():
        chunks.append(f'<marker id="arrow-{kind}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M 0 1 L 9 5 L 0 9 z" fill="{color}"/></marker>')
    chunks.append('</defs><style>text{font-family:DejaVu Sans,sans-serif;fill:#263544}.small{font-size:15px}.body{font-size:16px}.title{font-size:23px;font-weight:600}.edge-label{font-size:15px}</style><rect width="2048" height="1220" fill="white"/>')

    def text(x, y, value, cls='body', extra=''):
        chunks.append(f'<text x="{x}" y="{y}" class="{cls}" {extra}>{escape(value)}</text>')

    text(50, 48, 'WAM + VLM  |  General mobile-manipulation runtime', extra='style="font-size:32px;font-weight:600"')
    text(50, 80, 'Learned nominal policy • geometric safety shield • evidence-based completion and recovery', 'body')
    text(1980, 48, 'TASK 0: FIRST-SUCCESS GATE · TASK 1: STRESS TEST', 'small', 'text-anchor="end"')
    for source, target, name, points, kind in EDGES:
        path = 'M ' + ' L '.join(f'{x},{y}' for x, y in points)
        dash = ' stroke-dasharray="4 7" opacity="0.5"' if name in PENDING else ''
        start = f' marker-start="url(#arrow-{kind})"' if name in BIDIRECTIONAL else ''
        chunks.append(f'<g id="edge-{name}" data-source="{source}" data-target="{target}"><path d="{path}" fill="none" stroke="{colors[kind]}" stroke-width="2.6" stroke-linejoin="round" marker-end="url(#arrow-{kind})"{start}{dash}/></g>')

    def card(node, title, badge, lines, fill, accent):
        x, y, w, h = NODES[node]
        chunks.append(f'<g id="{node}"><rect x="{x}" y="{y+4}" width="{w}" height="{h}" rx="16" fill="#263544" opacity="0.045"/><rect x="{x}" y="{y}" width="{w}" height="{h}" rx="16" fill="{fill}" stroke="{accent}" stroke-width="1.3"/>')
        text(x+21, y+30, badge, 'small', f'style="fill:{accent};font-weight:600;font-size:12px;letter-spacing:1px"')
        text(x+21, y+67, title, 'title', 'style="font-size:22px"')
        for i, line in enumerate(lines):
            text(x+21, y+102+i*27, line, 'body')
        chunks.append('</g>')

    card('world', 'BEHAVIOR household', 'OMNIGIBSON · ISAAC SIM 6.1 PORT',
         ['Head RGB-D, wrist RGB, 61D', 'R1Pro: assisted grasping', 'No object state to policy'], 'url(#physical)', '#548795')
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
    text(71, 700, 'EVALUATOR ONLY', 'small', 'style="fill:#548795;font-weight:600"')
    text(71, 730, 'Object poses / BDDL / Q-score', 'small')
    text(71, 758, 'Post-hoc audit; never policy input', 'small')
    chunks.append('</g>')
    card('vlm', 'Goal grounding', 'OPTIONAL VLM · FIXED PLAN',
         ['Goal + RGB → skill programs', 'Catalog + supplied-object list', 'Safety cannot be bypassed'], '#f3eff9', '#8565aa')
    card('orch', 'Skill contracts', 'CANDIDATE HOLD · CPU-CHECKED',
         ['Typed skill(objects) contracts', 'Candidate → HOLD; no advance', 'Queue cleared; no blind retry', 'Unknown ≠ success or memory'], '#e9f6f0', '#388571')
    card('nav', 'RGB-D safe docking', 'WIRED · GEOMETRIC SAFETY LAYER',
         ['RGB-D → base-frame map', 'Dynamic body / arm envelope', 'Dock at train-calibrated pose', 'Unsafe motion filtered or stopped'], '#eaf6f7', '#438b98')
    card('wam', 'WAM V3 · 30k weights', 'TEMPORAL MEMORY + VIEW GATE',
         ['Head + 2 wrists · K=1 · exec 8', 'Proprio + frozen DINO history', 'Phase-gated 32 × 23D proposal'], 'url(#latent)', '#727aaf')
    # Status annotations stay inside existing cards; dotted means not yet implemented.
    chunks.append('<g id="pending-navigation" data-node="nav"><rect x="1194" y="400" width="312" height="38" rx="8" fill="none" stroke="#438b98"/>')
    text(1204, 424, 'LIMIT · task-specific colour detector', 'small')
    chunks.append('</g><g id="pending-manip-pilot" data-node="wam"><rect x="1194" y="817" width="312" height="27" rx="8" fill="none" stroke="#727aaf" stroke-dasharray="4 7"/>')
    text(1204, 836, '5k checkpoints · physical dev gate', 'small')
    chunks.append('</g>')
    card('controller', 'Policy / control adapter', '30 HZ SIM ADAPTER · I/O PASSED',
         ['RGB-D + wrist RGB + 61D input', 'Bounded commands → safety filter', 'Base velocity; absolute joint targets', 'One action owner per step'], '#fbf1e4', '#b77928')
    chunks.append('''<g id="control-ownership" data-node="controller">
<rect x="1622" y="500" width="336" height="112" rx="12" fill="#fffaf3" stroke="#d7a45a" stroke-width="1.2"/>
<text x="1642" y="530" class="small" style="fill:#b77928;font-weight:600">TRAVEL</text>
<text x="1642" y="560" class="body">Safe docking: base + posture</text>
<text x="1642" y="588" class="body">Depth / envelope shield active</text>
<rect x="1622" y="640" width="336" height="132" rx="12" fill="#f7f4fd" stroke="#8576b5" stroke-width="1.2"/>
<text x="1642" y="670" class="small" style="fill:#727aaf;font-weight:600">AFTER DOCKING</text>
<text x="1642" y="700" class="body">WAM proposes bounded actions</text>
<text x="1642" y="728" class="body">Shield / limits may override</text>
<text x="1642" y="756" class="small">IK primitive = diagnostic fallback only</text>
</g>''')
    card('human', 'Intent clarification', 'HUMAN / SIM · SOURCE LABELED',
         ['CPU Qwen3-1.7B / human CLI', 'Action-changing ambiguity', 'Episode / question IDs checked', 'No reply → HOLD'], '#f4eef9', '#8565aa')
    card('bank', 'Condition memory', 'INDEPENDENT REVIEW REQUIRED',
         ['Task / object / nav-next / version', 'Train targets / reviewed VTT', 'Unverified: no promotion', 'Support ≠ policy execution wins'], '#edf6f0', '#388571')
    card('predicates', 'Evidence gate + recovery', 'PROPRIO PROXIES · OUTCOME VERIFIER PENDING',
         ['Release state is contract-scoped; target placement not yet verified', 'Candidate → HOLD; independent outcome + safe resume pending', 'Force / tactile / current are pending; no new physical success'], '#edf5f7', '#548795')

    for x, y, label, anchor in [
        (385, 270, 'RGB', 'middle'), (740, 270, 'programs', 'middle'),
        (1145, 285, 'target', 'middle'), (1360, 170, 'EXECUTE / HOLD', 'middle'),
        (1360, 94, '30 Hz · bounded 23D action', 'middle'),
        (1000, 134, 'head RGB-D + wrist RGB + 61D proprio', 'middle'),
        (1560, 265, 'dock cmd', 'middle'), (1560, 365, 'RGB-D, v', 'middle'),
        (1135, 590, 'contract condition', 'end'),
        (1560, 685, '3 views', 'middle'), (1560, 765, '32×23D', 'middle'),
        (1145, 385, 'status', 'middle'),
        (765, 590, 'verdict / recovery', 'start'),
        (960, 530, 'arms ↕ reviewed VTT', 'start'),
        (705, 535, 'ASK ↕ answer', 'end'),
        (740, 350, 'replan', 'middle'),
        (1660, 980, 'vision + proprio evidence', 'middle')]:
        text(x, y, label, 'edge-label', f'text-anchor="{anchor}"')
    text(50, 1135, 'Solid = wired interfaces (not proven capability); dotted = pending. Evaluator object poses / BDDL never reach the policy.', 'small')
    text(50, 1160, 'Task 0: six 5k-spaced development evaluations failed. Old 25k repeat plan: 1/10 completed, failed; nine not run. No new GPU run.', 'small')
    text(50, 1185, 'CPU fixes: candidate clears queue and holds; release history isolated. BDDL on-state alone does not prove the full pick / press / return task.', 'small')
    text(50, 1205, 'Boundary: contact-assisted simulation grasping is not pure friction or real-world validation. Independent outcome verification / resume remain pending.', 'small')
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
    (OUT / 'topology.json').write_text(json.dumps({
        'nodes': NODES, 'edges': [dict(source=s, target=t, meaning=n, points=p, kind=k, implemented=n not in PENDING,
                                        bidirectional=n in BIDIRECTIONAL)
                                  for s, t, n, p, k in EDGES],
        'validation': 'ports, orthogonal routes, canvas/text bounds, no node/label penetration or crossings, XML, no raster'
    }, indent=2) + '\n')
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
        'rgbd_wrist_rgb_and_61d_proprio': 'behavior_closed_loop.py --env-wrapper RGBDFullResWrapper: head RGB + depth_linear, wrist RGB, 61D proprio incl. cam_rel_poses; no privileged object state.',
        'docking_base_and_posture_command': 'harness/docking_controller.py DockingController.step -> base velocity + trunk/arm posture target, dispatched by WamPolicy._act_orchestrated during move to.',
        'head_rgbd_campose_and_base_velocity': 'harness/task1_perception_adapter.TaskPerception.detect/obstacles -> harness/object_perception.py; head depth, camera pose, Odometry.update and ObjectMap.update are wired. Current semantic target detector is task-specific colour segmentation.',
        'docked_or_search_failed_status': 'DockingController.last_status (docked / search failed / budget) -> Orchestrator._completion with navigation_completion="docking".',
        'manipulation_contract_condition': 'ConditionBank.select(contract): frozen train-only task/skill condition vectors -> WamPolicy._sample for manipulation keys only.',
        'three_view_rgb_and_masked_61d': 'WamPolicy._preprocess/_view_pixels/_model_feats (head + left/right wrist at 224); BehaviorVLAWrapper.normalize_state applies the checkpoint proprio_mask (v2: positions_no_gripper).',
        'manipulation_32x23d_chunks': 'WamPolicy._sample/_bounded_chunk proposes 32x23D chunks. Controller bounds and harness/docking_controller.py geometric safety checks may hold or replace unsafe base motion.',
        'fail_closed_completion_status': 'Task0Postconditions scopes release state to episode/contract. Orchestrator.tick handles a satisfied contact candidate before completion: default HOLD, success=null, irreversible retry lock. WamPolicy._act_orchestrated clears queued actions before returning hold. Independent outcome verification and safe resume after this HOLD are not admitted.',
        'condition_arms_and_reviewed_outcomes': 'ConditionBank.select/update/add_memory_arm and harness/memory_vtt.py: scoped, independently reviewed evidence; frozen runs do not update; support is not policy success.',
        'ask_and_bound_reply': 'Orchestrator.tick/answer <-> policy_server.control_message: bounded ASK, episode/question IDs, source-labeled human/sim replies.',
        'replan_after_repeated_failure': 'PENDING: VLM recovery replanning is not wired; Task 1 uses WamPolicy._skip_failed_object as the orchestrator replanner (rule-based).',
        'finger_and_eef_proprio': 'Task0Postconditions and Task1Postconditions consume finger/EEF proprioception. This is proxy evidence, not proof of target-object placement. No admitted Task 0 outcome verifier or safe post-verification resume; force/tactile/current pending.',
    }
    assert evidence.keys() == {e[2] for e in EDGES}


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
        'scope': '2026-09-30 candidate HOLD and release-state isolation: CPU-verified interface changes, not physical success; 30k training complete, extra 20k not started',
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
        'pending_pilot': ['Replace task-specific colour target detection with category-general RGB-D grounding.',
                          'Admit independent target-bound outcome evidence and safe resume after candidate HOLD.',
                          'Verify the full Task 0 pick / press / return chain separately from BDDL toggled_on.',
                          'Add force, tactile or motor-current contact evidence before claiming grasp-empty recovery.'],
        'audited_evidence': {
            'demo_action_replay': {'summary': f'{REPLAY_RUN}/analysis/summary.json', 'train_instance': 290, 'episode': 389,
                                   'success': True, 'q_score': 1.0, 'interpretation': 'port physics + evaluator valid for Task 1; not a policy result'},
            'wam_v1_admission': {'summary': [f'{ADMISSION}/best/summary.json', f'{ADMISSION}/last/summary.json'],
                                 'admitted_contracts': 0, 'interpretation': 'direction right, reach 0.25-0.55 of demo, grasp/release late'},
            'gripper_copycat': {'summary': COPYCAT, 'follow_pooled': 0.965, 'interpretation': 'gripper command copies the finger-opening input'},
        },
        'runtime_limits': [
            'Semantic target detection is currently task-specific colour segmentation; raw depth obstacle points and the dynamic robot-body collision envelope are wired and CPU regression-tested.',
            'Task 1 has closed-loop Q=0 evidence: the chair was dragged during travel before the safety fixes; no post-fix task success exists yet.',
            'Task 1 WAM manipulation has a measured 7-9 cm end-effector error and failed to grasp even when demonstration base motion was supplied.',
            'Task 0 V3 completed 30k training and six 5k-spaced development evaluations, all failed/Q0. The separate old 25k repeat plan has one failed completed episode; nine remain unrun.',
            'Candidate HOLD and release-state isolation passed CPU tests only; independent outcome verification and safe resume remain pending.',
            'Grasping uses contact-triggered assisted constraints; no pure-friction or real-world grasp reliability is established.',
            'Completion inputs are visual/proprioceptive proxies; Task 0 target placement is not independently verified, and force/tactile/current sensing remain pending.',
            'Object poses / BDDL / Q-score are post-hoc evaluator audit only, with no path to policy.',
            'No long-horizon success, no generalization result, no matched RL comparison.',
        ],
        'evidence_sha256': {str(path.relative_to(ROOT)): digest(path) for path in [
            ROOT / 'harness/policy_server.py', ROOT / 'harness/orchestrator.py', ROOT / 'harness/docking_controller.py',
            ROOT / 'harness/task0_predicates.py', ROOT / 'harness/task1_predicates.py', ROOT / 'harness/task1_perception_adapter.py',
            ROOT / 'harness/object_perception.py', ROOT / 'harness/wam_v3.py',
            ROOT / f'{REPLAY_RUN}/analysis/summary.json', ROOT / f'{ADMISSION}/best/summary.json',
            ROOT / f'{ADMISSION}/last/summary.json', ROOT / COPYCAT]},
        'sha256': {Path(path).name: digest(path) for path in artifacts},
    }, indent=2) + '\n')
    print(f'PASS: {len(NODES)} modules, {len(EDGES)} checked arrows; SVG and 3072×1830 PNG in {OUT}')


if __name__ == '__main__':
    main()
