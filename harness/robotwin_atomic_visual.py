"""Independent, unadmitted image-only atomic predicates. No policy/eval integration.

Import is CPU-only. AtomicVisualVerifier(backend).query(request) accepts strict
native RGB packets and returns raw text + parsed four-field visual observations.
Backend must implement complete(prompt, images), guard(), resources(). Existing
robotwin_vlm_worker.GpuBackend can be injected without changing its source.
"""
from __future__ import annotations
import json
import time
from harness import robotwin_vlm_worker as v

API_VERSION = 1
STATIC_PREDICATES = ('drawer_open', 'object_between_gripper_fingers', 'object_visibly_inside_open_drawer')
TEMPORAL_PREDICATE = 'visible_release'
PREDICATES = STATIC_PREDICATES + (TEMPORAL_PREDICATE,)
COMMON = '''Use only the supplied RGB pixels. Do not infer a task stage, controller command, intended outcome or hidden state. Multiple objects may be present: identify the particular visible subject, without assuming a goal object.
Return ONLY JSON with exactly four fields. state: one of the strings "true", "false", "unknown". views: exact supporting camera labels from cam_head, cam_left_wrist, cam_right_wrist. evidence: at most 20 words naming an actually observed component and spatial relation or obstruction; cite only views showing that cue, not all views by default. object_description: a short visual description of the subject, or say it is not identifiable. Do not repeat the decision rules as evidence. No task success, actions or plans.'''
PROMPTS = {
    'drawer_open': '''Is any drawer visibly open?
The three RGB views are simultaneous. true requires a front visibly displaced from its cabinet with a visible opening, interior or sidewall; a handle alone is insufficient. false requires clearly visible closed drawer fronts aligned with the cabinet and no opening. If missing views, occlusion or ambiguous geometry prevents either conclusion, answer unknown. Describe the drawer/cabinet in object_description.
''' + COMMON,
    'object_between_gripper_fingers': '''Is a separate movable object visibly between two opposing gripper fingers?
The three RGB views are simultaneous. This is STATIC visible geometry, NOT grasp retention, contact force, lifting or stable holding. Exclude the robot itself and the fixed drawer handle. true requires an identifiable movable object clearly in the space between identifiable opposing fingers. false requires a clearly visible relevant finger gap that is empty or a clearly identified object outside it. Proximity, an obscured object, incomplete finger visibility, or uncertain object/finger correspondence requires unknown. Do not infer force or retention.
''' + COMMON,
    'object_visibly_inside_open_drawer': '''Is a particular object's complete visible body inside a visibly open drawer?
The three RGB views are simultaneous. This is visible image-space placement, NOT certified 3D containment or stability. true requires an identifiable object's full visible silhouette/outline to be uncut and unoccluded, with all of that visible body inside the observed open interior boundaries and not across the rim. A gripper or wall hiding part of the body, a cut-off outline, closed/hidden interior, or uncertain boundary requires unknown. false requires clear visible body outside or crossing the observed open-drawer boundary. Do not assume hidden portions are inside.
''' + COMMON,
    'visible_release': '''Do PREVIOUS and CURRENT images directly show a visible release transition of the same identified object from the same gripper?
Two montages are supplied in that order; each has the three simultaneous views. true requires identifiable matching object appearance across endpoints, the object visibly between that gripper's opposing fingers PREVIOUSLY, and a genuine visible gap between that same object and fingers CURRENTLY. Open fingers alone, object disappearance, an object now in a drawer, or assumed intermediate actions are insufficient. false requires clear correspondence and the object still visibly between those fingers currently. Otherwise, including occlusion, uncertain identity or missing prior finger-object geometry, answer unknown. These sparse endpoints cannot certify physical contact history, uninterrupted tracking, release time or stability.
''' + COMMON,
}
VERSIONS = {key: key + ':train_atomic_v1' for key in PREDICATES}
SCOPES = {
    'drawer_open': 'visible_opening_only',
    'object_between_gripper_fingers': 'static_visible_geometry_not_grasp_retention',
    'object_visibly_inside_open_drawer': 'visible_body_only_not_full_3d_containment_or_stability',
    'visible_release': 'two_endpoint_visible_identity_and_separation_not_continuous_physical_history',
}


def decode_request(value):
    v.keys(value, ('version', 'id', 'predicate', 'current'), ('previous',))
    if type(value['version']) is not int or value['version'] != API_VERSION:
        raise v.ProtocolError('atomic_version_mismatch')
    rid = v.identifier(value['id'])
    pred = value['predicate']
    if type(pred) is not str or pred not in PREDICATES:
        raise v.ProtocolError('unknown_atomic_predicate')
    if (pred == TEMPORAL_PREDICATE) != ('previous' in value):
        raise v.ProtocolError('previous_required_only_for_temporal_release')
    current = v.decode_frame(value['current'])
    previous = v.decode_frame(value['previous']) if 'previous' in value else None
    if previous and (previous['episode_id'] != current['episode_id'] or previous['observation_id'] >= current['observation_id']):
        raise v.ProtocolError('invalid_previous_binding')
    return {'id': rid, 'predicate': pred, 'current': current, 'previous': previous}


def build_input(decoded):
    images = ([v.montage(decoded['previous'])] if decoded['previous'] else []) + [v.montage(decoded['current'])]
    return PROMPTS[decoded['predicate']], images


def parse(raw):
    if type(raw) is not str or not raw.strip() or len(raw) > v.MAX_RAW_CHARS:
        raise v.ProtocolError('invalid_model_text')
    text = raw.strip()
    if text.startswith('```json\n') and text.endswith('\n```'): text = text[8:-4].strip()
    elif text.startswith('```\n') and text.endswith('\n```'): text = text[4:-4].strip()
    value = v.strict_json(text)
    v.keys(value, ('state', 'views', 'evidence', 'object_description'))
    if type(value['state']) is not str or value['state'] not in ('true', 'false', 'unknown'):
        raise v.ProtocolError('invalid_state')
    views = value['views']
    if type(views) is not list or not views or any(type(x) is not str or x not in v.VIEWS for x in views) or len(set(views)) != len(views):
        raise v.ProtocolError('invalid_views')
    for field, maximum in (('evidence', 600), ('object_description', 256)):
        if type(value[field]) is not str or not value[field].strip() or len(value[field]) > maximum:
            raise v.ProtocolError('invalid_' + field)
    return value


class AtomicVisualVerifier:
    def __init__(self, backend):
        self.backend = backend
        self.seen = set()
        self.frame_hashes = {}

    def query(self, request):
        start = time.monotonic()
        decoded = decode_request(request)  # invalid requests never call backend
        if decoded['id'] in self.seen:
            raise v.ProtocolError('duplicate_atomic_request_id')
        if len(self.seen) >= 10000:
            raise v.ProtocolError('atomic_session_limit')
        for frame in (decoded['previous'], decoded['current']):
            if frame:
                key = (frame['episode_id'], frame['observation_id'])
                if key in self.frame_hashes and self.frame_hashes[key] != frame['rgb_sha256']:
                    raise v.ProtocolError('observation_id_pixel_conflict')
        self.seen.add(decoded['id'])
        for frame in (decoded['previous'], decoded['current']):
            if frame: self.frame_hashes[(frame['episode_id'], frame['observation_id'])] = frame['rgb_sha256']
        pred = decoded['predicate']
        prompt, images = build_input(decoded)
        reply = {'version': API_VERSION, 'id': decoded['id'], 'predicate': pred, 'status': v.STATUS,
                 'prompt_version': VERSIONS[pred], 'prompt_sha256': v.digest(prompt.encode()),
                 'claim_scope': SCOPES[pred], 'binding': v.request_binding(decoded),
                 'montage_pixel_sha256': [v.digest(im.tobytes()) for im in images],
                 'raw_text': '', 'parsed': None, 'errors': [], 'cost': {}}
        try:
            self.backend.guard()
            infer_start = time.monotonic()
            raw, details = self.backend.complete(prompt, images)
            reply.update(raw_text=raw, generation=details)
            reply['cost']['inference_s'] = time.monotonic() - infer_start
            reply['cost']['output_tokens'] = details.get('new_tokens')
            reply['cost']['input_shape'] = details.get('input_shapes', {}).get('input_ids')
            self.backend.guard()
            reply['resources'] = self.backend.resources()
            try:
                reply['parsed'] = parse(raw)
                reply['outcome'] = 'ok'
            except v.ProtocolError as exc:
                reply['outcome'] = 'parse_error'; reply['errors'].append(str(exc))
        except Exception as exc:
            reply.update(outcome='runtime_error', parsed=None)
            reply['errors'].append(type(exc).__name__ + ': ' + str(exc))
        reply['cost']['total_s'] = time.monotonic() - start
        return reply
