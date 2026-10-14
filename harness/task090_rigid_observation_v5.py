"""Opt-in short appearance keyframes over immutable v4 material-point IDs.

Only correspondence maintenance and atomic commit change. One recent accepted
view plus the original view; no inverse-pose landmarks, depth interpolation,
semantic rebinding, new thresholds, or runtime/admission wiring.
"""
from dataclasses import dataclass

import numpy as np

from harness import task090_rigid_observation_v3 as gates
from harness.task090_rigid_observation_v4 import (
    Task090RigidObservationV4, immutable_array, seeded_flow_matches,
)

ACCEPTED = ('tracking_initialized', 'rigid_motion_measured', 'anchor_recovery_pending')
VERSION = 'immutable_original_short_keyframe_rgbd_v5'


@dataclass(frozen=True)
class AppearanceKeyframe:
    gray: np.ndarray
    uv: np.ndarray
    ids: np.ndarray
    step: int


class Task090RigidObservationV5(Task090RigidObservationV4):
    # Parent methods replace these arrays rather than mutate them. A shallow
    # transaction preserves exact old objects without copying the dense surface.
    _FRAME_FIELDS = (
        'uv', 'reference', 'errors', 'anchor_uv', 'anchor_reference', 'origin_floor',
        'origin_margin', 'origin_surface', 'previous_gray', 'previous_depth',
        'previous_camera', 'previous_eef', 'previous_width', 'previous_pose',
        'previous_hands', 'last_good_step', 'anchor_total', 'error_high_water',
        'total_added', 'keyframes', 'track_id', 'original_bank', 'last_accepted_mask',
        'appearance_keyframe',
    )

    def reset(self):
        super().reset()
        self.appearance_keyframe = None
        self._frame_before = None

    def _begin_frame(self):
        self._frame_before = {key: getattr(self, key, None) for key in self._FRAME_FIELDS}

    @staticmethod
    def _validate_floor(floor):
        if floor is None:
            raise ValueError('floor_not_observed')
        coefficients = np.asarray(floor['coefficients'], float)
        margin = float(floor['margin_m'])
        if coefficients.shape != (3,) or not np.isfinite(coefficients).all() or not np.isfinite(margin) or margin < 0:
            raise ValueError('invalid_floor_geometry')

    def _initialize(self, *args):
        self._begin_frame()
        self._validate_floor(args[8])
        super()._initialize(*args)

    def _matches(self, source, gray, depth, camera, appearance, seeds, visible, excluded, diagnostic):
        """Raw RGB-D observations only; return at most one observation per ID."""
        source_gray, source_uv, source_ids, name = source
        eligible = visible[source_ids] & ~excluded[source_ids]
        ids = source_ids[eligible]
        uv, valid = seeded_flow_matches(source_gray, gray, source_uv[eligible], seeds[ids])
        counts = dict(projected=len(ids), after_lk=int(valid.sum()))
        ids, uv = ids[valid], uv[valid]
        u, v = np.rint(uv).astype(int).T
        valid = np.isfinite(depth[v, u]) & (depth[v, u] > .05) & (depth[v, u] < self.calibration['depth_max_m'])
        counts['after_depth'] = int(valid.sum())
        valid &= appearance[v, u] > 0
        counts['after_appearance'] = int(valid.sum())
        ids, uv = ids[valid], uv[valid]
        points = gates.sample_points(camera, depth, uv)
        if self.self_filter is None:
            if self._current_hands is None:
                raise ValueError('current_hand_observations_missing')
            keep = gates.hand_separated(points, self._current_hands)
        else:
            keep = self._self_keep(points, self._current_state, counts, 'anchors')
        counts['after_self'] = int(keep.sum())
        diagnostic['correspondence_sources'][name] = counts
        return ids[keep], uv[keep], points[keep]

    def _tracked(self, gray, depth, camera, appearance, diagnostic):
        self._begin_frame()
        bank = self.original_bank
        if bank is None:
            raise ValueError('original_landmark_bank_missing')
        r0, t0 = self.previous_pose
        projected = (bank.reference @ r0.T + t0 - camera.t) @ camera.R
        positive = np.isfinite(projected).all(axis=1) & (projected[:, 2] > .05)
        seeds = np.zeros_like(bank.uv, dtype=np.float32)
        seeds[positive] = projected[positive, :2] / projected[positive, 2, None] * camera.f + [camera.cx, camera.cy]
        h, w = gray.shape
        visible = positive & np.isfinite(seeds).all(axis=1) & (seeds[:, 0] >= 1) & (seeds[:, 0] < w-1) & (seeds[:, 1] >= 1) & (seeds[:, 1] < h-1)
        diagnostic.update(pose_source='original_anchors_only', original_projection_seeds=int(visible.sum()),
                          correspondence_sources={}, reobserved_original_anchors=0)
        recent = self.appearance_keyframe
        sources = []
        if recent is not None:
            if not 0 < self.last_step-recent.step <= self.max_frame_gap:
                raise ValueError('appearance_keyframe_expired_no_rebind')
            sources.append((recent.gray, recent.uv, recent.ids, 'recent'))
        sources.append((bank.gray, bank.uv, np.arange(len(bank.uv)), 'original_fallback'))
        used = np.zeros(len(bank.uv), bool)
        matches = []
        for source in sources:
            match = self._matches(source, gray, depth, camera, appearance, seeds, visible, used, diagnostic)
            used[match[0]] = True  # Recent valid measurement wins; never average alternatives.
            matches.append(match)
        ids, uv, points = (np.concatenate([match[i] for match in matches]) for i in range(3))
        order = np.argsort(ids)
        ids, uv, points = ids[order], uv[order], points[order]
        counts = list(diagnostic['correspondence_sources'].values())
        diagnostic.update(original_anchors_after_lk=sum(c['after_lk'] for c in counts),
                          original_anchors_after_depth=sum(c['after_depth'] for c in counts),
                          original_anchors_after_appearance=sum(c['after_appearance'] for c in counts),
                          original_anchors_after_lk_depth_appearance=sum(c['after_appearance'] for c in counts),
                          original_anchors_after_hand_separation=len(ids),
                          original_anchors_self_rejected=sum(c['after_appearance']-c['after_self'] for c in counts))
        # Diagnostic spans are not a new gate. The measured depth is the original
        # nearest pixel, not a foreground/background average. Ownership outliers
        # must be rejected by the unchanged rigid consensus or the frame is Unknown.
        spans = []
        for u, v in np.rint(uv).astype(int):
            patch = depth[max(0, v-2):v+3, max(0, u-2):u+3]
            finite = patch[np.isfinite(patch)]
            spans.append(None if not len(finite) else float(np.ptp(finite)))
        diagnostic['depth_samples'] = dict(original_ids=ids.tolist(), local_span_5x5_m=spans,
            sampling='nearest_pixel_no_cross_surface_average', rejection_gate='original_rigid_consensus_1cm')
        try:
            r, t, rms, keep, maximum = gates.original_anchor_check(bank.reference[ids], points, diagnostic=diagnostic)
        except ValueError as error:
            diagnostic['depth_samples'].update(rejected_frame_reason=str(error), accepted_original_ids=[])
            raise
        diagnostic['depth_samples'].update(rejected_original_ids=ids[~keep].tolist(),
                                            rejection_reason='not_in_final_original_rigid_consensus')
        ids, uv, points = ids[keep], uv[keep], points[keep]
        reference, errors = bank.reference[ids].copy(), bank.errors[ids].copy()
        self.error_high_water = max(self.error_high_water, float(np.max(errors)))
        self._pending_mask = np.zeros(len(bank.uv), bool)
        self._pending_mask[ids] = True
        diagnostic.update(overlap_after_lk_depth_appearance=len(reference), overlap_inliers=len(reference),
                          fit_rms_m=rms, overlap_pose_error_max_m=maximum,
                          reobserved_original_anchors=0 if self.last_accepted_mask is None else int(np.sum(self._pending_mask & ~self.last_accepted_mask)),
                          accepted_original_ids=ids.tolist())
        return uv, reference, points, errors, r, t, rms, uv.copy(), reference.copy()

    def update(self, *args, **kwargs):
        self._frame_before = None
        report = super().update(*args, **kwargs)
        diagnostic = report['rigid_tracking']
        accepted = diagnostic['status'] in ACCEPTED
        if accepted:
            try:
                # Parent does not retain the current floor; independently recheck
                # its domain, not a different margin/height threshold.
                self._validate_floor(gates.observed_floor_plane(self.previous_depth, self.previous_camera))
                self._validate_floor(dict(coefficients=self.origin_floor, margin_m=self.origin_margin))
                if any(isinstance(value, (float, np.floating)) and not np.isfinite(value)
                       for value in diagnostic.values()):
                    raise ValueError('invalid_height_or_geometry_diagnostic')
                if diagnostic.get('assumed_measurement_allowance_m', 0.) < 0:
                    raise ValueError('invalid_height_measurement_allowance')
            except (ValueError, TypeError, KeyError) as error:
                accepted = False
                self._unknown(diagnostic, type(error).__name__ + ':' + str(error))
                report.update(status='unknown', reason=diagnostic['reason'])
                for key, value in list(diagnostic.items()):
                    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
                        diagnostic[key] = None
        if accepted:
            ids = np.flatnonzero(self.last_accepted_mask)
            self.appearance_keyframe = AppearanceKeyframe(
                immutable_array(self.previous_gray), immutable_array(self.anchor_uv),
                immutable_array(ids), self.last_good_step)
        elif self._frame_before is not None:
            # Consume last_step/seen and retain the failure/recovery latch. Roll
            # back only observation-derived state, including late height failure.
            for key, value in self._frame_before.items():
                setattr(self, key, value)
            self._unknown(diagnostic, report['reason'])
        self._frame_before = None
        diagnostic.update(tracker_version=VERSION, appearance_keyframe_committed=accepted,
                          appearance_keyframe_step=None if self.appearance_keyframe is None else self.appearance_keyframe.step,
                          appearance_keyframe_count=int(self.appearance_keyframe is not None),
                          original_bank_size=0 if self.original_bank is None else len(self.original_bank.uv),
                          last_accepted_original_count=0 if self.last_accepted_mask is None else int(self.last_accepted_mask.sum()))
        return report
