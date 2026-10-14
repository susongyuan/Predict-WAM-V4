"""CPU-only RGB-D refinement of *unverified* semantic surface proposals.

No model, task/category constants, top-k ranking, threshold fitting, gap filling,
object-center inference, or presence/absence certification lives here. Each
returned cloud is one connected visible surface, not necessarily one object.
Touching lookalikes can remain connected; an occluded object can split. A coarse
score field cannot establish instance identity, even if only one surface remains.

Caller contract:
* RGB is uint8 HxWx3/4 (RGB, not BGR); depth is aligned HxW linear z in metres.
* camera.W/H must match and camera.backproject(u, v, z) returns Nx3 base points,
  as in harness.object_perception.Camera. Camera calibration is the caller's job.
* Positive and negative scores share a scale and a full-frame registered 2D grid.
  Cell (r,c) owns pixels floor(v*Gh/H)==r, floor(u*Gw/W)==c; there is no resize of
  depth, RGB, or masks. Fields larger than the image are rejected, not subsampled.
* Finite score/depth/point magnitudes must be <=1e6; larger numerical domains
  are rejected before float64 conversion, median or raw-statistics reductions.
* All semantic/geometric thresholds and their provenance are caller-supplied.
  The utility cannot audit provenance strings or certify threshold calibration.
* robot_self_mask is an optional aligned bool image, True means exclude. Without
  it, robot pixels may survive. A supplied mask needs robot_self_mask_source.

The graph has four-neighbor pixel edges, gated by Euclidean base-frame distance
and max-channel RGB difference. Only independently supported pixels enter it;
there is no growth into low-margin regions. Thresholds are local, so connected
chains can have large total color/metric spans. Returned pixels are not full
object masks. Component IDs/order are frame-local row-major, not persistent IDs.

Work/memory is bounded by max_image_pixels; component grouping is O(P log P).
There are no per-component full-image scans or Python per-pixel loops. OpenCV
thread settings belong to the caller (this library never changes global state).
A candidate-budget overflow returns *no* subset, with an explicit diagnostic.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from numbers import Integral, Real

import cv2
import numpy as np


@dataclass(frozen=True)
class SurfaceThresholds:
    """Required caller settings, not pretrained defaults or confidence levels.

    Score and margin comparisons are inclusive; margin_min must be positive.
    Depth interval is open. Neighbor bounds are inclusive. RGB differences use
    the native uint8 0..255 scale; 255 explicitly disables color-edge rejection.
    min_component_pixels counts native pixels, not coarse cells/independent votes.
    """
    positive_min: float
    margin_min: float
    depth_min_m: float
    depth_max_m: float
    neighbor_distance_m: float
    neighbor_color_linf: float
    min_component_pixels: int

    def __post_init__(self):
        for name in ("positive_min", "margin_min", "depth_min_m", "depth_max_m",
                     "neighbor_distance_m", "neighbor_color_linf"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
                raise ValueError(f"{name} must be a finite real number")
        if self.margin_min <= 0:
            raise ValueError("margin_min must be positive, not a positive-only ranking")
        if not 0 <= self.depth_min_m < self.depth_max_m:
            raise ValueError("require 0 <= depth_min_m < depth_max_m")
        if self.neighbor_distance_m <= 0:
            raise ValueError("neighbor_distance_m must be positive")
        if not 0 <= self.neighbor_color_linf <= 255:
            raise ValueError("neighbor_color_linf must be on uint8 scale [0,255]")
        _positive_int(self.min_component_pixels, "min_component_pixels")


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _real_array(value, name):
    array = np.asarray(value)
    if array.dtype.kind not in "fiu":
        raise ValueError(f"{name} must contain real numeric values")
    # Bound the supported arithmetic domain before float64 conversion/reduction.
    # Nonfinite depth/score samples remain unsupported pixels, not huge finite values.
    finite = np.isfinite(array)
    if np.any(array[finite] > 1e6) or np.any(array[finite] < -1e6):
        raise ValueError(f"{name} exceeds supported numeric magnitude 1e6")
    return np.asarray(array, dtype=np.float64)


def refine_target_surfaces(
    rgb, depth, camera, positive_scores, negative_scores, *,
    thresholds: SurfaceThresholds, provenance: Mapping[str, str],
    robot_self_mask=None, max_image_pixels: int = 1_048_576,
    max_candidates: int = 256,
):
    """Return separate metric candidates plus labels/diagnostics, never a winner.

    Result: {status, candidates, labels, diagnostics, thresholds, provenance,
             visibility='unknown', object_presence_verified=False,
             physical_success_verified=False}.
    labels: int32 HxW, 0 unsupported/discarded, 1..N match candidate component_id.
    Each candidate has row-major pixel_uv (Nx2 int32), cloud (Nx3 float64 base
    points), bbox_xyxy (exclusive upper bounds), pixels, score_statistics, and
    xyz_base/surface_pixel_uv/surface_depth_m from one actual member nearest its
    own cloud median. The latter is an observed sample, NOT a center or grasp.
    Candidate flags explicitly keep presence, center and grasp unverified.

    Invalid contracts raise ValueError. Missing support returns an empty result
    with diagnostic status, NOT evidence of absence. candidate_budget_exceeded
    also returns empty candidates/labels, with the full eligible component count;
    it does not silently prefer a subset or erase ambiguity by merging it.
    """
    if not isinstance(thresholds, SurfaceThresholds):
        raise ValueError("thresholds must be explicit SurfaceThresholds")
    _positive_int(max_image_pixels, "max_image_pixels")
    _positive_int(max_candidates, "max_candidates")
    if not isinstance(provenance, Mapping):
        raise ValueError("provenance must be a mapping")
    required = ["score_source", "threshold_source", "alignment_source"]
    if robot_self_mask is not None:
        required.append("robot_self_mask_source")
    for name in required:
        if not isinstance(provenance.get(name), str) or not provenance[name].strip():
            raise ValueError(f"nonempty provenance.{name} is required")
    if any(not isinstance(k, str) or not isinstance(v, str) for k, v in provenance.items()):
        raise ValueError("provenance keys and values must be strings")

    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[2] not in (3, 4) or image.dtype != np.uint8:
        raise ValueError("rgb must be uint8 HxWx3/4 RGB")
    h, w = image.shape[:2]
    if h < 1 or w < 1 or h * w > max_image_pixels:
        raise ValueError("image empty or exceeds max_image_pixels")
    if (getattr(camera, "W", None) != w or getattr(camera, "H", None) != h
            or not callable(getattr(camera, "backproject", None))):
        raise ValueError("camera.W/H and backproject must match the RGB-D image")
    # Check shapes before converting or allocating large input copies.
    raw_depth = np.asarray(depth)
    if raw_depth.shape != (h, w):
        raise ValueError("depth must be aligned HxW linear z in metres")
    d = _real_array(raw_depth, "depth")
    raw_pos, raw_neg = np.asarray(positive_scores), np.asarray(negative_scores)
    if (raw_pos.ndim != 2 or raw_pos.shape != raw_neg.shape
            or not 1 <= raw_pos.shape[0] <= h or not 1 <= raw_pos.shape[1] <= w):
        raise ValueError("scores must be matching nonempty 2D grids no larger than RGB")
    positive = _real_array(raw_pos, "positive_scores")
    negative = _real_array(raw_neg, "negative_scores")
    if robot_self_mask is None:
        robot = np.zeros((h, w), dtype=bool)
    else:
        robot = np.asarray(robot_self_mask)
        if robot.shape != (h, w) or robot.dtype != np.bool_:
            raise ValueError("robot_self_mask must be an aligned bool image")

    # Exact nearest-cell lifting also defines ownership at non-divisible sizes.
    gh, gw = positive.shape
    rows = (np.arange(h, dtype=np.int64) * gh // h)[:, None]
    cols = (np.arange(w, dtype=np.int64) * gw // w)[None, :]
    pos, neg = positive[rows, cols], negative[rows, cols]
    with np.errstate(invalid="ignore", over="ignore"):
        margin = pos - neg
    semantic = (np.isfinite(pos) & np.isfinite(neg) & np.isfinite(margin)
                & (pos >= thresholds.positive_min) & (margin >= thresholds.margin_min))
    depth_ok = np.isfinite(d) & (d > thresholds.depth_min_m) & (d < thresholds.depth_max_m)
    supported_depth = semantic & depth_ok
    support = supported_depth & ~robot
    diagnostics = {
        "input_pixels": h * w,
        "semantic_support_pixels": int(semantic.sum()),
        "depth_supported_pixels": int(supported_depth.sum()),
        "excluded_robot_pixels": int((supported_depth & robot).sum()),
        "robot_mask_applied": robot_self_mask is not None,
        "nonfinite_backprojection_pixels": 0,
        "surface_support_pixels": 0,
        "connected_components": 0,
        "small_components": 0,
        "eligible_components": 0,
        "returned_components": 0,
    }
    result = {
        "status": "no_semantic_support", "candidates": [],
        "labels": np.zeros((h, w), dtype=np.int32), "diagnostics": diagnostics,
        "thresholds": asdict(thresholds), "provenance": dict(provenance),
        "visibility": "unknown", "object_presence_verified": False,
        "physical_success_verified": False,
    }
    if not semantic.any():
        return result
    if not supported_depth.any():
        result["status"] = "no_valid_depth_support"
        return result
    if not support.any():
        result["status"] = "all_depth_support_excluded_by_robot"
        return result

    v, u = np.nonzero(support)
    points = _real_array(camera.backproject(u, v, d[v, u]), "backprojected points")
    if points.shape != (len(u), 3):
        raise ValueError("camera.backproject must return Nx3 base-frame points")
    finite = np.isfinite(points).all(axis=1)
    diagnostics["nonfinite_backprojection_pixels"] = int((~finite).sum())
    support[v[~finite], u[~finite]] = False
    u, v, points = u[finite], v[finite], points[finite]
    diagnostics["surface_support_pixels"] = len(u)
    if not len(u):
        result["status"] = "no_finite_backprojection"
        return result

    xyz = np.zeros((h, w, 3), dtype=np.float64)
    xyz[v, u] = points
    color = image[..., :3].astype(np.int16)
    # Edge-midpoint lattice: adjacent vertices connect only through their own
    # accepted edge. Four-connectivity never leaks through diagonal corners.
    lattice = np.zeros((2 * h - 1, 2 * w - 1), dtype=np.uint8)
    lattice[::2, ::2] = support
    for a, b, target in (
        ((slice(None), slice(None, -1)), (slice(None), slice(1, None)), (slice(None, None, 2), slice(1, None, 2))),
        ((slice(None, -1), slice(None)), (slice(1, None), slice(None)), (slice(1, None, 2), slice(None, None, 2))),
    ):
        # hypot avoids overflow from squaring otherwise finite metric coordinates.
        with np.errstate(over="ignore", invalid="ignore"):
            delta = xyz[a] - xyz[b]
            distance = np.hypot(np.hypot(delta[..., 0], delta[..., 1]), delta[..., 2])
        color_distance = np.max(np.abs(color[a] - color[b]), axis=2)
        lattice[target] = (support[a] & support[b]
                           & (distance <= thresholds.neighbor_distance_m)
                           & (color_distance <= thresholds.neighbor_color_linf))
    nlabels, expanded_labels = cv2.connectedComponents(lattice, connectivity=4)
    raw_labels = expanded_labels[::2, ::2]
    point_labels = raw_labels[v, u]
    counts = np.bincount(point_labels, minlength=nlabels)
    eligible = np.flatnonzero(counts >= thresholds.min_component_pixels)
    eligible = eligible[eligible != 0]
    diagnostics["connected_components"] = nlabels - 1
    diagnostics["small_components"] = nlabels - 1 - len(eligible)
    diagnostics["eligible_components"] = len(eligible)
    if not len(eligible):
        result["status"] = "no_components_above_min_pixels"
        return result
    if len(eligible) > max_candidates:
        result["status"] = "candidate_budget_exceeded"
        return result

    # Stable spatial ordering independent of OpenCV's internal component IDs.
    first_pixel = np.full(nlabels, h * w, dtype=np.int64)
    np.minimum.at(first_pixel, point_labels, v * w + u)
    eligible = eligible[np.argsort(first_pixel[eligible], kind="stable")]
    lookup = np.zeros(nlabels, dtype=np.int32)
    lookup[eligible] = np.arange(1, len(eligible) + 1, dtype=np.int32)
    result["labels"] = lookup[raw_labels]
    order = np.argsort(point_labels, kind="stable")
    offsets = np.r_[0, np.cumsum(counts)]
    for component_id, raw_id in enumerate(eligible, 1):
        indices = order[offsets[raw_id]:offsets[raw_id + 1]]
        cu, cv, cloud = u[indices], v[indices], points[indices]
        median = np.median(cloud, axis=0)
        delta = cloud - median
        k = int(np.argmin(np.hypot(np.hypot(delta[:, 0], delta[:, 1]), delta[:, 2])))
        stats = {}
        for name, field in (("positive", pos), ("negative", neg), ("margin", margin)):
            values = field[cv, cu]
            stats[name] = {"min": float(values.min()), "max": float(values.max()),
                           "mean": float(values.mean())}
        result["candidates"].append({
            "component_id": component_id, "pixels": len(cu),
            "bbox_xyxy": [int(cu.min()), int(cv.min()), int(cu.max() + 1), int(cv.max() + 1)],
            "pixel_uv": np.column_stack((cu, cv)).astype(np.int32), "cloud": cloud,
            "xyz_base": cloud[k].tolist(), "surface_pixel_uv": [int(cu[k]), int(cv[k])],
            "surface_depth_m": float(d[cv[k], cu[k]]), "score_statistics": stats,
            "source": "rgbd", "point_semantics": "observed_surface_sample",
            "semantic_status": "unverified_surface_candidate",
            "object_presence_verified": False, "center_valid": False,
            "grasp_reference_valid": False,
        })
    diagnostics["returned_components"] = len(result["candidates"])
    result["status"] = "candidates_available"
    return result
