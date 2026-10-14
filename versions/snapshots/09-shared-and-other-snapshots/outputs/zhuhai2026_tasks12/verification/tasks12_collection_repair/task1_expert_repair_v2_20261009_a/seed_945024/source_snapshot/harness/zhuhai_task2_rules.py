"""Auditable physical proxy rules for the 2026 Zhuhai task-2 simulations.

The PDF is a consultation draft, not a simulator specification. Simulation
tolerances below are declared proxies; competition coefficients are excluded.
"""
from dataclasses import dataclass
from math import cos, isfinite, radians, sqrt
from typing import Mapping, Sequence


LIFE_DESTINATIONS = {
    "body_text": {"bowl": "white", "spoon": "white", "cup": "green", "plate": "blue"},
    "scoring_table": {"bowl": "white", "plate": "white", "cup": "green", "spoon": "blue"},
}
PRODUCTION_DESTINATIONS = {"cd294": "red", "inductor": "green", "mkp62": "blue"}
LIFE_POINTS = {"plate": (5, 4), "bowl": (4, 4), "cup": (3, 3), "spoon": (4, 3)}
PRODUCTION_POINTS = {"cd294": (3, 5), "inductor": (7, 5), "mkp62": (10, 5)}
PRODUCTION_SPECS = {
    "cd294": {"dimensions_m": (0.035, 0.035, 0.070), "mass_range_kg": (0.140, 0.160)},
    "inductor": {"dimensions_m": (0.037, 0.027, 0.040), "mass_range_kg": (0.180, 0.220)},
    "mkp62": {"dimensions_m": (0.030, 0.013, 0.025), "mass_range_kg": (0.020, 0.035)},
}


@dataclass(frozen=True)
class PhysicalState:
    """Geometry in world coordinates, measured from simulation rigid bodies."""
    corners: Sequence[Sequence[float]]
    linear_velocity: Sequence[float]
    angular_velocity: Sequence[float]
    released: bool
    upright_cosine: float = 1.0
    touching_robot: bool = False
    touching_gripper: bool | None = None
    support_destinations: Sequence[str] | None = None

    def serialize(self):
        return {
            "world_corners_m": [[float(value) for value in corner] for corner in self.corners],
            "linear_velocity_m_s": [float(value) for value in self.linear_velocity],
            "angular_velocity_rad_s": [float(value) for value in self.angular_velocity],
            "released": bool(self.released), "touching_robot": bool(self.touching_robot),
            "upright_cosine": float(self.upright_cosine),
            "touching_gripper": bool(self.touching_robot if self.touching_gripper is None else self.touching_gripper),
            "support_destinations": None if self.support_destinations is None else list(self.support_destinations),
        }


def wholly_contained(corners, interior_min, interior_max, tolerance=0.001):
    """Every transformed object-bound corner must lie inside the destination."""
    return len(corners) > 0 and len(interior_min) == len(interior_max) == 3 and all(
        len(corner) == 3 and all(isfinite(value) and lo - tolerance <= value <= hi + tolerance
                               for value, lo, hi in zip(corner, interior_min, interior_max))
        for corner in corners
    )


def norm(vector):
    return sqrt(sum(float(value) ** 2 for value in vector))


class Task2Judge:
    """Track continuous grasp holds and final stable released placements.

    Call observe once per physics step in both expert collection and evaluation.
    Production's >5s threshold deliberately differs from life's >=5s.
    """
    def __init__(self, track, mapping="body_text", stable_seconds=1.0):
        if track not in {"life", "production"}:
            raise ValueError(track)
        self.track = track
        self.mapping_name = mapping
        self.destinations = dict(LIFE_DESTINATIONS[mapping] if track == "life" else PRODUCTION_DESTINATIONS)
        self.points = LIFE_POINTS if track == "life" else PRODUCTION_POINTS
        self.stable_seconds = stable_seconds
        self.hold = {key: 0.0 for key in self.destinations}
        self.grasp_awarded = {key: False for key in self.destinations}
        self.stable = {key: 0.0 for key in self.destinations}
        self.placement_valid = {key: False for key in self.destinations}

    def observe(self, dt: float, states: Mapping[str, PhysicalState], interiors, table_surface):
        if not isfinite(dt) or dt < 0:
            raise ValueError("physics timestep must be finite and nonnegative")
        for key, color in self.destinations.items():
            state = states[key]
            # Clearance is an explicit simulation proxy for "fully off table".
            lifted = min(point[2] for point in state.corners) > table_surface + 0.01
            gripper_contact = state.touching_robot if state.touching_gripper is None else state.touching_gripper
            held = lifted and not state.released and gripper_contact
            self.hold[key] = self.hold[key] + dt if held else 0.0
            threshold_met = self.hold[key] >= 5.0 - 1e-9 if self.track == "life" else self.hold[key] > 5.0 + 1e-9
            self.grasp_awarded[key] |= threshold_met
            bounds = interiors[color]
            valid = (
                wholly_contained(state.corners, bounds[0], bounds[1])
                and state.released
                and not state.touching_robot
                and norm(state.linear_velocity) < 0.015
                and norm(state.angular_velocity) < 0.15
                and (state.support_destinations is None or color in state.support_destinations)
                and (key != "cup" or state.upright_cosine >= cos(radians(10)))
            )
            self.stable[key] = self.stable[key] + dt if valid else 0.0
            # Placement is based on final current state, not permanently latched.
            self.placement_valid[key] = valid and self.stable[key] >= self.stable_seconds

    def report(self):
        items = {}
        for key, (grasp_points, placement_points) in self.points.items():
            items[key] = {
                "destination": self.destinations[key],
                "grasp_awarded": self.grasp_awarded[key],
                "placement_valid": self.placement_valid[key],
                "hold_seconds": self.hold[key],
                "stable_seconds": self.stable[key],
                "base_points": int(self.grasp_awarded[key]) * grasp_points + int(self.placement_valid[key]) * placement_points,
            }
        return {
            "track": self.track,
            "mapping": self.mapping_name,
            "draft_mapping_conflict": {"body_text": LIFE_DESTINATIONS["body_text"],
                "scoring_table": LIFE_DESTINATIONS["scoring_table"]} if self.track == "life" else None,
            "rule_source": "consultation draft PDF; simulation tolerances explicitly declared",
            "full_task_success": all(self.grasp_awarded.values()) and all(self.placement_valid.values()),
            "base_score": sum(item["base_points"] for item in items.values()),
            "base_score_max": sum(sum(pair) for pair in self.points.values()),
            "items": items,
            "simulation_tolerances": {"stable_seconds": self.stable_seconds, "linear_speed_m_s": 0.015,
                "angular_speed_rad_s": 0.15, "cup_upright_degrees": 10, "clearance_m": 0.01,
                "bounds_tolerance_m": 0.001},
            "not_verified": ["material damage", "physical dual-arm cooperation", "dexterous-hand eligibility"],
        }
