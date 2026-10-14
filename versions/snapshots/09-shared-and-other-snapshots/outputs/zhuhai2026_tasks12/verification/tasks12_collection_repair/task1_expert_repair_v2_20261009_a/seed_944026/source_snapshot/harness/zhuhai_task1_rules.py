"""Life task-1 draft rules with explicit, simulator-only physical tolerances.

The three competition objects are milk carton, squeeze bottle, and banana peel.
The draft asks for opening lids and keeping them open; final lid closure is
reported diagnostically and is deliberately not a success requirement.
"""
from math import radians, isfinite

from harness.zhuhai_task2_rules import PhysicalState, norm, wholly_contained


DESTINATIONS = {"milk_carton": "dry", "squeeze_bottle": "dry", "banana_peel": "wet"}


class Task1Judge:
    """Actual-contact grasp evidence and current, released stable containment."""

    def __init__(self, stable_seconds=1.0, open_degrees=75.0):
        if stable_seconds <= 0 or not 0 < open_degrees < 180:
            raise ValueError("invalid task-1 physical tolerances")
        self.destinations = dict(DESTINATIONS)
        self.stable_seconds = float(stable_seconds)
        self.open_angle = radians(open_degrees)
        self.hold = {key: 0.0 for key in DESTINATIONS}
        self.grasp_awarded = {key: False for key in DESTINATIONS}
        self.stable = {key: 0.0 for key in DESTINATIONS}
        self.placement_valid = {key: False for key in DESTINATIONS}
        self.passed_mouth = {key: False for key in DESTINATIONS}
        self.above_open_mouth = {key: False for key in DESTINATIONS}
        self.lid_open_awarded = {key: False for key in ("dry", "wet")}
        self.lid_open_now = dict(self.lid_open_awarded)
        self.lid_closed_now = dict(self.lid_open_awarded)
        self.bin_fixed = True
        self.elapsed_seconds = 0.0

    def observe(self, dt, states, interiors, table_surface, lid_angles, mouth_heights=None, bin_fixed=True):
        if not isfinite(dt) or dt < 0:
            raise ValueError("physics timestep must be finite and nonnegative")
        self.elapsed_seconds += float(dt)
        self.bin_fixed &= bool(bin_fixed)
        for color in self.lid_open_now:
            angle = abs(float(lid_angles[color]))
            self.lid_open_now[color] = angle >= self.open_angle
            self.lid_closed_now[color] = angle <= radians(5.0)
            self.lid_open_awarded[color] |= self.lid_open_now[color]
        for key, color in DESTINATIONS.items():
            state: PhysicalState = states[key]
            lifted = min(point[2] for point in state.corners) > table_surface + 0.01
            held = lifted and not state.released and state.touching_robot
            self.hold[key] = self.hold[key] + dt if held else 0.0
            self.grasp_awarded[key] |= self.hold[key] >= 5.0 - 1e-9
            lo, hi = interiors[color]
            # A rigid object can only enter through the open top in this scene.
            # Detect all corners below the mouth with XY inside it while lid open;
            # do not infer a valid passage merely from a final position teleport.
            mouth = (mouth_heights or {}).get(color, hi[2])
            above = self.lid_open_now[color] and all(
                lo[0] - .001 <= p[0] <= hi[0] + .001
                and lo[1] - .001 <= p[1] <= hi[1] + .001 and p[2] > mouth
                for p in state.corners
            )
            self.above_open_mouth[key] |= above
            passed = self.lid_open_now[color] and all(
                lo[0] - .001 <= p[0] <= hi[0] + .001
                and lo[1] - .001 <= p[1] <= hi[1] + .001 and p[2] <= mouth + .001
                for p in state.corners
            )
            self.passed_mouth[key] |= passed and self.above_open_mouth[key]
            valid = (self.passed_mouth[key] and self.lid_open_now[color]
                     and wholly_contained(state.corners, lo, hi)
                     and state.released and not state.touching_robot
                     and norm(state.linear_velocity) < .015
                     and norm(state.angular_velocity) < .15)
            self.stable[key] = self.stable[key] + dt if valid else 0.0
            self.placement_valid[key] = valid and self.stable[key] >= self.stable_seconds

    def report(self):
        items = {key: {"destination": DESTINATIONS[key], "grasp_awarded": self.grasp_awarded[key],
                 "placement_valid": self.placement_valid[key], "passed_open_mouth": self.passed_mouth[key],
                 "hold_seconds": self.hold[key], "stable_seconds": self.stable[key],
                 "base_points": 2 * int(self.grasp_awarded[key]) + 6 * int(self.placement_valid[key])}
                 for key in DESTINATIONS}
        return {"track": "life", "task": "garbage_identification_and_disposal",
                "base_score": sum(x["base_points"] for x in items.values())
                              + 3 * sum(self.lid_open_awarded.values()), "base_score_max": 30,
                "full_task_success": self.bin_fixed and all(self.lid_open_now.values())
                                     and all(self.grasp_awarded.values()) and all(self.placement_valid.values()),
                "lids": {key: {"opened_awarded": self.lid_open_awarded[key],
                                "open_now": self.lid_open_now[key], "closed_now": self.lid_closed_now[key],
                                "base_points": 3 * int(self.lid_open_awarded[key])} for key in self.lid_open_now},
                "items": items, "bins_fixed_throughout": self.bin_fixed,
                "elapsed_physics_seconds": self.elapsed_seconds,
                "rule_source": "2026 Zhuhai consultation draft life task-1; no autonomy/body/end-effector coefficients",
                "simulation_tolerances": {"stable_seconds": self.stable_seconds, "hold_seconds": 5.0,
                    "lid_open_degrees": self.open_angle * 180 / 3.141592653589793,
                    "linear_speed_m_s": .015, "angular_speed_rad_s": .15, "clearance_m": .01,
                    "bounds_tolerance_m": .001},
                "final_lid_closed_required": False,
                "not_verified": ["banana-peel deformation/fragmentation", "material damage",
                                 "official object models and initial fixture layout"]}
