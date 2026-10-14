"""Perception + IK manipulation primitives for Task 1 contracts (pick up can / pick up bin / place can in held bin).

Why (progress.md 2026-09-27 02:10): in the demo-prefix handoff the 3-view WAM drifts 7-10 cm from the demonstrated
grasp (closed-loop compounding of an arm prediction anchored on the current joint positions), so the left gripper
never closes, even with the base replayed from the demo. The primitives below execute the same contracts from
policy-visible inputs only: joint positions / eef poses from proprio (R1Pro FK/IK, harness/r1pro_kinematics.py) and
the target object's position in the robot base frame from onboard perception (head RGB-D) or, for placing, the held
bin's pose from the right-hand eef. Grasp geometry (eef height, orientation, trunk posture, place offset) is the
median of TRAIN demonstrations at the gripper command crossings
(outputs/validation/task1_docking_calibration_20260926/grasp_geometry_train.json).

Action layout (23-D, R1Pro eval): base 0:3 (held at 0), trunk 3:7, left arm 7:14, left gripper 14 (+1 open, -1 close),
right arm 15:22, right gripper 22. Arms/trunk are absolute joint position targets.

CPU self-test: `python3 -m harness.grasp_primitive` (IK feasibility over TRAIN-calibrated targets + phase sequencing).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from harness.r1pro_kinematics import EEF, LARM_Q, RARM_Q, TRUNK_Q, R1ProKinematics, quat_to_mat

ROOT = Path(__file__).resolve().parents[1]
GEOMETRY = ROOT / "outputs/validation/task1_docking_calibration_20260926/grasp_geometry_train.json"
JOINT_SPEED = 0.6          # rad/s max joint speed of the interpolated segments (30 Hz control)
DT = 1.0 / 30.0
CAN_XY_OFFSET = np.array([-0.009, 0.006])      # eef - can centre at grasp (demo replay ep389, 3 grasps; ~1 cm)
BIN_RIM = np.array([-0.017, -0.13])            # right-hand rim grasp point - bin centre, base frame (demo replay ep389)
PRE_DZ = 0.12              # pre-grasp / pre-place height above the grasp point
LIFT_Z = 0.36              # eef height after lifting a can (base frame, m)
CAN_GRASP_Z = 0.105        # eef height at the can grasp: TRAIN median 0.123 (p10 0.113) grips only the top ~1 cm; a
#                            2.8 cm xy error then slips (dev10 v4 attempt 5), so the fingers go ~2 cm deeper
BIN_CARRY_EEF = np.array([0.68, -0.50, 0.37])   # right eef (bin rim) while picking a can with the crouched trunk:
#                            TRAIN median (0.675, -0.298, 0.321) raised so the (tilted) bin clears the floor (dev10 v4:
#                            the bin bottom hit the floor and blocked trunk joint 1 at 1.505 of 1.619); moved outward
#                            (y -0.35 -> -0.50) and down (0.40 -> 0.37) after dev10 v6, where the bin held in front of
#                            the torso blocked the crouch (trunk joint 1 at 1.496) and pushed the left shoulder 0.48 rad
CLEAR_Z = 0.26             # transit height of the can-holding eef above the bin rim (can hangs ~0.19 m below the eef)


class Segment:
    """Joint-space linear interpolation from q0 to q1 (trunk 4 + left 7 + right 7) plus constant gripper commands."""

    def __init__(self, name, q0, q1, grip, min_steps=10, hold_steps=0):
        self.name, self.q0, self.q1, self.grip = name, np.asarray(q0, float), np.asarray(q1, float), grip
        n = int(np.ceil(np.abs(self.q1 - self.q0).max() / (JOINT_SPEED * DT))) if len(self.q0) else 0
        self.n = max(int(min_steps), n) + int(hold_steps)
        self.move_n = max(int(min_steps), n)

    def at(self, k):
        a = min(1.0, (k + 1) / self.move_n)
        a = 0.5 - 0.5 * np.cos(np.pi * a)          # smooth start/stop
        return self.q0 + a * (self.q1 - self.q0)


class Primitive:
    """One contract executed as a list of segments planned once at start (from the state at that step)."""

    def __init__(self, kind, state, target=None, geometry=None, kin=None, carry_posture=None, align_to=None):
        """align_to (optional, pick only): base-frame xy where the grasp eef point should end up (docking calibration
        `target_at="grasp"`). The base is first servoed on odometry until the planned grasp point is there (the object
        position is carried along by odometry), then the arm plan is made from the state at that moment."""
        self.kind = kind
        self.carry = carry_posture      # {"trunk", "rarm"} (docking_controller calibration nav_carrying_bin) or None
        self.kin = kin or R1ProKinematics()
        self.geo = geometry or json.loads(GEOMETRY.read_text())
        self.state0 = np.asarray(state, float)
        self.segments, self.k, self.i = [], 0, 0
        self.target = None if target is None else np.asarray(target, float)
        self.plan_info = {"kind": kind, "target_base": None if target is None else [round(float(v), 3) for v in target]}
        self.feasible = True
        self.align_to = None if align_to is None or kind == "place_in" else np.asarray(align_to, float)
        self.align = None
        if self.align_to is not None:
            off = CAN_XY_OFFSET if kind == "pick_can" else BIN_RIM
            self.align = {"pose": np.zeros(3), "steps": 0, "eef_offset": off, "hold": self._q18(self.state0)}
            self.plan_info["align_to"] = [round(float(v), 3) for v in self.align_to]
        else:
            self._plan(self.state0)

    def _plan(self, state):
        self.state0 = np.asarray(state, float)
        {"pick_can": self._plan_pick_can, "pick_bin": self._plan_pick_bin, "place_in": self._plan_place_in}[self.kind](self.target)

    # --- helpers -----------------------------------------------------------------------------------------------
    def _q18(self, s):
        return np.r_[s[TRUNK_Q], s[LARM_Q], s[RARM_Q]]

    def _ik(self, arm, trunk, arm_q, p, R, bias, move_trunk=False):
        """Returns (trunk, arm_q, ok, info); with move_trunk the trunk joins the solve (null-space bias to `trunk`)."""
        q11, info = self.kin.ik(arm, np.r_[trunk, arm_q], p, R, q_bias=np.r_[trunk, bias], iters=400, move_trunk=move_trunk)
        ok = info["pos_err"] < 0.01 and (info["rot_err"] is None or info["rot_err"] < 0.15)
        return q11[:4], q11[4:], ok, info

    OPEN_FINGER_SUM = 0.09   # m; open 0.100, holding a can 0.063-0.078, holding the bin rim 0.006-0.026 (TRAIN)

    def _grip_now(self):
        """Keep each gripper as it is: +1 only if it is (nearly) fully open. A threshold inside the holding range
        re-opened the hand holding the bin rim (finger sum 0.0246) and dropped the bin (dev10 v2 run, 2026-09-27)."""
        s = self.state0
        return ((1.0 if s[24:26].sum() > self.OPEN_FINGER_SUM else -1.0),
                (1.0 if s[49:51].sum() > self.OPEN_FINGER_SUM else -1.0))

    def _add(self, name, q1, gl, gr, min_steps=10, hold=0):
        q0 = self.segments[-1].q1 if self.segments else self._q18(self.state0)
        self.segments.append(Segment(name, q0, q1, (gl, gr), min_steps, hold))

    def _fail(self, why, info=None):
        self.feasible = False
        self.plan_info.update(infeasible=why, ik=info)

    # --- plans ---------------------------------------------------------------------------------------------------
    def _plan_pick_can(self, target):
        g = self.geo["can"]
        s = self.state0
        trunk = np.array(g["trunk_median"])
        R = np.array(g["eef_R_median"])
        grasp = np.r_[np.asarray(target[:2]) + CAN_XY_OFFSET, CAN_GRASP_Z]
        pre = grasp + [0, 0, PRE_DZ]
        bias = np.array(g["arm_q_median"])
        t_gr, q_gr, ok2, i2 = self._ik("left", trunk, bias, grasp, R, bias)
        t_pre, q_pre, ok1, i1 = self._ik("left", t_gr, q_gr, pre, R, q_gr, move_trunk=True)
        t_up, q_up, ok3, i3 = self._ik("left", t_gr, q_gr, np.r_[grasp[:2], LIFT_Z], R, q_gr, move_trunk=True)
        self.plan_info.update(grasp_eef=grasp.round(3).tolist(), ik_err_m=[round(i["pos_err"], 4) for i in (i1, i2, i3)])
        if not (ok1 and ok2 and ok3):
            return self._fail("ik", [i1, i2, i3])
        gl, gr = 1.0, self._grip_now()[1]
        # The right hand holds the bin: with the crouched grasp trunk the bin must be carried where the demos carry it
        # (right eef z 0.32 median) or its bottom hits the floor and blocks the crouch (prim_b run 2026-09-27: trunk
        # joint 1 stuck at 1.435 of 1.62, eef 11 cm too high). Otherwise the right arm keeps its measured posture.
        rq = s[RARM_Q]
        self.plan_info["right_arm"] = "hold"
        if gr < 0 and "rarm_median" in g:
            rq = np.array(g["rarm_median"])
            self.plan_info["right_arm"] = "demo_can_grasp_median"
            if "right_eef_R_median" in g:
                _, rq2, okr, ir = self._ik("right", t_gr, rq, BIN_CARRY_EEF, np.array(g["right_eef_R_median"]), rq)
                if okr:
                    rq, self.plan_info["right_arm"] = rq2, "bin_raised_ik"
        self._add("pre_grasp", np.r_[t_pre, q_pre, rq], gl, gr, 30)
        self._add("descend", np.r_[t_gr, q_gr, rq], gl, gr, 30, hold=5)
        self._add("close", np.r_[t_gr, q_gr, rq], -1.0, gr, 25)
        self._add("lift", np.r_[t_up, q_up, rq], -1.0, gr, 40, hold=10)

    def _plan_pick_bin(self, target):
        """target = bin centre (base frame); the right hand grasps the rim on the robot's right side (demo geometry)."""
        g = self.geo["bin"]
        s = self.state0
        trunk = np.array(g["trunk_median"])
        R = np.array(g["eef_R_median"])
        grasp = np.r_[np.asarray(target[:2]) + BIN_RIM, g["eef_z_median"]]
        pre = grasp + [0, 0, PRE_DZ]
        bias = np.array(g["arm_q_median"])
        t_gr, q_gr, ok2, i2 = self._ik("right", trunk, bias, grasp, R, bias)
        t_pre, q_pre, ok1, i1 = self._ik("right", t_gr, q_gr, pre, R, q_gr, move_trunk=True)
        t_up, q_up, ok3, i3 = self._ik("right", t_gr, q_gr, grasp + [0, 0, PRE_DZ], R, q_gr, move_trunk=True)
        self.plan_info.update(grasp_eef=grasp.round(3).tolist(), ik_err_m=[round(i["pos_err"], 4) for i in (i1, i2, i3)])
        if not (ok1 and ok2 and ok3):
            return self._fail("ik", [i1, i2, i3])
        lq = s[LARM_Q]
        gl = self._grip_now()[0]
        self._add("pre_grasp", np.r_[t_pre, lq, q_pre], gl, 1.0, 30)
        self._add("descend", np.r_[t_gr, lq, q_gr], gl, 1.0, 30, hold=5)
        self._add("close", np.r_[t_gr, lq, q_gr], gl, -1.0, 25)
        self._add("lift", np.r_[t_up, lq, q_up], gl, -1.0, 30)
        if self.carry is not None:     # demo carrying posture (trunk + right arm medians while navigating with the bin)
            self._add("carry", np.r_[self.carry["trunk"], lq, self.carry["rarm"]], gl, -1.0, 45, hold=10)

    def _plan_place_in(self, target=None):
        """Left hand (holding a can from the top) drops it into the bin held by the right hand (demo release geometry).

        Path (prim_b4 run 2026-09-27: a direct joint move swept the can into the bin wall and released it 14 cm off):
        1) trunk + right arm to the demo release posture (the bin comes down; left joints fixed), 2) left eef straight up
        to rim + CLEAR_Z (the can hangs ~0.19 m below the eef), 3) across to above the bin at that height, 4) down to the
        demo release point, 5) open, 6) back up."""
        g = self.geo["place"]
        s = self.state0
        trunk = np.array(g["trunk_median"])
        rq = np.array(g["rarm_median"])                  # the right hand brings the bin to the demo release posture
        pr, Rr = self.kin.fk("right", np.r_[trunk, rq])
        rel = np.array(g["left_minus_right_in_right_eef_frame_median"])
        place = pr + Rr @ rel
        z_clear = float(pr[2]) + CLEAR_Z                  # right eef = rim grasp point
        lq0 = s[LARM_Q]
        pl0, _ = self.kin.fk("left", np.r_[trunk, lq0])
        R = np.array(g["left_eef_R_median"])
        bias = np.array(g["larm_median"])
        _, q_up, ok1, i1 = self._ik("left", trunk, lq0, np.r_[pl0[:2], max(z_clear, pl0[2])], None, lq0)
        _, q_over, ok2, i2 = self._ik("left", trunk, q_up, np.r_[place[:2], z_clear], None, bias)
        _, q_pl, ok3, i3 = self._ik("left", trunk, q_over, place, R, q_over)
        if not ok3:
            _, q_pl, ok3, i3 = self._ik("left", trunk, q_over, place, None, q_over)
        self.plan_info.update(place_eef=place.round(3).tolist(), clear_z=round(z_clear, 3),
                              ik_err_m=[round(i["pos_err"], 4) for i in (i1, i2, i3)])
        if not (ok1 and ok2 and ok3):
            return self._fail("ik", [i1, i2, i3])
        self._add("bin_to_release_posture", np.r_[trunk, lq0, rq], -1.0, -1.0, 45)
        self._add("raise", np.r_[trunk, q_up, rq], -1.0, -1.0, 25)
        self._add("over_bin", np.r_[trunk, q_over, rq], -1.0, -1.0, 30)
        self._add("lower", np.r_[trunk, q_pl, rq], -1.0, -1.0, 25, hold=5)
        self._add("release", np.r_[trunk, q_pl, rq], 1.0, -1.0, 25)
        self._add("retract", np.r_[trunk, q_over, rq], 1.0, -1.0, 25, hold=5)

    # --- execution -----------------------------------------------------------------------------------------------
    @property
    def done(self):
        return self.align is None and self.i >= len(self.segments)

    @property
    def phase(self):
        return "align" if self.align is not None else ("done" if self.done else self.segments[self.i].name)

    ALIGN_TOL, ALIGN_MAX_STEPS, ALIGN_VMAX, ALIGN_K = 0.015, 240, 0.12, 1.5

    def _align_step(self, state):
        """Holonomic base servo on odometry (measured local base velocity) toward the grasp-time relative pose."""
        from harness.docking_controller import BASE_SCALE, Odometry
        al = self.align
        if al["steps"] > 0:
            odo = Odometry()
            odo.pose = al["pose"]
            odo.update(np.asarray(state, float)[:3])
            al["pose"] = odo.pose
        x, y, th = al["pose"]
        c, s_ = np.cos(-th), np.sin(-th)
        d = self.target[:2] - np.array([x, y])
        obj = np.array([c * d[0] - s_ * d[1], s_ * d[0] + c * d[1]])       # object centre in the current base frame
        err = obj + al["eef_offset"] - self.align_to
        al["steps"] += 1
        if np.linalg.norm(err) < self.ALIGN_TOL or al["steps"] > self.ALIGN_MAX_STEPS:
            self.plan_info.update(align_steps=al["steps"], align_residual_m=round(float(np.linalg.norm(err)), 4),
                                  target_base_after_align=[round(float(v), 3) for v in obj])
            self.target = np.r_[obj, self.target[2:]]
            self.align = None
            self._plan(state)
            return None
        v = np.clip(self.ALIGN_K * err, -self.ALIGN_VMAX, self.ALIGN_VMAX)
        a = np.zeros(23, np.float32)
        a[0:2] = v / BASE_SCALE[:2]
        q = al["hold"]
        a[3:7], a[7:14], a[15:22] = q[:4], q[4:11], q[11:18]
        g = self._grip_now()
        a[14], a[22] = (1.0 if self.kind == "pick_can" else g[0]), (1.0 if self.kind == "pick_bin" else g[1])
        return a

    def step(self, state=None):
        """Next 23-D action. `state` (61-D proprio) is required while aligning. After the last segment it keeps
        commanding the final targets (base zero)."""
        if self.align is not None:
            a = self._align_step(state)
            if a is not None:
                return a
        if not self.feasible:
            return self._hold_action(self.state0 if state is None else state)
        seg = self.segments[min(self.i, len(self.segments) - 1)]
        q = seg.at(self.k) if not self.done else seg.q1
        a = np.zeros(23, np.float32)
        a[3:7], a[7:14], a[15:22] = q[:4], q[4:11], q[11:18]
        a[14], a[22] = seg.grip
        if not self.done:
            self.k += 1
            if self.k >= seg.n:
                self.i, self.k = self.i + 1, 0
        return a

    def _hold_action(self, state):
        s = np.asarray(state, float)
        a = np.zeros(23, np.float32)
        a[3:7], a[7:14], a[15:22] = s[TRUNK_Q], s[LARM_Q], s[RARM_Q]
        a[14], a[22] = self._grip_now()
        return a

    @property
    def total_steps(self):
        return int(sum(s.n for s in self.segments))


if __name__ == "__main__":
    kin = R1ProKinematics()
    geo = json.loads(GEOMETRY.read_text())
    split = json.loads((ROOT / "data/behavior-1k/2026-challenge-demos/split_task_001.json").read_text())
    from harness.docking_controller import load_calibration
    cal = load_calibration(target_at="grasp")
    # IK feasibility at docked poses: the dock targets are eef-at-grasp positions; convert back to object centres.
    rng = np.random.default_rng(0)
    res = {}
    for ep in split["validation_episode_indices"][:10]:
        S = np.load(ROOT / f"data/lowdim_cache_task001/episode_{ep}_state.npy")
        s = S[len(S) // 3]
        for kind, lab in (("pick_can", "can_of_soda"), ("pick_bin", "trash_can")):
            d = cal["dock"][lab]
            for _ in range(3):
                eef_xy = rng.uniform(d["p10"], d["p90"])
                centre = eef_xy - (CAN_XY_OFFSET if kind == "pick_can" else BIN_RIM)
                p = Primitive(kind, s, np.r_[centre, 0.06], geo, kin, carry_posture=cal["posture"]["nav_carrying_bin"])
                res.setdefault(kind, []).append(p.feasible)
        A = np.load(ROOT / f"data/lowdim_cache_task001/episode_{ep}_action.npy")
        rel = np.where((A[:-1, 14] < 0) & (A[1:, 14] >= 0))[0] + 1        # demo releases (place in)
        for t in rel[:3]:
            p = Primitive("place_in", S[max(0, t - 60)], None, geo, kin)  # plan from 2 s before the release
            res.setdefault("place_in", []).append(p.feasible)
    for k, v in res.items():
        print(f"{k}: IK feasible {sum(v)}/{len(v)} at docking targets sampled in the TRAIN p10-p90 box (validation states)")
    p = Primitive("pick_can", s, np.r_[cal["dock"]["can_of_soda"]["target"] - CAN_XY_OFFSET, 0.06], geo, kin)
    acts = [p.step() for _ in range(p.total_steps + 5)]
    assert p.done and all(np.isfinite(a).all() for a in acts)
    phases = [s.name for s in p.segments]
    assert phases == ["pre_grasp", "descend", "close", "lift"], phases
    # the planned final targets reach the planned eef poses
    qf = p.segments[1].q1
    pe, _ = kin.fk("left", np.r_[qf[:4], qf[4:11]])
    assert np.linalg.norm(pe - np.array(p.plan_info["grasp_eef"])) < 0.01
    assert all(sum(v) >= 0.8 * len(v) for v in res.values()), res
    print(f"pick_can plan {p.total_steps} steps: {p.plan_info}")
    # align phase on a kinematic base (measured velocity = commanded, one-step lag): object 11 cm too far, 3 cm left
    from harness.docking_controller import BASE_SCALE
    dock = cal["dock"]["can_of_soda"]["target"]
    s1 = s.copy()
    s1[:3] = 0
    p = Primitive("pick_can", s1, np.r_[dock - CAN_XY_OFFSET + [0.11, 0.03], 0.06], geo, kin, align_to=dock)
    n = 0
    while p.phase == "align" and n < 400:
        a = p.step(s1)
        s1 = s1.copy()
        s1[:3] = a[:3] * BASE_SCALE
        n += 1
    assert p.phase == "pre_grasp" and p.feasible, p.plan_info
    assert abs(p.plan_info["target_base_after_align"][0] - (dock - CAN_XY_OFFSET)[0]) < 0.03, p.plan_info
    print(f"align: {p.plan_info['align_steps']} steps, residual {p.plan_info['align_residual_m']} m -> {p.plan_info['target_base_after_align']}")
    sh = s.copy()
    sh[49:51] = 0.0123                  # right hand on the bin rim (finger sum 0.0246)
    p = Primitive("pick_can", sh, np.r_[cal["dock"]["can_of_soda"]["target"] - CAN_XY_OFFSET, 0.06], geo, kin)
    assert all(seg.grip[1] == -1.0 for seg in p.segments) and p.plan_info["right_arm"] in ("demo_can_grasp_median", "bin_raised_ik")
    print("PASS grasp_primitive self-test")
