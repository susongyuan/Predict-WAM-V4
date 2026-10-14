"""R1Pro forward/inverse kinematics in numpy (trunk 4 + arm 7 joints -> end-effector pose in the robot base frame).

Why (progress.md 2026-09-27 02:10): the WAM executor's arm drifts 7-10 cm from the demonstrated grasp even with the
base placed like the demo, so pick/place contracts get a perception + IK primitive. Everything here uses policy-
visible quantities only (joint positions from proprio); the URDF is the robot asset shipped with OmniGibson.

The proprio eef pose (`eef_{arm}_pos/quat`, robot.get_relative_eef_pose) is relative to the articulation root and the
eef link is not in the URDF, so a constant eef offset in the gripper link frame (and a constant root offset) is fitted
once on TRAIN demo states (`fit_offsets`) and stored in configs/r1pro_eef_offsets.json.

CPU self-test: `python3 -m harness.r1pro_kinematics` (FK vs proprio eef on train states + IK round trips).
"""
from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
URDF = ROOT / "data/omnigibson/omnigibson-robot-assets/models/r1pro/urdf/r1pro.urdf"
OFFSETS = ROOT / "configs/r1pro_eef_offsets.json"
TRUNK_Q, LARM_Q, RARM_Q = slice(53, 57), slice(3, 10), slice(28, 35)             # proprio qpos slices
EEF = {"left": (slice(17, 20), slice(20, 24)), "right": (slice(42, 45), slice(45, 49))}


def rpy_mat(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def axis_angle(axis, a):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def quat_to_mat(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def mat_to_quat(R):
    w = np.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    if w > 1e-6:
        return np.array([(R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w), w])
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = np.sqrt(max(0.0, 1 + R[i, i] - R[j, j] - R[k, k])) * 2
    q = np.zeros(4)
    q[i], q[j], q[k], q[3] = s / 4, (R[j, i] + R[i, j]) / s, (R[k, i] + R[i, k]) / s, (R[k, j] - R[j, k]) / s
    return q


def rot_err(Ra, Rb):
    """Rotation vector of Ra @ Rb^T (small-angle error used by the IK)."""
    E = Ra @ Rb.T
    ang = np.arccos(np.clip((np.trace(E) - 1) / 2, -1, 1))
    if ang < 1e-9:
        return np.zeros(3)
    v = np.array([E[2, 1] - E[1, 2], E[0, 2] - E[2, 0], E[1, 0] - E[0, 1]])
    return v * ang / (2 * np.sin(ang)) if ang < np.pi - 1e-6 else v


class R1ProKinematics:
    def __init__(self, urdf=URDF, offsets=OFFSETS):
        root = ET.parse(str(urdf)).getroot()
        self.joints = {}
        for j in root.findall("joint"):
            o = j.find("origin")
            xyz = np.array([float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()])
            rpy = np.array([float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()])
            ax = j.find("axis")
            lim = j.find("limit")
            self.joints[j.get("name")] = {
                "type": j.get("type"), "parent": j.find("parent").get("link"), "child": j.find("child").get("link"),
                "T": self._T(rpy_mat(*rpy), xyz),
                "axis": np.array([float(v) for v in ax.get("xyz").split()]) if ax is not None else np.array([0, 0, 1.0]),
                "lim": (float(lim.get("lower", -np.pi)), float(lim.get("upper", np.pi))) if lim is not None else (-np.pi, np.pi)}
        child2j = {v["child"]: k for k, v in self.joints.items()}
        self.chain = {}
        for arm in ("left", "right"):
            leaf, names = f"{arm}_gripper_link", []
            while leaf in child2j:
                names.append(child2j[leaf])
                leaf = self.joints[child2j[leaf]]["parent"]
            self.chain[arm] = list(reversed(names))
        self.movable = {arm: [n for n in self.chain[arm] if self.joints[n]["type"] != "fixed"] for arm in self.chain}
        for arm in self.movable:     # trunk 4 then arm 7
            assert len(self.movable[arm]) == 11, (arm, self.movable[arm])
        self.limits = {arm: np.array([self.joints[n]["lim"] for n in self.movable[arm]]) for arm in self.movable}
        off = json.loads(Path(offsets).read_text()) if Path(offsets).exists() else {}
        self.eef_offset = {arm: np.array(off.get(arm, {}).get("eef_xyz", [0, 0, 0]), float) for arm in ("left", "right")}
        self.eef_rot = {arm: np.array(off.get(arm, {}).get("eef_R", np.eye(3).tolist()), float) for arm in ("left", "right")}
        self.root_offset = np.array(off.get("root_xyz", [0, 0, 0]), float)

    @staticmethod
    def _T(R, p):
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R, p
        return T

    def fk(self, arm, q11, with_offset=True, jac=False):
        """q11 = trunk(4) + arm(7). Returns (p, R) of the eef in the proprio base frame; with jac also the 6x11 Jacobian."""
        T = np.eye(4)
        axes, origins = [], []
        k = 0
        for n in self.chain[arm]:
            j = self.joints[n]
            T = T @ j["T"]
            if j["type"] != "fixed":
                a = T[:3, :3] @ j["axis"]
                axes.append(a)
                origins.append(T[:3, 3].copy())
                T = T @ self._T(axis_angle(j["axis"], q11[k]), np.zeros(3))
                k += 1
        if with_offset:
            p = T[:3, 3] + T[:3, :3] @ self.eef_offset[arm] + self.root_offset
            R = T[:3, :3] @ self.eef_rot[arm]
        else:
            p, R = T[:3, 3], T[:3, :3]
        if not jac:
            return p, R
        J = np.zeros((6, 11))
        for i, (a, o) in enumerate(zip(axes, origins)):
            J[:3, i] = np.cross(a, p - self.root_offset - o) if with_offset else np.cross(a, p - o)
            J[3:, i] = a
        return p, R, J

    def ik(self, arm, q11_init, p_target, R_target=None, iters=200, damping=0.05, rot_weight=0.3, q_bias=None,
           bias_weight=0.02, tol_p=0.004, tol_r=0.05, move_trunk=False):
        """Damped least squares on the arm joints (trunk fixed unless move_trunk). Returns (q11, info)."""
        q = np.array(q11_init, float).copy()
        cols = list(range(11)) if move_trunk else list(range(4, 11))
        lo, hi = self.limits[arm][:, 0], self.limits[arm][:, 1]
        for it in range(iters):
            p, R, J = self.fk(arm, q, jac=True)
            ep = np.asarray(p_target) - p
            er = rot_err(R_target, R) if R_target is not None else np.zeros(3)
            if np.linalg.norm(ep) < tol_p and (R_target is None or np.linalg.norm(er) < tol_r):
                break
            e = np.r_[ep, rot_weight * er] if R_target is not None else ep
            Jc = J[:, cols] if R_target is not None else J[:3, cols]
            if R_target is not None:
                Jc = np.r_[Jc[:3], rot_weight * Jc[3:]]
            A = Jc @ Jc.T + damping ** 2 * np.eye(Jc.shape[0])
            dq = Jc.T @ np.linalg.solve(A, e)
            if q_bias is not None:      # null-space pull toward a demonstrated posture
                N = np.eye(len(cols)) - np.linalg.pinv(Jc) @ Jc
                dq = dq + bias_weight * N @ (np.asarray(q_bias)[cols] - q[cols])
            step = np.clip(dq, -0.2, 0.2)
            q[cols] = np.clip(q[cols] + step, lo[cols], hi[cols])
        p, R = self.fk(arm, q)
        info = {"iters": it + 1, "pos_err": float(np.linalg.norm(np.asarray(p_target) - p)),
                "rot_err": float(np.linalg.norm(rot_err(R_target, R))) if R_target is not None else None}
        return q, info


def q11(state, arm):
    return np.r_[state[TRUNK_Q], state[LARM_Q] if arm == "left" else state[RARM_Q]]


def fit_offsets(episodes, stride=25):
    """Least-squares fit of the constant eef offset (gripper-link frame) + root offset from proprio eef positions,
    and the constant eef rotation from the proprio eef quaternions. TRAIN episodes only."""
    kin = R1ProKinematics(offsets="/nonexistent")
    out = {}
    rows_A, rows_b = [], []
    Rs = {"left": [], "right": []}
    for ep in episodes:
        S = np.load(ROOT / f"data/lowdim_cache_task001/episode_{ep}_state.npy")[::stride]
        for s in S:
            for ai, arm in enumerate(("left", "right")):
                p, R = kin.fk(arm, q11(s, arm), with_offset=False)
                A = np.zeros((3, 9))
                A[:, 3 * ai:3 * ai + 3] = R            # eef offset of this arm (gripper-link frame)
                A[:, 6:9] = np.eye(3)                  # shared root offset
                rows_A.append(A)
                rows_b.append(s[EEF[arm][0]] - p)
                Rs[arm].append(R.T @ quat_to_mat(s[EEF[arm][1]]))
    x, *_ = np.linalg.lstsq(np.concatenate(rows_A), np.concatenate(rows_b), rcond=None)
    for ai, arm in enumerate(("left", "right")):
        M = np.mean(Rs[arm], 0)
        U, _, Vt = np.linalg.svd(M)
        out[arm] = {"eef_xyz": x[3 * ai:3 * ai + 3].tolist(), "eef_R": (U @ Vt).tolist()}
    out["root_xyz"] = x[6:9].tolist()
    return out


def _split():
    d = json.loads((ROOT / "data/behavior-1k/2026-challenge-demos/split_task_001.json").read_text())
    return sorted(int(e) for e in d["train_episode_indices"]), sorted(int(e) for e in d["validation_episode_indices"])


if __name__ == "__main__":
    import sys
    train, val = _split()
    if "--fit" in sys.argv:
        off = fit_offsets(train[:60])
        off["source"] = "fit_offsets on 60 TRAIN episodes (data/lowdim_cache_task001, stride 25); harness/r1pro_kinematics.py"
        OFFSETS.write_text(json.dumps(off, indent=1))
        print("wrote", OFFSETS, json.dumps(off)[:300])
    kin = R1ProKinematics()
    errs, rerr = {"left": [], "right": []}, {"left": [], "right": []}
    for ep in val:
        S = np.load(ROOT / f"data/lowdim_cache_task001/episode_{ep}_state.npy")[::50]
        for s in S:
            for arm in ("left", "right"):
                p, R = kin.fk(arm, q11(s, arm))
                errs[arm].append(np.linalg.norm(p - s[EEF[arm][0]]))
                rerr[arm].append(np.linalg.norm(rot_err(quat_to_mat(s[EEF[arm][1]]), R)))
    for arm in errs:
        e = np.array(errs[arm])
        print(f"FK vs proprio eef on VALIDATION ({arm}): pos median {np.median(e)*1000:.1f} mm p99 {np.percentile(e,99)*1000:.1f} mm; "
              f"rot median {np.degrees(np.median(rerr[arm])):.2f} deg (n={len(e)})")
    # IK round trips: random validation states -> perturbed start -> recover the original eef pose
    rng = np.random.default_rng(0)
    ok, n = 0, 0
    S = np.load(ROOT / f"data/lowdim_cache_task001/episode_{val[0]}_state.npy")
    for s in S[::200]:
        for arm in ("left", "right"):
            q = q11(s, arm)
            p, R = kin.fk(arm, q)
            q0 = q.copy()
            q0[4:] += rng.normal(0, 0.25, 7)
            qs, info = kin.ik(arm, q0, p, R)
            n += 1
            ok += info["pos_err"] < 0.005 and info["rot_err"] < 0.06
    print(f"IK round trips: {ok}/{n} within 5 mm / 3.4 deg")
    assert np.median(errs["left"]) < 0.01 and np.median(errs["right"]) < 0.01 and ok >= 0.9 * n
    print("PASS r1pro_kinematics self-test")
