"""Low-dimensional predictability probe for open lid(car) upper-body chunks (CPU, train->validation).

Question: how much of the 32-step upper-body motion is predictable from (a) current joint positions only
(what the positions-only WAM sees besides the image), (b) + short proprio history, (c) + the non-deployable
oracle segment phase. If (b) >> (a), history/phase ambiguity is the executor bottleneck; if all ~ hold,
the low-dim state carries no motion information and vision/contact cues must supply it.
Diagnostic only: no image, no training of the WAM, no policy change.
"""
import json, sys
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from harness.wam_behavior import LOWDIM, DATA, ACTION_GROUPS, STATE_SLICES

UB = [ACTION_GROUPS[g] for g in ("trunk", "left_arm", "right_arm")]
POS = [STATE_SLICES[k] for k in ("trunk_qpos", "arm_left_qpos", "arm_right_qpos", "eef_left_pos", "eef_right_pos",
                                 "gripper_left_qpos", "gripper_right_qpos")]
H, LAGS = 32, (8, 16, 32, 64)
seg = json.loads((DATA / "skill_segments_task013.json").read_text())["episodes"]
split = json.loads((DATA / "split_task_013.json").read_text())

def rows(eps, stride):
    X0, XH, PH, Y = [], [], [], []
    for ep in eps:
        a = np.load(LOWDIM / f"episode_{ep}_action.npy"); s = np.load(LOWDIM / f"episode_{ep}_state.npy")
        for sg in seg[str(ep)]["segments"]:
            if sg["skill"] != "open lid": continue
            st, en = sg["start"], sg["end"]
            for t in range(st + max(LAGS), en - H, stride):
                cur = np.concatenate([s[t, lo:hi] for lo, hi in POS])
                qub = np.concatenate([s[t, lo:hi] for lo, hi in (STATE_SLICES["trunk_qpos"], STATE_SLICES["arm_left_qpos"], STATE_SLICES["arm_right_qpos"])])
                fut = np.concatenate([a[t:t + H, lo:hi] for lo, hi in UB], 1) - qub[None]
                hist = np.concatenate([cur - np.concatenate([s[t - L, lo:hi] for lo, hi in POS]) for L in LAGS])
                X0.append(cur); XH.append(hist); PH.append([(t - st) / (en - st)]); Y.append(fut.reshape(-1))
    return [np.asarray(v, np.float32) for v in (X0, XH, PH, Y)]

tr = rows(split["train_episode_indices"], 8); va = rows(split["validation_episode_indices"], 8)
print("train", tr[3].shape, "val", va[3].shape)

def fit_mlp(Xtr, Ytr, Xva, seed=0, epochs=60):
    torch.manual_seed(seed)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6; ym, ys = Ytr.mean(0), Ytr.std(0) + 1e-6
    xt, yt = torch.tensor((Xtr - mu) / sd), torch.tensor((Ytr - ym) / ys)
    net = torch.nn.Sequential(torch.nn.Linear(xt.shape[1], 512), torch.nn.GELU(), torch.nn.Linear(512, 512), torch.nn.GELU(),
                              torch.nn.Linear(512, yt.shape[1]))
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    for e in range(epochs):
        perm = torch.randperm(len(xt))
        for i in range(0, len(xt), 1024):
            b = perm[i:i + 1024]; loss = (net(xt[b]) - yt[b]).abs().mean()
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return net(torch.tensor((Xva - mu) / sd)).numpy() * ys + ym

def mae(P, Y):
    P, Y = P.reshape(len(Y), H, 18), Y.reshape(len(Y), H, 18)
    e = np.abs(P - Y)
    return {"trunk": float(e[..., 0:4].mean()), "left_arm": float(e[..., 4:11].mean()), "right_arm": float(e[..., 11:18].mean()),
            "upper_body_18d": float(e.mean())}

torch.set_num_threads(12)
res = {"hold(delta=0)": mae(np.zeros_like(va[3]), va[3]),
       "train_mean_delta": mae(np.broadcast_to(tr[3].mean(0), va[3].shape), va[3])}
feats = {"current_positions": (tr[0], va[0]), "current+history": (np.hstack(tr[:2]), np.hstack(va[:2])),
         "current+oracle_phase": (np.hstack([tr[0], tr[2]]), np.hstack([va[0], va[2]])),
         "current+history+oracle_phase": (np.hstack(tr[:3]), np.hstack(va[:3]))}
for k, (a, b) in feats.items():
    res[k] = mae(fit_mlp(a, tr[3], b), va[3]); print(k, res[k], flush=True)
out = ROOT / "outputs/validation/openlid_predictability_probe_20260924/report.json"
out.write_text(json.dumps({"horizon": H, "lags": LAGS, "n_train": len(tr[3]), "n_val": len(va[3]),
                           "note": "MLP probe, L1 loss, absolute rad over 32 steps; oracle_phase is NOT deployable", "mae": res}, indent=1))
print(json.dumps(res, indent=1))

# Copycat control: constant-velocity extrapolation from the same history (no learning). If the history MLP is
# not clearly better than this, "history helps" would just mean "keep doing what you did" (copycat risk).
def const_vel(X0, XH, lag):
    j = LAGS.index(lag); n = X0.shape[1]
    dq = XH[:, j * n:(j + 1) * n][:, :18] / lag           # rad/frame of trunk+arms (first 18 of POS)
    steps = np.arange(1, H + 1, dtype=np.float32)[None, :, None]
    return (dq[:, None, :] * steps).reshape(len(X0), -1)
for lag in LAGS:
    res[f"const_velocity_lag{lag}"] = mae(const_vel(va[0], va[1], lag), va[3])
    res[f"const_velocity_lag{lag}_damped0.5"] = mae(0.5 * const_vel(va[0], va[1], lag), va[3])
out.write_text(json.dumps({"horizon": H, "lags": LAGS, "n_train": len(tr[3]), "n_val": len(va[3]),
                           "note": "MLP probe, L1 loss, absolute rad over 32 steps; oracle_phase is NOT deployable", "mae": res}, indent=1))
print({k: round(v["upper_body_18d"], 5) for k, v in res.items()})
