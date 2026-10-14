"""Train-only per-lag statistics for explicit proprio-history tokens: std over frames of s[t] - s[t-lag] (61D).

History token j (lag L_j > 0) fed to the WAM = clip((s_t - s_{t-L}) / std_L, -5, 5), velocity groups masked by the
checkpoint's proprio mask. Frames before the episode start are clamped to frame 0 (same rule as training/deployment).
Validation/test data are never read.
"""
import hashlib, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from harness.wam_behavior import LOWDIM, DATA, stack_proprio_history

LAGS = [0, 8, 16, 32, 64]
split = json.loads((DATA / "split_task_013.json").read_text())
diffs = {L: [] for L in LAGS[1:]}
for ep in split["train_episode_indices"]:
    s = np.load(LOWDIM / f"episode_{ep}_state.npy")
    for t in range(0, len(s), 4):
        h = stack_proprio_history(s, t, LAGS)
        for j, L in enumerate(LAGS[1:], 1):
            diffs[L].append(h[0] - h[j])
std = {}
for L, d in diffs.items():
    d = np.asarray(d, np.float64); sd = d.std(0)
    std[str(L)] = np.where(sd < 1e-6, 1.0, sd).tolist()
out = {"lags": LAGS, "rule": "token_j = clip((s_t - s_{t-lag_j}) / std[lag_j], -5, 5); clamp t-lag to frame 0",
       "split": "train only", "train_episodes": len(split["train_episode_indices"]), "frame_stride": 4,
       "n_frames": len(diffs[LAGS[1]]), "diff_std": std,
       "split_sha256": hashlib.sha256((DATA / "split_task_013.json").read_bytes()).hexdigest()}
p = DATA / "train_stats_task013_prophist.json"
p.write_text(json.dumps(out))
print(p, out["n_frames"], {L: np.round(np.array(v)[53:57], 4).tolist() for L, v in std.items()})
