#!/usr/bin/env python3
"""Offline wiring diagnostic for the UNVERIFIED patch END appearance candidate.

Uses existing demonstration frames and the deployment encoder; it does not test task success.
A high score must remain verified=False/appearance_candidate and cannot advance a contract
or promote memory. Independent physical labels/calibration are not supplied by this script.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness.frame_cache import FramePack  # noqa: E402
from harness.policy_server import WamPolicy  # noqa: E402
from harness.wam_behavior import FRAMES, HEAD_CAMERA, LOWDIM, SkillTable  # noqa: E402

OUT = ROOT / "outputs/validation/postcondition_gate_runtime_20260923"
CKPT = "outputs/training/stage3_contract_posonly_20260922/checkpoint_best.pt"


def main():
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    table = SkillTable()
    index = json.loads((LOWDIM / "index.json").read_text())
    eps = [e for e in sorted(table.episodes) if index["episodes"][str(e)]["split"] == "validation"][:6]
    orch = {"vlm": "none", "answer": "none", "ask_budget": 2, "max_attempts": 2, "frozen": True, "precondition": True,
            "train_targets": "data/skill_vtt_task013/contract_targets.json", "arrival_gate": "patch", "postcondition_gate": "patch",
            "base_control": "wam_whole_body"}
    policy = WamPolicy(CKPT, skill="move to", exec_horizon=32, num_steps=10, device="cuda", cuda_fraction=0.3, seed=0,
                       log_dir=OUT, orchestrate=orch, samples=1)
    thr = policy.post_gate.threshold("open lid(car)")
    rows = []
    for ep in eps:
        seg = next(s for s in table.episodes[ep]["segments"] if s["skill"] == "open lid")
        pack = FramePack(FRAMES / HEAD_CAMERA, ep)
        st = np.load(LOWDIM / f"episode_{ep}_state.npy", mmap_mode="r")
        lo, hi = seg["start"], seg["end"]
        fr = {"start": (lo // 2) * 2, "mid": ((lo + (hi - lo) // 2) // 2) * 2, "end": ((hi - 1) // 2) * 2}
        rec = {"episode": ep}
        for name, f in fr.items():
            _, px = policy._preprocess({"r::proprio": np.asarray(st[f], np.float32), "r::zed_link::rgb": pack.rgb(f)})
            policy._encode(px)
            rec[name] = round(policy.post_gate.score(policy._last_patch, "open lid(car)"), 4)
        rows.append(rec)
    # orchestrated wiring: navigation tick on a start frame, then force the open-lid contract and feed start/end frames
    ep = eps[0]
    seg = next(s for s in table.episodes[ep]["segments"] if s["skill"] == "open lid")
    pack = FramePack(FRAMES / HEAD_CAMERA, ep)
    st = np.load(LOWDIM / f"episode_{ep}_state.npy", mmap_mode="r")
    f0, f1 = (seg["start"] // 2) * 2, ((seg["end"] - 1) // 2) * 2
    obs0 = {"r::proprio": np.asarray(st[f0], np.float32), "r::zed_link::rgb": pack.rgb(f0)}
    obs1 = {"r::proprio": np.asarray(st[f1], np.float32), "r::zed_link::rgb": pack.rgb(f1)}
    policy.act(obs0)                       # contract 0 = move to(car): no postcondition evidence expected
    assert policy._last_post is None, "navigation contract must not carry postcondition evidence"
    policy.orch.index, policy.orch.contract = 1, None
    policy.act(obs0)                       # creates open lid(car) contract (evidence still for the previous contract)
    policy.act(obs0)                       # start frame: evidence computed, must not be satisfied
    e_start = dict(policy._last_post)
    for _ in range(3):
        policy.act(obs1)                   # end frame: evidence computed for open lid(car)
    e_end = dict(policy._last_post)
    ends = [e for e in policy.orch.realized if e.get("key") == "open lid(car)"]
    report = {"threshold": thr, "scores": rows, "evidence_start": e_start, "evidence_end": e_end,
              "open_lid_contract_ends": [{k: e[k] for k in ("success", "steps", "info")} for e in ends],
              "note": "offline runtime wiring check on demonstration frames; not task success"}
    (OUT / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    ok_scores = all(r["end"] >= thr for r in rows) and all(r["start"] < thr for r in rows)
    print(json.dumps(report, indent=1))
    print("SCORE_SEPARATION", "PASS" if ok_scores else "PARTIAL (see per-episode scores; gate recall is 0.65 by calibration)")
    assert e_start["satisfied"] is False and e_start["verifier_kind"] == "appearance_candidate"
    assert e_start["verified"] is False and e_end["verified"] is False and not ends
    assert e_end["key"] == "open lid(car)" and isinstance(e_end["satisfied"], bool)
    print("WIRING PASS: unverified appearance candidate is keyed to the active contract; no success advancement")


if __name__ == "__main__":
    main()
