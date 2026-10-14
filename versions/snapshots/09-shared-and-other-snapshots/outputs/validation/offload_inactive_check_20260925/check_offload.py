"""GPU check (no simulator): offload_inactive gives bit-identical samples and lower resident VRAM.
Reference = both action models resident; candidate = offload_inactive. Same seeds, same inputs (demo frames)."""
import gc, json, subprocess, sys, time
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT))
from harness.policy_server import WamPolicy
from harness.wam_behavior import FRAMES, HEAD_CAMERA, LOWDIM, stack_proprio_history
from harness.frame_cache import FramePack
MAIN = "outputs/training/stage3_contract_posonly_20260922/checkpoint_best.pt"
MANIP = "outputs/training/stage7b_prophistdiff_manip_20260924/checkpoint_best.pt"
T = json.loads((ROOT / "data/skill_vtt_task013/contract_targets.json").read_text())["contracts"]
ep, a = 2604, 1600
state_all = np.load(LOWDIM / f"episode_{ep}_state.npy", mmap_mode="r")
rgb = FramePack(FRAMES / HEAD_CAMERA, ep).rgb(a)
def smi():
    return int(subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.split()[0])
def run(offload):
    orch = {"vlm": "none", "frozen": True, "train_targets": "data/skill_vtt_task013/contract_targets.json", "manip_checkpoint": MANIP}
    if offload: orch["offload_inactive"] = True
    base = smi(); torch.cuda.reset_peak_memory_stats()
    peak = [base]
    pol = WamPolicy(MAIN, skill="move to", exec_horizon=32, num_steps=10, device="cuda", cuda_fraction=0.9, seed=0,
                    log_dir=Path("/tmp/offload_check_trace"), orchestrate=orch, samples=1)
    state = np.array(state_all[a], np.float32)
    _, px = pol._preprocess({"r::proprio": state, "r::zed_link::rgb": rgb})
    feats, _ = pol._encode(px)
    out, loaded = [], smi() - base
    for model, key in ((pol.model, "move to(car)"), (pol.manip, "open lid(car)"), (pol.model, "move to(car)"), (pol.manip, "open lid(car)")):
        cond = torch.tensor(T[key]["vector"], dtype=torch.float32)[None]
        st = state if model.proprio_lags == [0] else stack_proprio_history(state_all, a, model.proprio_lags)
        pol._ensure_resident(model)
        pol.gen.manual_seed(1000 * ep + a)
        out.append(np.mean([pol._sample(feats, st, cond, model=model) for _ in range(8)], axis=0))
    res = {"offload": offload, "smi_after_load_mib": loaded, "smi_end_mib": smi() - base,
           "torch_peak_alloc_mib": torch.cuda.max_memory_allocated() / 2**20, "torch_reserved_mib": torch.cuda.memory_reserved() / 2**20}
    pol.trace.close(); del pol, feats; gc.collect(); torch.cuda.empty_cache()
    return out, res
mode = sys.argv[1]   # "reference" | "offload" | "compare"; one config per process for clean VRAM numbers
D = Path(__file__).parent
if mode == "compare":
    ref, cand = np.load(D / "reference.npy"), np.load(D / "offload.npy")
    r0, r1 = (json.loads((D / f"{m}.json").read_text()) for m in ("reference", "offload"))
    same = [bool(np.array_equal(x, y)) for x, y in zip(ref, cand)]
    report = {"bit_identical_per_call": same, "reference_two_resident": r0, "offload_inactive": r1,
              "note": "GPU check without simulator, one process per config; not a robot result",
              "inputs": {"episode": ep, "anchor": a, "K": 8, "calls": "main, manip, main, manip"}}
    (D / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2)); assert all(same), "offload changed samples"; print("PASS")
else:
    out, res = run(mode == "offload")
    res["smi_process_peak_note"] = "smi_* are deltas vs. the pre-load baseline of this fresh process"
    np.save(D / f"{mode}.npy", np.stack(out)); (D / f"{mode}.json").write_text(json.dumps(res, indent=2) + "\n")
    print(mode, res)
