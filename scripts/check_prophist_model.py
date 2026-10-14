"""CPU check for the proprio-history WAM contract (stage7). Not a capability test.

1. lags=[0] dataset items are identical to the pre-change code (legacy path untouched).
2. history dataset: state is (5,61) = frames t-[0,8,16,32,64] clamped at 0; state[0] == legacy state.
3. stage3 checkpoint loads into a proprio_len=5 model (only the sin-cos pos-emb buffer is extended; row 0 equal).
4. forward loss finite, gradient reaches proprio_proj; sample_actions returns (1,32,23) with (1,5,61) state.
"""
import importlib.util, json, sys
from pathlib import Path
import numpy as np, torch
from omegaconf import OmegaConf
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import harness.wam_behavior as W

out = {}
spec = importlib.util.spec_from_file_location("wam_old", str(ROOT / "harness/_wam_behavior_pre_prophist.py"))
old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
kw = dict(stride=2, cond_mode="contract_proto", instance_prob=0.0, seed=0, episodes=[2604, 2605],
          skip_unsupported_contracts=True, scope="manipulation", in_segment=True)
new0 = W.BehaviorChunkDataset("validation", **kw); old0 = old.BehaviorChunkDataset("validation", **kw)
newh = W.BehaviorChunkDataset("validation", proprio_lags=[0, 8, 16, 32, 64], **kw)
assert len(new0) == len(old0) == len(newh)
idx = [0, 1, len(new0) // 2, len(new0) - 1]
for i in idx:
    a, b, h = new0[i], old0[i], newh[i]
    for k in a:
        va, vb = a[k], b[k]
        assert (torch.equal(va, vb) if isinstance(va, torch.Tensor) else va == vb), k
    assert h["state"].shape == (5, 61) and torch.equal(h["state"][0:1], a["state"])
    ep, t = h["episode"], h["anchor"]
    s = np.load(W.LOWDIM / f"episode_{ep}_state.npy")
    for j, lag in enumerate([0, 8, 16, 32, 64]):
        assert np.array_equal(h["state"][j].numpy(), s[max(t - lag, 0)].astype(np.float32))
out["legacy_items_identical"] = True; out["history_items_ok"] = len(idx)
hs = W.stack_proprio_history(np.arange(10 * 61, dtype=np.float32).reshape(10, 61), 3, [0, 8])
assert np.array_equal(hs[1], np.arange(61, dtype=np.float32))   # clamped to frame 0

cfg = OmegaConf.load(ROOT / "configs/lilawam_behavior_prophistdiff_manip_train.yaml")
assert W.model_proprio_lags(cfg) == [0, 8, 16, 32, 64]
wrapper, _ = W.build_models(cfg, torch.device("cpu"), dino_dtype=torch.float32)
ck = torch.load(ROOT / "outputs/training/stage3_contract_posonly_20260922/checkpoint_best.pt", map_location="cpu", weights_only=False)
missing = W.load_action_state(wrapper.action_model, ck["model_state_dict"])
pe = wrapper.action_model.proprio_pos_emb
assert pe.shape[1] == 5 and torch.allclose(pe[:, :1], ck["model_state_dict"]["proprio_pos_emb"].float())
for k, v in ck["model_state_dict"].items():
    if k != "proprio_pos_emb":
        assert torch.equal(wrapper.action_model.state_dict()[k], v), k
out["stage3_load_missing"] = missing; out["shared_weights_identical"] = True
batch = W.collate([newh[i] for i in idx[:2]])
st = batch["state"][:2]
norm = wrapper.normalize_state(st)
std = torch.tensor(np.asarray([json.loads((ROOT / cfg.model.proprio_history_stats).read_text())["diff_std"][str(L)]
                               for L in (8, 16, 32, 64)], np.float32))
expect = ((st[:, :1] - st[:, 1:]) / std[None]).clamp(-5, 5) * wrapper.proprio_keep
assert torch.allclose(norm[:, 1:], expect, atol=1e-5)
assert torch.allclose(norm[:, :1], wrapper.normalize_state(st[:, :1]))   # token 0 = legacy normalization
assert float(norm[:, 1:].abs().max()) <= 5.0
out["diff_token_abs_mean"] = float(norm[:, 1:].abs().mean()); out["cur_token_abs_mean"] = float(norm[:, :1].abs().mean())
wrapper.train()
loss, info = wrapper.forward(batch)[:2] if isinstance(wrapper.forward(batch), tuple) else (wrapper.forward(batch), {})
loss = loss if torch.is_tensor(loss) else loss["loss"]
loss.backward()
g = wrapper.action_model.proprio_proj.weight.grad
assert torch.isfinite(loss) and g is not None and g.abs().sum() > 0
out["loss"] = float(loss); out["proprio_grad_norm"] = float(g.norm())
wrapper.eval()
with torch.no_grad():
    act, _, _ = wrapper.sample_actions(batch["pixel_values"][:1], batch["state"][:1], batch["task_cond"][:1], num_steps=2,
                                       generator=torch.Generator().manual_seed(0))
assert act.shape == (1, 32, 23) and torch.isfinite(act).all()
out["sample_shape"] = list(act.shape); out["status"] = "PASS (CPU contract check, not capability)"
(ROOT / "outputs/validation/prophist_stage7_prep_20260924/cpu_check_stage7b.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
