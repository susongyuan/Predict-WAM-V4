#!/usr/bin/env python3
"""Read-only CPU dataset/metric check: existing caches only, no model, training or downloads."""
import hashlib
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness.wam_behavior import BehaviorChunkDataset, DATA, collate, hold_baseline
from scripts.train_lilawam_behavior import checkpoint_score, validate


class Probe(torch.nn.Module):
    """Stochastic CPU stand-in solely for the validation RNG/metric check; not a policy."""
    def forward(self, batch):
        loss = torch.randn(()).square()
        return loss, {"loss_mse": loss, "loss_future_feat": torch.rand(())}

    def sample_actions(self, pixels, state, cond, steps, generator):
        held = hold_baseline(state)
        return held + 0.01 * torch.randn(held.shape, generator=generator), None, None


def main():
    torch.set_num_threads(2)
    cfg = OmegaConf.load(ROOT / "configs/lilawam_behavior_manipulation_train.yaml")
    stage3 = OmegaConf.load(ROOT / "configs/lilawam_behavior_contract_posonly_train.yaml")
    assert cfg.model == stage3.model and cfg.common == stage3.common and cfg.dataset.stats_path == stage3.dataset.stats_path
    assert cfg.common.action_chunk_size == 32 and cfg.common.action_dim == 23 and cfg.dataset.instance_vtt_prob == 0
    split_path = DATA / "split_task_013.json"
    splits = json.loads(split_path.read_text())
    train_ids, val_ids = set(splits["train_episode_indices"]), set(splits["validation_episode_indices"])
    assert len(train_ids) == 180 and len(val_ids) == 20 and train_ids.isdisjoint(val_ids)
    kwargs = dict(stride=int(cfg.dataset.frame_stride), chunk=int(cfg.common.action_chunk_size),
                  future_offset=int(cfg.dataset.future_frame_offset), image_size=int(cfg.dataset.image_size[0]),
                  cond_mode=cfg.dataset.cond_mode, instance_prob=0, seed=int(cfg.training.seed),
                  contract_targets=cfg.dataset.contract_targets, contract_targets_sha256=cfg.dataset.contract_targets_sha256,
                  skip_unsupported_contracts=bool(cfg.dataset.skip_unsupported_contracts))
    reports = {}
    for split, ids in (("train", train_ids), ("validation", val_ids)):
        ds = BehaviorChunkDataset(split, scope=cfg.dataset.scope, in_segment=cfg.dataset.in_segment, **kwargs)
        assert set(ds.episodes) == ids
        assert hashlib.sha256(split_path.read_bytes()).hexdigest() == ds.index["split_sha256"] == ds.skills.raw["split_sha256"]
        packed = np.asarray(ds.anchors)
        moving_count = 0
        for ep in ds.episodes:
            lo, hi = np.searchsorted(packed[:, 0], [ep, ep + 1])
            anchors = packed[lo:hi, 1]
            assert len(anchors) and (anchors % ds.stride == 0).all() and ds.future_offset % ds.stride == 0
            expected = []
            for i, seg in enumerate(ds.skills.episodes[ep]["segments"]):
                kept = anchors[(anchors >= seg["fill_start"]) & (anchors < seg["fill_end"])]
                start, end = max(seg["start"], seg["fill_start"]), min(seg["end"], seg["fill_end"])
                horizon = max(ds.chunk - 1, ds.future_offset)
                if seg["skill"] != "move to" and (ep, i) in ds._contract_vectors:
                    expected.extend(range((start + ds.stride - 1) // ds.stride * ds.stride, end - horizon, ds.stride))
                if len(kept):
                    assert seg["skill"] != "move to"
                    # Continuous [start,end) contains all 32 actions AND the future, not gap-filled padding.
                    assert (kept >= start).all() and (kept + horizon < end).all()
                    np.testing.assert_array_equal(ds.condition(ep, int(kept[0]))[0].numpy(), ds._contract_vectors[ep, i])
            np.testing.assert_array_equal(anchors, expected)  # no unexplained exclusions, including same-skill boundaries
            action, state = ds.lowdim(ep)
            length = ds.index["episodes"][str(ep)]["length"]
            assert action.shape == (length, 23) and state.shape == (length, 61)
            prefix = np.r_[0, np.cumsum(np.any(np.abs(action[:, :3]) > 1e-6, axis=1))]
            moving = prefix[anchors + ds.chunk] - prefix[anchors] > 0
            assert moving.any(), (split, ep, "no retained local base motion")
            moving_count += int(moving.sum())
            if ep == ds.episodes[0]:  # Real __getitem__, including cached JPEG decode; no fixture/frame replacement.
                offset = int(np.flatnonzero(moving)[0]); a = int(anchors[offset])
                item = ds[int(lo) + offset]
                np.testing.assert_array_equal(item["action_sequence"].numpy(), action[a:a + 32])
                assert item["action_sequence"].shape == (32, 23) and torch.any(item["action_sequence"][:, :3].abs() > 1e-6)
            ds._lowdim.clear()  # Keep this check's mmap working set bounded to one episode.
        report = ds.anchor_filter
        assert report["excluded_anchors"] == report["original_anchors"] - len(ds) == sum(report["excluded_by_reason"].values())
        for group in ("by_skill", "by_contract"):
            for field in ("original_anchors", "retained_anchors", "excluded_anchors"):
                assert sum(row[field] for row in report[group].values()) == report[field]
        reports[split] = {"episodes": len(ds.episodes), "anchors_with_nonzero_base": moving_count,
                          "filter": report, "unsupported_segments": ds.contract_filter["unsupported_segments"]}

    # Default all must still be the old episode-tail-only grid, not the new manipulation filter.
    ep = ds.episodes[0]
    legacy = BehaviorChunkDataset("validation", episodes=[ep])
    assert legacy.scope == "all" and not legacy.in_segment
    assert legacy.anchors == [(ep, a) for a in range(0, legacy.index["episodes"][str(ep)]["length"] - 32, 2)]
    for options in ({"scope": "unknown"}, {"scope": "manipulation", "episodes": []}):
        try:
            BehaviorChunkDataset("train", cond_mode="none", **options)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Invalid/empty dataset accepted: {options}")

    # One mocked validation check: fixed noise, caller RNG untouched, group counts, dimension-weighted metric.
    first_cls = ds.skills.segment(*ds.anchors[0])["class"]
    other = next(i for i, anchor in enumerate(ds.anchors) if ds.skills.segment(*anchor)["class"] != first_cls)
    loader = DataLoader(Subset(ds, [0, 0, other]), batch_size=2, collate_fn=collate, num_workers=0)
    probe, results = Probe(), []
    for _ in range(2):
        torch.rand(7)
        before = torch.get_rng_state().clone()
        results.append(validate(probe, None, loader, "cpu", 2, gen_seed=123))
        assert torch.equal(before, torch.get_rng_state()) and probe.training
    assert results[0] == results[1]
    assert sorted(r["val_samples"] for r in results[0]["val_by_skill"].values()) == [1, 2]
    for key in ("val_mae", "val_hold_mae"):
        for group, value in results[0][key].items():
            np.testing.assert_allclose(value, sum(r[key][group] * r["val_samples"] for r in results[0]["val_by_skill"].values()) / 3)
    oracle = {"val_loss_mse": 9.0, "val_mae": {"trunk": 1.0, "left_arm": 2.0, "right_arm": 4.0}}
    assert checkpoint_score(oracle) == 9.0 and checkpoint_score(oracle, "upper_body_mae") == (4 + 14 + 28) / 18
    assert not torch.cuda.is_initialized()
    print(json.dumps({"passed": True, "cpu_only": True, "model_loaded": False, "written_files": [], "splits": reports,
                      "default_all_compatible": True, "validation_probe_passed": True,
                      "interpretation": "Data/interface check only; no training or policy evaluation."}, indent=2))


if __name__ == "__main__":
    main()
