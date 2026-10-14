#!/usr/bin/env python3
"""CPU check of the actual contract-conditioned training dataset; no model/encoder is loaded."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness.wam_behavior import BehaviorChunkDataset  # noqa: E402


def main():
    cfg = OmegaConf.load(ROOT / "configs/lilawam_behavior_contract_train.yaml")
    old = OmegaConf.load(ROOT / "configs/lilawam_behavior_train.yaml")
    assert cfg.model == old.model and cfg.common == old.common
    assert cfg.training.max_steps == cfg.training.schedule_steps == 6000
    assert cfg.training.learning_rate == 4e-5 and cfg.dataset.instance_vtt_prob == 0.2
    target_path = ROOT / cfg.dataset.contract_targets
    targets = json.loads(target_path.read_text())
    assert hashlib.sha256(target_path.read_bytes()).hexdigest() == cfg.dataset.contract_targets_sha256
    kwargs = {"cond_mode": "contract_proto", "contract_targets": cfg.dataset.contract_targets,
              "contract_targets_sha256": cfg.dataset.contract_targets_sha256,
              "skip_unsupported_contracts": cfg.dataset.skip_unsupported_contracts}
    train = BehaviorChunkDataset("train", instance_prob=0.0, **kwargs)
    val = BehaviorChunkDataset("validation", instance_prob=1.0, **kwargs)
    for dataset in (train, val):
        report = dataset.contract_filter
        assert len(dataset) == report["original_anchors"] - report["skipped_anchors"]
        assert report["skipped_anchors"] == sum(row["anchors"] for row in report["unsupported_segments"])
        for row in report["unsupported_segments"]:
            seg = dataset.skills.episodes[row["episode"]]["segments"][row["segment"]]
            assert not any(ep == row["episode"] and seg["fill_start"] <= a < seg["fill_end"] for ep, a in dataset.anchors)

    initial, cls = train.condition(2600, 0)
    np.testing.assert_array_equal(initial.numpy(), targets["navigation"]["move to(car)"]["open lid(car)"]["vector"])
    assert cls == 0
    # The same destination has a distinct return-with-box phase and seven-class skill ID stays unchanged.
    segments = train.skills.episodes[2600]["segments"]
    i = next(i for i, s in enumerate(segments[:-1]) if s["skill"] == "move to"
             and segments[i + 1]["skill"] == "place in")
    returning, return_cls = train.condition(2600, segments[i]["fill_start"])
    np.testing.assert_array_equal(returning.numpy(), targets["navigation"]["move to(car)"]["place in(toy_box,car)"]["vector"])
    assert cls == return_cls and not np.array_equal(initial.numpy(), returning.numpy())

    # Validation cannot consume its own cached future VTT even with instance_prob=1.
    heldout, _ = val.condition(2604, 0)
    np.testing.assert_array_equal(initial.numpy(), heldout.numpy())
    assert not np.array_equal(heldout.numpy(), val.instances[val._inst_lookup[2604, 0]])
    train.instance_prob = 1.0
    own_instance, _ = train.condition(2600, 0)
    np.testing.assert_array_equal(own_instance.numpy(), train.instances[train._inst_lookup[2600, 0]])

    unsupported = val.contract_filter["unsupported_segments"][0]
    try:
        BehaviorChunkDataset("validation", episodes=[unsupported["episode"]], cond_mode="contract_proto")
    except ValueError as exc:
        assert "explicit skip required" in str(exc)
    else:
        raise AssertionError("Unsupported validation silently changed the dataset")
    seg = val.skills.episodes[unsupported["episode"]]["segments"][unsupported["segment"]]
    try:
        val.condition(unsupported["episode"], seg["fill_start"])
    except ValueError as exc:
        assert "no training target" in str(exc)
    else:
        raise AssertionError("Unsupported contract fell back to a class prototype")

    legacy = BehaviorChunkDataset("validation", episodes=[2604], cond_mode="skill_proto", max_anchors_per_episode=1)
    np.testing.assert_array_equal(legacy.condition(2604, 0)[0].numpy(), legacy.prototypes[0])
    result = {"passed": True, "cpu_only": True, "model_loaded": False,
              "contract_targets_sha256": train.contract_targets_sha256,
              "init_from_sha256": cfg.training.init_from_sha256,
              "train_filter": train.contract_filter, "validation_filter": val.contract_filter,
              "checks": ["unchanged architecture", "separate navigation phases", "validation uses train means",
                         "training instance mixture", "explicit unsupported exclusions", "no prototype fallback", "legacy skill mode"],
              "interpretation": "Training interface readiness only; no task success or claim about a need for SLAM."}
    output = ROOT / "outputs/validation/contract_training_check.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"passed": True, "report": str(output.relative_to(ROOT)),
                      "train_anchors": len(train), "validation_anchors": len(val)}))


if __name__ == "__main__":
    main()
