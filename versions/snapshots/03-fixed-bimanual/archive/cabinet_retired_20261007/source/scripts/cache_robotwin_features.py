#!/usr/bin/env python3
"""Plan/cache every frame of selected RoboTwin episodes, offline and single-process.

  .venvs/wam-fixed/bin/python -B scripts/cache_robotwin_features.py \
      --split-manifest outputs/validation/robotwin_dataset_20261004/split_manifest.json \
      --out data/robotwin_features --splits train val --plan-only

CLI: --split-manifest, --out required; --splits train val by default (heldout
ONLY when explicitly selected); --episodes optional IDs, stems or filenames;
--batch 16; --device cuda|cpu (cuda default); --plan-only never loads the encoder
or queries/initializes CUDA. Planning hashes selected HDF files and reports exact
logical final bytes (NPY headers + JSON included) and bytes requiring rebuilding.

The caller must hold its external gpu_lock and must not run training alongside
this process. Two CPU threads, RAM <=85%, global GPU memory <=15300 MiB guards.
No model download, simulator, image/frame cache or augmentation. Failure leaves
only incomplete temporary payloads; meta.complete is written last. New config
identities get new directories, never overwrite a previous configuration.
"""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_var] = "2"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import h5py
import numpy as np
import torch
from harness.robotwin_dataset import RoboTwinEpisode, decode_rgb, POLICY_VIEWS
from harness.wam_behavior import load_upstream
from harness.wam_v3 import pool_from_hidden
from harness.robotwin_training_data import (
    ENCODER_PATH, FEAT_LAYERS, FEAT_SHAPE, atomic_json, cache_identity, completed_meta,
    digest, episode_directory, episode_identity, json_bytes, load_manifest, npy_bytes,
    payload_specs, preprocess_rgb, sha256_file, validate_payload,
)


def resource_guard(device="cpu"):
    info = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    total, available = (int(info[k].split()[0]) for k in ("MemTotal", "MemAvailable"))
    used = 100 * (1 - available / total)
    if used >= 85:
        raise RuntimeError(f"RAM guard: {used:.1f}% >=85%")
    if device == "cuda":
        result = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                check=True, capture_output=True, text=True, timeout=10)
        if any(int(v.strip()) >= 15300 for v in result.stdout.splitlines()):
            raise RuntimeError("GPU memory guard: >=15300 MiB")


def build_plan(split_manifest, out, splits=("train", "val"), episodes=None):
    """Pure CPU metadata plan. Payload reuse validates identity, shape, dtype and size."""
    manifest = load_manifest(split_manifest)
    if not splits or any(s not in manifest["splits"] for s in splits) or len(set(splits)) != len(splits):
        raise ValueError("splits must be unique train/val/heldout names")
    names = [name for split in splits for name in manifest["splits"][split]]
    if episodes is not None:
        selected = {f"episode_{int(e):07d}.hdf5" if str(e).isdigit()
                    else (str(e) if str(e).endswith(".hdf5") else str(e) + ".hdf5") for e in episodes}
        if not selected or not selected.issubset(names):
            raise ValueError("--episodes must be nonempty and belong to --splits")
        names = [n for n in names if n in selected]
    identity = cache_identity(manifest)
    config_id = digest(identity)
    root = Path(out).resolve() / config_id
    config_path = root / "cache_config.json"
    if config_path.exists() and config_path.read_bytes() != json_bytes(identity):
        raise ValueError("existing configuration differs; refusing overwrite")
    entries = []
    for name in names:
        resource_guard()
        ep = episode_identity(Path(manifest["data_root"]) / name, config_id)
        sizes = {k: npy_bytes(shape, dtype) for k, (shape, dtype) in payload_specs(ep).items()}
        total = sum(sizes.values()) + len(json_bytes(completed_meta(ep)))
        try:
            validate_payload(root, ep, load_features=False)
            reuse = True
        except ValueError:
            reuse = False
        entries.append({"identity": ep, "payload_bytes": sizes, "final_bytes": total, "reuse": reuse})
    config_bytes = len(json_bytes(identity))
    summary = {"cache_root": str(root), "config_id": config_id, "splits": list(splits),
               "episodes": len(entries), "frames_per_view": sum(e["identity"]["length"] for e in entries),
               "final_logical_bytes": config_bytes + sum(e["final_bytes"] for e in entries),
               "rebuild_logical_bytes": (0 if config_path.exists() else config_bytes)
                 + sum(e["final_bytes"] for e in entries if not e["reuse"]),
               "reused_episodes": sum(e["reuse"] for e in entries),
               "size_scope": "selected episodes plus config; logical bytes, not filesystem blocks",
               "entries": entries}
    return identity, summary


def encode_episode(encoder, path, root, identity, batch, device):
    episode = RoboTwinEpisode(path)
    if len(episode) != identity["length"]:
        raise ValueError("episode changed after planning")
    directory = episode_directory(root, identity)
    directory.mkdir(parents=True, exist_ok=True)
    # Invalidate previously corrupted/incomplete cache before replacing its payload.
    atomic_json(directory / "meta.json", {"complete": False, "identity": identity})
    arrays = {k: np.lib.format.open_memmap(directory / f"{k}.tmp.npy", mode="w+", dtype=dtype,
                                         shape=shape, version=(1, 0))
              for k, (shape, dtype) in payload_specs(identity).items()}
    with h5py.File(path, "r") as file, torch.inference_mode():
        for view_index, view in enumerate(POLICY_VIEWS):
            colors = file[f"vision/{view}/colors"]
            for lo in range(0, len(episode), batch):
                resource_guard(device)
                hi = min(lo + batch, len(episode))
                pixels = np.stack([preprocess_rgb(decode_rgb(colors[t], episode.image_shape))
                                   for t in range(lo, hi)])
                px = torch.from_numpy(pixels).to(device=device, dtype=torch.bfloat16)
                hs = encoder(pixel_values=px, output_hidden_states=True, return_dict=True).hidden_states
                hidden = torch.stack([hs[layer] for layer in FEAT_LAYERS], dim=1)
                if tuple(hidden.shape[1:]) != FEAT_SHAPE[1:] or hidden.dtype != torch.bfloat16:
                    raise ValueError(f"unexpected encoder output {hidden.shape}/{hidden.dtype}")
                pooled = pool_from_hidden(hidden[:, -1], 4).reshape(hi - lo, 2048)
                if not torch.isfinite(hidden).all() or not torch.isfinite(pooled).all():
                    raise ValueError("nonfinite encoder output")
                arrays["feat"][lo:hi, view_index] = hidden.contiguous().view(torch.int16).cpu().numpy().view(np.uint16)
                arrays["pool"][lo:hi, view_index] = pooled.cpu().numpy()
                del hs, hidden, pooled, px, pixels
                for arr in arrays.values():
                    arr.flush()
                resource_guard(device)
    for arr in arrays.values():
        arr.flush()
    arrays.clear()
    del arr
    if sha256_file(path) != identity["hdf_sha256"]:
        raise ValueError("HDF changed during encoding")
    for name in payload_specs(identity):
        (directory / f"{name}.tmp.npy").replace(directory / f"{name}.npy")
    atomic_json(directory / "meta.json", completed_meta(identity))
    validate_payload(root, identity, load_features=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split-manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", nargs="+", choices=("train", "val", "heldout"), default=["train", "val"])
    ap.add_argument("--episodes", nargs="+")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    ap.add_argument("--plan-only", action="store_true")
    args = ap.parse_args()
    if args.batch <= 0:
        ap.error("--batch must be positive")
    cv2.setNumThreads(2)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    resource_guard()
    identity, plan = build_plan(args.split_manifest, args.out, args.splits, args.episodes)
    print(json_bytes(plan).decode(), flush=True)
    if args.plan_only:
        return 0
    root = Path(plan["cache_root"])
    ancestor = root
    while not ancestor.exists():
        ancestor = ancestor.parent
    if shutil.disk_usage(ancestor).free < plan["rebuild_logical_bytes"]:
        raise RuntimeError("insufficient free disk for planned payloads")
    atomic_json(root / "cache_config.json", identity)
    todo = [entry["identity"] for entry in plan["entries"] if not entry["reuse"]]
    if not todo:
        return 0
    # Verify actual local weights against recorded provenance, before using them.
    if sha256_file(ENCODER_PATH / "model.safetensors") != identity["encoder_sha256"]:
        raise ValueError("local encoder weights differ from DOWNLOAD_PROVENANCE")
    resource_guard(args.device)
    if args.device == "cuda":
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, 15300 * 2**20 / total))
    _, runner = load_upstream()
    encoder, hidden, registers, patch = runner.ModelFactory.create_vision_encoder(
        str(ENCODER_PATH), torch.bfloat16, torch.device(args.device))
    if (hidden, registers, patch) != (1024, 4, 16):
        raise ValueError("unexpected DINOv3 geometry")
    encoder.requires_grad_(False).eval()
    resource_guard(args.device)
    for ep in todo:
        encode_episode(encoder, Path(identity["data_root"]) / ep["episode"], root, ep, args.batch, args.device)
        print(json_bytes({"event": "complete", "episode": ep["episode"], "cache_root": str(root)}).decode(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
