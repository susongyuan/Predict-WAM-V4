#!/usr/bin/env python3
"""Download the ungated facebook/dinov2-with-registers-large encoder into models/ with SHA-256 provenance.

Substitute for the gated DINOv3 ViT-L/16 that LiLa-WAM used. Same hidden size (1024) and
4 register tokens; patch size 14 instead of 16, so image sides must be multiples of 14.
"""
import hashlib, json, os, sys, time
from pathlib import Path
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
REPO = "facebook/dinov2-with-registers-large"
REVISION = os.environ.get("DINOV2_REVISION", "main")
OUT = ROOT / "models/dinov2-with-registers-large"
FILES = ["config.json", "preprocessor_config.json", "model.safetensors", "README.md"]

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    record = {"repo_id": REPO, "requested_revision": REVISION, "files": {}, "license": "Apache-2.0 (DINOv2 with registers, Meta)",
              "role": "frozen vision encoder substitute for gated facebook/dinov3-vitl16-pretrain-lvd1689m"}
    for name in FILES:
        for attempt in range(5):
            try:
                path = hf_hub_download(repo_id=REPO, filename=name, revision=REVISION, local_dir=OUT, token=False)
                break
            except Exception as e:
                if attempt == 4:
                    raise
                print(f"retry {name}: {type(e).__name__}: {e}", flush=True)
                time.sleep(5 * (attempt + 1))
        record["files"][name] = {"bytes": os.path.getsize(path), "sha256": sha256(path)}
        print(f"{name}: {record['files'][name]}", flush=True)
    cfg = json.loads((OUT / "config.json").read_text())
    record["config"] = {k: cfg.get(k) for k in ("model_type", "hidden_size", "patch_size", "num_register_tokens", "num_hidden_layers", "image_size")}
    record["downloaded_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (OUT / "DOWNLOAD_PROVENANCE.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record["config"]))

if __name__ == "__main__":
    main()
