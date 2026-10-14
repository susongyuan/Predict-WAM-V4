#!/usr/bin/env python3
"""Download gated facebook/dinov3-vitl16-pretrain-lvd1689m at a pinned revision, verify SHA-256, write provenance."""
import hashlib, json, os, sys, time
from pathlib import Path
from huggingface_hub import hf_hub_download, HfApi

ROOT = Path(__file__).resolve().parents[1]
REPO = "facebook/dinov3-vitl16-pretrain-lvd1689m"
REV = "ea8dc2863c51be0a264bab82070e3e8836b02d51"
OUT = ROOT / "models/dinov3-vitl16-pretrain-lvd1689m"
FILES = ["config.json", "preprocessor_config.json", "model.safetensors", "README.md", "LICENSE.md"]
token = os.environ.get("HF_TOKEN")
if not token:
    sys.exit("HF_TOKEN not set")
api = HfApi(token=token)
info = api.model_info(REPO, revision=REV, token=token, files_metadata=True)
remote = {s.rfilename: s for s in info.siblings}
prov = {"repo_id": REPO, "revision": REV, "gated": str(info.gated), "files": {}, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
for name in FILES:
    p = hf_hub_download(REPO, name, revision=REV, token=token, local_dir=str(OUT), cache_dir=None)
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    size = os.path.getsize(p)
    exp = remote[name].size
    ok = (exp is None) or (size == exp)
    lfs_sha = getattr(remote[name].lfs, "sha256", None) if remote[name].lfs else None
    sha_ok = (lfs_sha is None) or (lfs_sha == h.hexdigest())
    prov["files"][name] = {"bytes": size, "expected_bytes": exp, "sha256": h.hexdigest(), "lfs_sha256": lfs_sha, "size_ok": ok, "sha_ok": sha_ok}
    print(name, size, h.hexdigest(), "size_ok", ok, "sha_ok", sha_ok, flush=True)
    if not (ok and sha_ok):
        sys.exit(f"verification failed for {name}")
prov["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
prov["config"] = json.loads((OUT / "config.json").read_text())
(OUT / "DOWNLOAD_PROVENANCE.json").write_text(json.dumps(prov, indent=2))
print("DONE", flush=True)
