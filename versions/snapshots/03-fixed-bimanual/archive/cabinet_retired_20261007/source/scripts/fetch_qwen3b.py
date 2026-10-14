#!/usr/bin/env python3
"""Download Qwen/Qwen2.5-VL-3B-Instruct (pinned revision) into the shared HF cache for a load-on-demand VLM."""
import json, sys, time
from huggingface_hub import snapshot_download
REV = "66285546d2b821cf421d4f5eb2576359d3770cd3"
t = time.time()
path = snapshot_download("Qwen/Qwen2.5-VL-3B-Instruct", revision=REV, max_workers=4)
print(json.dumps({"path": path, "revision": REV, "seconds": round(time.time() - t, 1)}))
