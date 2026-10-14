#!/usr/bin/env python3
"""Minimal Isaac Sim 5.1 headless RTX probe: no OmniGibson, no scene assets, no policy.

Isolates Kit/driver renderer initialization from OmniGibson configuration. Writes
outputs/behavior/isaac_probe/probe.json. Exit 0 only when frames rendered.
"""
import json, os, sys, time
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "outputs/behavior/isaac_probe"
OUT.mkdir(parents=True, exist_ok=True)
mode = sys.argv[1] if len(sys.argv) > 1 else "default"
t0 = time.time()
from isaacsim import SimulationApp
cfg = {"headless": True, "width": 256, "height": 256, "multi_gpu": False}
if mode == "no_window_flags":
    cfg["extra_args"] = ["--no-window", "--/app/window/hideUi=1"]
app = SimulationApp(cfg)
import omni.kit.app
frames = 0
for _ in range(40):
    app.update(); frames += 1
info = {"mode": mode, "frames": frames, "seconds": round(time.time() - t0, 1), "kit": omni.kit.app.get_app().get_build_version()}
try:
    import omni.replicator.core as rep
    cam = rep.create.camera(position=(3, 3, 3), look_at=(0, 0, 0))
    rp = rep.create.render_product(cam, (256, 256))
    ann = rep.AnnotatorRegistry.get_annotator("rgb"); ann.attach(rp)
    for _ in range(5): app.update()
    import numpy as np
    rgb = ann.get_data()
    info["rgb_shape"] = list(np.asarray(rgb).shape)
except Exception as e:
    info["render_capture_error"] = f"{type(e).__name__}: {e}"
(OUT / "probe.json").write_text(json.dumps(info, indent=2) + "\n")
print(json.dumps(info), flush=True)
app.close()
