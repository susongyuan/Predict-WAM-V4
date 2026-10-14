"""Isaac Sim headless probe variant: same as scripts/isaac_headless_probe.py but takes extra Kit args
from env PROBE_EXTRA_ARGS (space separated) and tags output by PROBE_TAG. Works for 5.1 and 6.1."""
import json, os, sys, time
from pathlib import Path
OUT = Path("/home/user/Robotic/project2/outputs/behavior/isaac_probe")
tag = os.environ.get("PROBE_TAG", "variant")
extra = [a for a in os.environ.get("PROBE_EXTRA_ARGS", "").split() if a]
t0 = time.time()
from isaacsim import SimulationApp
cfg = {"headless": True, "width": 256, "height": 256, "multi_gpu": False}
if extra:
    cfg["extra_args"] = extra
print("PROBE_CFG " + json.dumps(cfg), flush=True)
app = SimulationApp(cfg)
import omni.kit.app
frames = 0
for _ in range(40):
    app.update(); frames += 1
info = {"tag": tag, "extra_args": extra, "frames": frames, "seconds": round(time.time() - t0, 1),
        "kit": omni.kit.app.get_app().get_build_version()}
try:
    import omni.replicator.core as rep
    import numpy as np
    cam = rep.create.camera(position=(3, 3, 3), look_at=(0, 0, 0))
    rp = rep.create.render_product(cam, (256, 256))
    ann = rep.AnnotatorRegistry.get_annotator("rgb"); ann.attach(rp)
    for _ in range(5): app.update()
    rgb = np.asarray(ann.get_data())
    info["rgb_shape"] = list(rgb.shape); info["rgb_nonzero"] = bool(rgb.any())
except Exception as e:
    info["render_capture_error"] = f"{type(e).__name__}: {e}"
(OUT / f"probe_{tag}.json").write_text(json.dumps(info, indent=2) + "\n")
print("PROBE_RESULT " + json.dumps(info), flush=True)
app.close()
