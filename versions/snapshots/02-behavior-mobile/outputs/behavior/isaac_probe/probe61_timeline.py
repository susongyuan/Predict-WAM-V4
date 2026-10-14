"""Isaac Sim 6.1 headless render probe v3: replicator render_product + rgb annotator with the timeline PLAYING
(Kit 110 sensor scheduling ticks sensors on simulation time). Falls back to rep.orchestrator.step()."""
import json, os, time
from pathlib import Path
OUT = Path("/home/user/Robotic/project2/outputs/behavior/isaac_probe")
t0 = time.time()
from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "width": 256, "height": 256, "multi_gpu": False})
import numpy as np, omni.kit.app, omni.usd, omni.timeline
from pxr import UsdGeom, UsdLux, Gf
import omni.replicator.core as rep
info = {"tag": "isaac61_timeline", "kit": omni.kit.app.get_app().get_build_version(), "startup_s": round(time.time() - t0, 1)}
def poll(ann, seconds, label):
    t1 = time.time(); n = 0
    while time.time() - t1 < seconds:
        app.update(); n += 1
        data = ann.get_data()
        arr = np.asarray(data["data"] if isinstance(data, dict) else data)
        if arr.size and arr.any():
            return arr, n, round(time.time() - t1, 2)
    print(f"PROBE_PHASE {label}: no data after {n} updates / {seconds}s", flush=True)
    return None, n, None
try:
    stage = omni.usd.get_context().get_stage()
    UsdGeom.Xform.Define(stage, "/World")
    UsdLux.DistantLight.Define(stage, "/World/light").CreateIntensityAttr(3000)
    cube = UsdGeom.Cube.Define(stage, "/World/cube"); cube.CreateSizeAttr(0.5)
    UsdGeom.XformCommonAPI(cube).SetTranslate(Gf.Vec3d(0, 0, 0.25)); cube.CreateDisplayColorAttr([Gf.Vec3f(1.0, 0.2, 0.2)])
    floor = UsdGeom.Cube.Define(stage, "/World/floor"); floor.CreateSizeAttr(1.0)
    UsdGeom.XformCommonAPI(floor).SetScale(Gf.Vec3f(4, 4, 0.02))
    cam = UsdGeom.Camera.Define(stage, "/World/cam")
    UsdGeom.XformCommonAPI(cam).SetTranslate(Gf.Vec3d(2.0, 2.0, 1.5)); UsdGeom.XformCommonAPI(cam).SetRotate(Gf.Vec3f(60, 0, 135))
    rp = rep.create.render_product("/World/cam", (224, 224))
    ann = rep.AnnotatorRegistry.get_annotator("rgb"); ann.attach([rp])
    tl = omni.timeline.get_timeline_interface(); tl.play()
    for _ in range(4): app.update()
    rgb, n, dt = poll(ann, 90, "timeline_play")
    info["timeline_play"] = {"updates": n, "first_rgb_after_s": dt}
    if rgb is None:
        t1 = time.time()
        for i in range(30):
            rep.orchestrator.step(rt_subframes=2, pause_timeline=False)
            data = ann.get_data(); arr = np.asarray(data["data"] if isinstance(data, dict) else data)
            if arr.size and arr.any():
                rgb = arr; info["orchestrator_step"] = {"steps": i + 1, "seconds": round(time.time() - t1, 2)}; break
    info["rgb_shape"] = list(rgb.shape) if rgb is not None else [0]; info["rgb_nonzero"] = rgb is not None
    if rgb is not None:
        info["rgb_mean"] = float(rgb[..., :3].mean()); info["unique_colors"] = int(len(np.unique(rgb[..., :3].reshape(-1, 3), axis=0)))
        from PIL import Image; Image.fromarray(rgb[..., :3].astype(np.uint8)).save(OUT / "probe61_timeline.png")
except Exception as e:
    import traceback; info["error"] = f"{type(e).__name__}: {e}"; info["trace"] = traceback.format_exc()[-1500:]
info["seconds"] = round(time.time() - t0, 1)
(OUT / "probe_isaac61_timeline.json").write_text(json.dumps(info, indent=2) + "\n")
print("PROBE_RESULT " + json.dumps(info), flush=True)
app.close()
