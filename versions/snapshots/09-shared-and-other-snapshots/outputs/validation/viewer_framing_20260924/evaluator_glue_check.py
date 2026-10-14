"""CPU glue test: Evaluator.start_recording/_write_video/stop_recording with auto framing on fake sim objects."""
import json, os, tempfile, types
import numpy as np, torch as th
from omegaconf import OmegaConf
import omnigibson as og
import omnigibson.eval.evaluator as E

class Prim:
    def __init__(self, p, q=(0, 0, 0, 1)): self.p, self.q = th.tensor(p, dtype=th.float32), th.tensor(q, dtype=th.float32)
    def get_position_orientation(self): return self.p.clone(), self.q.clone()

yaw = 0.0
links = {n: Prim(p) for n, p in {"base": (2.7, 2.15, 0), "torso": (2.75, 2.15, 1.0), "head": (2.8, 2.15, 1.6),
                                 "l_eef": (3.3, 2.4, 1.0), "r_eef": (3.3, 1.9, 1.0)}.items()}
robot = types.SimpleNamespace(prim_path="/World/scene_0/robot_r1", links=links, is_manipulation=True,
                              eef_links={"left": links["l_eef"], "right": links["r_eef"]},
                              get_position_orientation=links["base"].get_position_orientation)
car = types.SimpleNamespace(prim_path="/World/scene_0/car_xyz",
                            aabb=(th.tensor([3.4, 1.2, 0.0]), th.tensor([5.9, 3.1, 1.4])))
floor = types.SimpleNamespace(prim_path="/World/scene_0/floors", aabb=(th.zeros(3), th.ones(3)))
scope = {"car.n.01_1": car, "floor.n.01_1": floor, "agent.n.01_1": robot, "container.n.01_1": None}

# Fake PhysX: a south wall at y=0 (box), reported as hits; robot/car hits too, to exercise the ignore filter.
WALL = (np.array([0, -0.15, 0]), np.array([6, 0, 2.6]))
calls = {"n": 0}
def raycast_all(origin, dir, distance, reportFn):
    calls["n"] += 1
    o, d = np.array(origin), np.array(dir)
    end = o + d * distance
    hits = []
    # robot hit near the end of every ray (should be ignored for robot rays)
    hits.append(types.SimpleNamespace(rigid_body="/World/scene_0/robot_r1/base_link", collision="c"))
    t0, t1 = 0.0, 1.0; seg = end - o; hit_wall = True
    for k in range(3):
        if abs(seg[k]) < 1e-12:
            if not (WALL[0][k] <= o[k] <= WALL[1][k]): hit_wall = False
            continue
        a, b = (WALL[0][k] - o[k]) / seg[k], (WALL[1][k] - o[k]) / seg[k]
        t0, t1 = max(t0, min(a, b)), min(t1, max(a, b))
    if hit_wall and t0 <= t1:
        hits.append(types.SimpleNamespace(rigid_body="/World/scene_0/walls/base_link", collision="c"))
    for h in hits:
        if not reportFn(h): break

class Cam:
    horizontal_aperture = focal_length = None
    def set_position_orientation(self, p, q): self.pose = (p, q)
    def get_obs(self): return [{"rgb": th.zeros((720, 1280, 4), dtype=th.uint8)}]

og.sim = types.SimpleNamespace(viewer_camera=Cam(), render=lambda: None, get_sim_step_dt=lambda: 1 / 30,
                               psqi=types.SimpleNamespace(raycast_all=raycast_all))
ev = E.Evaluator.__new__(E.Evaluator)
ev.cfg = OmegaConf.create({"video_view": "viewer", "video_width": 1280, "video_height": 720, "video_framing": "auto",
                           "video_camera_offset": [-2.5, -2.5, 2.1], "video_camera_target": [0.6, 0.0, 0.9]})
ev.robot = robot
ev.env = types.SimpleNamespace(task=types.SimpleNamespace(object_scope=scope))
ev._video_writer = ev._video_path = ev._video_metadata = None
ev._video_rate, ev._video_frames = 30, 0
ev._framing = ev._framing_log = None; ev._framing_pois = []
out = tempfile.mkdtemp()
path = os.path.join(out, "v.mp4")
assert E.Evaluator._viewer_ray_blocked((1, 1, 1), (1, -1, 1), ("/World/scene_0/robot_r1",))  # through wall
assert not E.Evaluator._viewer_ray_blocked((1, 1, 1), (1, 2, 1), ("/World/scene_0/robot_r1",))  # robot only -> clear
assert E.Evaluator._viewer_ray_blocked((1, 1, 1), (1, 2, 1), ())  # robot hit counts when not ignored
assert E.Evaluator._viewer_ray_blocked((1, 1, 1), (1, 2, 1), ("/World/scene_0/robot",))  # sibling-name prefix is not ignored
ev.start_recording(path, rate=30)
for i in range(10):
    ev._write_video()
ev.stop_recording(completed=True, success=False, steps=10)
meta = json.load(open(path + ".json"))
rows = [json.loads(l) for l in open(path + ".camera.jsonl")]
print("pois:", meta["framing_points_of_interest"], "frames:", meta["frames"], "coverage:", meta["complete_frame_coverage"])
print("stats:", meta["framing_stats"])
print("first row:", {k: rows[0][k] for k in ("candidate", "offset_yaw_frame", "valid", "optional_missing", "pois")})
print("legacy default camera would be at y =", 2.15 - 2.5, "(outside the south wall at y=0)")
print("rays:", calls["n"], "sha:", meta["recording_utils_sha256"][:12], "policy_inputs_unchanged:", meta["policy_inputs_unchanged"])
assert meta["framing"] == "auto" and meta["framing_points_of_interest"] == ["car.n.01_1"]
assert len(rows) == 11 and all(r["valid"] for r in rows) and rows[0]["pois"] == ["car.n.01_1"]
assert os.path.getsize(path) > 0 and meta["frames"] == 11 and meta["complete_frame_coverage"]
# fixed mode still uses the legacy follow pose and writes no camera log
ev.cfg.video_framing = "fixed"; path2 = os.path.join(out, "v2.mp4")
ev.start_recording(path2, rate=30); ev.stop_recording(completed=True, success=False, steps=0)
assert not os.path.exists(path2 + ".camera.jsonl") and json.load(open(path2 + ".json"))["framing"] == "fixed"
print("GLUE PASS")
