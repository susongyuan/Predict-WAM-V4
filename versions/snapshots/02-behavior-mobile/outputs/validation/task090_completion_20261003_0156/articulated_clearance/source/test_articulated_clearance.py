#!/usr/bin/env python3
"""CPU-only synthetic math tests; never physical labels or production admission."""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import shlex
import sys
import threading
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "outputs/validation/task090_completion_20261003_0156"
DEFAULT_OUT = EVIDENCE / "articulated_clearance"
# Bound this standalone CPU test before importing numeric libraries.
if __name__ == "__main__":
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    resource.setrlimit(resource.RLIMIT_AS, (1024**3, 1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (120, 120))
import numpy as np
sys.path.insert(0, str(ROOT))
from harness.articulated_clearance import conservative_clearance

COMPARISONS = []
STATIC_REUSE = {}


def rotations(axis, angles):
    """Independent quaternion-to-matrix sampler, not the analytic A/B/C code."""
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    half = np.atleast_1d(angles) / 2
    w = np.cos(half)
    x, y, z = axis[:, None] * np.sin(half)
    return np.stack((w*w+x*x-y*y-z*z, 2*(x*y-w*z), 2*(x*z+w*y),
                     2*(x*y+w*z), w*w-x*x+y*y-z*z, 2*(y*z-w*x),
                     2*(x*z-w*y), 2*(y*z+w*x), w*w-x*x-y*y+z*z), -1).reshape(-1, 3, 3)


def transform(axis=(0, 0, 1), angle=0, translation=(0, 0, 0)):
    t = np.eye(4)
    t[:3, :3] = rotations(axis, angle)[0]
    t[:3, 3] = translation
    return t


def fixture():
    source = "synthetic_test:analytic_fixture_not_a_sensor_measurement"
    return dict(
        root_pose=dict(transform=transform(translation=(0, 0, 2)),
                       kind="synthetic_test", source=source),
        parts=[dict(id=pid, frame=frame, vertices=vertices, complete=True,
                    representation="vertices", source=source)
               for pid, frame, vertices in (("fixed", "root", [[0, 0, 10], [.1, 0, 10]]),
                                             ("moving", "child", [[0, 0, 1]]))],
        model=dict(part_ids=["fixed", "moving"], joint_count=1, complete=True, source=source),
        joints=[dict(axis_root=[0, 1, 0], origin_root_m=[0, 0, 0],
                     root_from_child_zero=transform(), angle_interval_rad=[0, 1],
                     includes_error=True, source=source)],
        scale=dict(interval=[1, 1], includes_error=True, source=source),
        plane=dict(normal_world=[0, 0, 1], point_world_m=[0, 0, 0],
                   kind="synthetic_test", source=source),
        errors=dict(translation_m=0, rotation_rad=0, plane_point_m=0,
                    plane_normal_rad=0, geometry_m=0, source=source),
        units=dict(length="m", angle="rad", scale="dimensionless"))


def sampled_min(data, angles, scales, pose=None, normal=None, anchor=None, geometry_shift=None):
    pose = data["root_pose"]["transform"] if pose is None else pose
    normal = np.asarray(data["plane"]["normal_world"] if normal is None else normal)
    anchor = np.asarray(data["plane"]["point_world_m"] if anchor is None else anchor)
    best = math.inf
    for part in data["parts"]:
        v = np.asarray(part["vertices"])
        if part["frame"] == "child":
            j = data["joints"][0]
            c = j["root_from_child_zero"]
            pivot = np.asarray(j["origin_root_m"])
            zero = v @ c[:3, :3].T + c[:3, 3] - pivot
            local = np.einsum("qij,vj->qvi", rotations(j["axis_root"], angles), zero) + pivot
        else:
            local = v[None, :, :]
        for scale in scales:
            world = scale * local @ pose[:3, :3].T + pose[:3, 3]
            if geometry_shift is not None:
                world = world + geometry_shift
            best = min(best, float(((world - anchor) @ normal).min()))
    return best


class ClearanceTests(unittest.TestCase):
    def compare(self, data, *, count=65537, expected=None, **perturbation):
        result = conservative_clearance(**data)
        self.assertEqual(result["status"], "geometry_only")
        lo, hi = data["joints"][0]["angle_interval_rad"] if data["joints"] else (0, 0)
        angles = np.linspace(lo, hi, count)
        scales = np.linspace(*data["scale"]["interval"], 3)
        actual = sampled_min(data, angles, scales, **perturbation)
        self.assertLessEqual(result["lower_bound_m"], actual)
        if expected is not None:
            self.assertAlmostEqual(result["nominal_min_m"], expected, places=11)
        COMPARISONS.append(dict(test=self.id(), angles=count, scales=len(scales),
                                sampled_min_m=actual, lower_bound_m=result["lower_bound_m"],
                                nominal_min_m=result["nominal_min_m"],
                                perturbed=bool(perturbation), synthetic_only=True))
        return result

    def test_cross_positive_pi_stationary(self):
        d = fixture(); d["joints"][0]["angle_interval_rad"] = [2.8, 3.5]
        self.compare(d, expected=1)

    def test_cross_negative_pi_stationary(self):
        d = fixture(); d["joints"][0]["angle_interval_rad"] = [-3.5, -2.8]
        self.compare(d, expected=1)

    def test_both_angle_endpoints_and_singleton(self):
        for lo, hi, expected in ((0, .9, 2 + math.cos(.9)),
                                 (3.4, 4.1, 2 + math.cos(3.4)),
                                 (.7, .7, 2 + math.cos(.7)),
                                 (math.pi, math.pi + .2, 1)):
            with self.subTest(interval=(lo, hi)):
                d = fixture(); d["joints"][0]["angle_interval_rad"] = [lo, hi]
                self.compare(d, expected=expected)

    def test_full_period_and_many_turns(self):
        for limits in ((-9, 12), (1e5, 1e5 + .8), (-math.pi, math.pi)):
            d = fixture(); d["joints"][0]["angle_interval_rad"] = limits
            self.compare(d, expected=1 if limits[1] - limits[0] >= math.tau else None)

    def test_zero_angular_amplitude(self):
        d = fixture(); d["joints"][0]["axis_root"] = [0, 0, 2]
        self.compare(d, expected=3)

    def test_non_aligned_axis_pivot_child_and_root_transforms(self):
        d = fixture(); j = d["joints"][0]
        j.update(axis_root=[1, -2, 3], origin_root_m=[.3, -.1, .7],
                 root_from_child_zero=transform((2, 1, -3), .63, (.4, .2, -.3)),
                 angle_interval_rad=[-2.5, 4])
        d["root_pose"]["transform"] = transform((1, 2, -1), .8, (.4, -.5, .9))
        d["plane"]["normal_world"] = np.array([2, -1, 3]) / math.sqrt(14)
        d["plane"]["point_world_m"] = [.1, -.2, .3]
        self.compare(d)

    def test_fixed_body_uses_all_vertices_and_parts(self):
        d = fixture(); d["joints"] = []; d["model"]["joint_count"] = 0
        d["parts"][1]["frame"] = "root"
        d["parts"][0]["vertices"] = [[0, 0, 10], [0, 0, -3]]
        d["parts"][0]["representation"] = "convex_hull"
        self.compare(d, expected=-1)

    def test_pivot_stays_in_contact_while_link_point_rises(self):
        d = fixture(); d["root_pose"]["transform"] = transform()
        d["parts"][1]["vertices"] = [[0, 0, 0], [1, 0, 0]]
        d["joints"][0]["angle_interval_rad"] = [-math.pi / 2, 0]
        r = self.compare(d, expected=0)
        self.assertFalse(r["positive_clearance_candidate"])

    def test_stationary_body_contact_prevents_whole_clearance(self):
        d = fixture(); d["parts"][0]["vertices"] = [[0, 0, -2], [0, 0, 1]]
        r = self.compare(d, expected=0)
        self.assertGreater(r["per_part_nominal_min_m"]["moving"], 0)
        self.assertFalse(r["positive_clearance_candidate"])

    def test_scale_both_signs_require_different_endpoints(self):
        for limits, expected in (([0, 0], 2.5), ([math.pi, math.pi], 0)):
            d = fixture(); d["joints"][0]["angle_interval_rad"] = limits
            d["scale"]["interval"] = [.5, 2]
            self.compare(d, expected=expected)

    def test_scale_interval_and_angle_widening_monotonic(self):
        d = fixture(); old = conservative_clearance(**d)["lower_bound_m"]
        d["scale"]["interval"] = [.5, 2]
        new = conservative_clearance(**d)["lower_bound_m"]
        self.assertLessEqual(new, old)
        d["joints"][0]["angle_interval_rad"] = [-3, 4]
        self.assertLessEqual(conservative_clearance(**d)["lower_bound_m"], new)

    def test_each_error_bound_monotonic_and_explained(self):
        for key in ("translation_m", "rotation_rad", "plane_point_m", "plane_normal_rad", "geometry_m"):
            d = fixture(); old = conservative_clearance(**d)["lower_bound_m"]
            d["errors"][key] = .12
            r = conservative_clearance(**d)
            self.assertLess(r["lower_bound_m"], old)
            self.assertAlmostEqual(r["nominal_min_m"] - sum(r["error_envelope_m"].values()),
                                   r["lower_bound_m"], places=12)

    def test_uncertainty_envelope_against_actual_perturbations(self):
        d = fixture(); d["parts"][0]["vertices"] = [[-.4, .2, -.3], [.3, -.1, .4]]
        d["parts"][1]["vertices"] = [[-.3, .1, .6], [.5, -.1, -.4]]
        d["joints"][0].update(axis_root=[1, 3, -2], origin_root_m=[.1, -.2, .3],
                             angle_interval_rad=[-3.4, 3.6])
        d["scale"]["interval"] = [.7, 1.4]
        d["errors"].update(translation_m=.04, rotation_rad=.3, plane_point_m=.02,
                            plane_normal_rad=.2, geometry_m=.01)
        for sign in (-1, 1):
            for axis in ([1, 0, 0], [1, 2, 3], [0, -1, 1]):
                pose = d["root_pose"]["transform"].copy()
                pose[:3, :3] = rotations(axis, sign * .3)[0] @ pose[:3, :3]
                pose[:3, 3] += np.array([0, 0, -.04])
                n = rotations(axis, -sign * .2)[0] @ np.array([0, 0, 1.])
                self.compare(d, count=32769, pose=pose, normal=n, anchor=.02*n,
                             geometry_shift=-.01*n)

    def test_seeded_independent_dense_comparisons(self):
        rng = np.random.default_rng(20261003)
        for _ in range(8):
            d = fixture(); j = d["joints"][0]
            j.update(axis_root=rng.normal(size=3), origin_root_m=rng.normal(size=3),
                     root_from_child_zero=transform(rng.normal(size=3), rng.uniform(-3, 3),
                                                    rng.normal(size=3)))
            lo = rng.uniform(-6, 6); j["angle_interval_rad"] = [lo, lo + rng.uniform(.1, 7)]
            d["root_pose"]["transform"] = transform(rng.normal(size=3), rng.uniform(-3, 3),
                                                      rng.normal(size=3))
            normal = rng.normal(size=3); normal /= np.linalg.norm(normal)
            d["plane"].update(normal_world=normal, point_world_m=rng.normal(size=3))
            for part in d["parts"]:
                part["vertices"] = rng.normal(size=(9, 3))
            d["scale"]["interval"] = [.6, 1.7]
            self.compare(d)

    def test_missing_inputs_never_default_to_zero(self):
        for key in fixture():
            d = fixture(); d[key] = None
            r = conservative_clearance(**d)
            self.assertEqual(r["status"], "Unknown"); self.assertIsNone(r["lower_bound_m"])
        for field in ("translation_m", "rotation_rad", "plane_point_m", "plane_normal_rad", "geometry_m"):
            d = fixture(); del d["errors"][field]
            self.assertEqual(conservative_clearance(**d)["status"], "Unknown")
        for group, field in (("root_pose", "transform"), ("plane", "point_world_m"),
                             ("scale", "interval"), ("model", "joint_count")):
            d = fixture(); del d[group][field]
            self.assertEqual(conservative_clearance(**d)["status"], "Unknown")

    def test_missing_or_partial_parts_and_provenance(self):
        for mutate in (lambda d: d["parts"].pop(),
                       lambda d: d["parts"][1].update(complete=False),
                       lambda d: d["model"].update(complete=False),
                       lambda d: d["parts"][1].update(source=""),
                       lambda d: d["root_pose"].update(source=None),
                       lambda d: d["scale"].update(includes_error=False),
                       lambda d: d["joints"][0].update(includes_error=False)):
            d = fixture(); mutate(d)
            self.assertEqual(conservative_clearance(**d)["status"], "Unknown")

    def test_multiple_or_missing_joint_unknown(self):
        d = fixture(); d["joints"].append(copy.deepcopy(d["joints"][0]))
        self.assertEqual(conservative_clearance(**d)["status"], "Unknown")
        d = fixture(); d["model"]["joint_count"] = 2
        self.assertEqual(conservative_clearance(**d)["status"], "Unknown")
        d = fixture(); d["joints"] = []
        self.assertEqual(conservative_clearance(**d)["status"], "Unknown")

    def test_static_pose_or_plane_not_observations(self):
        for key in ("root_pose", "plane"):
            d = fixture(); d[key]["kind"] = "authored_default"
            self.assertEqual(conservative_clearance(**d)["status"], "Unknown")

    def test_invalid_units_and_intervals(self):
        for key, value in (("length", "mm"), ("angle", "deg"), ("scale", "m")):
            d = fixture(); d["units"][key] = value
            with self.assertRaises(ValueError): conservative_clearance(**d)
        for interval in ((0, 1), (-1, 2), (2, 1), (1, math.nan), (1, math.inf)):
            d = fixture(); d["scale"]["interval"] = interval
            with self.assertRaises(ValueError): conservative_clearance(**d)
        d = fixture(); d["joints"][0]["angle_interval_rad"] = [1, -1]
        with self.assertRaises(ValueError): conservative_clearance(**d)

    def test_nan_vertices_bad_axes_and_missing_shape(self):
        for vertices in ([[0, 0, math.nan]], [[0, 0, math.inf]], [], [[0, 0]]):
            d = fixture(); d["parts"][1]["vertices"] = vertices
            with self.assertRaises(ValueError): conservative_clearance(**d)
        for group, key, value in (("plane", "normal_world", [0, 0, 2]),
                                   ("plane", "normal_world", [0, 0, 0]),
                                   ("errors", "rotation_rad", -.1),
                                   ("errors", "translation_m", math.nan)):
            d = fixture(); d[group][key] = value
            with self.assertRaises(ValueError): conservative_clearance(**d)
        d = fixture(); d["joints"][0]["axis_root"] = [0, 0, 0]
        with self.assertRaises(ValueError): conservative_clearance(**d)

    def test_non_se3_root_and_child_transforms(self):
        for row, col, value in ((0, 0, 2), (0, 0, -1), (0, 1, .1), (3, 0, .1), (0, 3, math.nan)):
            for child in (False, True):
                d = fixture()
                t = d["joints"][0]["root_from_child_zero"] if child else d["root_pose"]["transform"]
                t[row, col] = value
                with self.assertRaises(ValueError): conservative_clearance(**d)

    def test_geometry_only_never_admits_or_creates_labels(self):
        r = conservative_clearance(**fixture())
        self.assertTrue(r["positive_clearance_candidate"])
        self.assertTrue(r["synthetic_inputs"])
        self.assertFalse(r["verified"]); self.assertFalse(r["admitted"])
        self.assertIsNone(r["physical_label"]); self.assertNotIn("grasp", r)
        self.assertEqual(len(r["sources"]["parts"]), 2)

    def test_reuse_static_facts_without_constructing_measurements(self):
        paths = [EVIDENCE / "object_geometry_scope" / name for name in ("assessment.json", "scope.json")]
        assessment, scope = [json.loads(p.read_text()) for p in paths]
        facts = assessment["facts"]
        self.assertEqual(facts["material_geometry_links"], ["base_link", "link_0"])
        self.assertEqual(scope["static_usd"]["whole_declared_collision_native_root"]["vertex_count"], 529)
        self.assertEqual([p["enabled_collision_count"] for p in scope["static_usd"]["links"][:2]], [8, 4])
        self.assertIsNone(assessment["verdict"]["runtime_instance_scale"])
        d = fixture(); d.update(root_pose=None, plane=None, scale=None)
        self.assertEqual(conservative_clearance(**d)["status"], "Unknown")
        STATIC_REUSE.update(material_links=facts["material_geometry_links"],
            collision_counts=[8, 4], vertex_count=529, joint_axis=facts["joint_axis"],
            angle_limits_degrees=facts["joint_limits_degrees"], real_clearance_status="Unknown",
            no_asset_parsing=True, no_static_pose_injection=True,
            missing=["measured body/link membership", "external root pose and its error bounds",
                     "instance scale/error bounds", "observed support plane/error bounds",
                     "complete applicable enclosing vertices; audit stores bounds/counts only"],
            static_minimum_is_not_scene_ground=True)


def memory_percent():
    fields = {line.split(':')[0]: int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()}
    return 100 * (1 - fields['MemAvailable'] / fields['MemTotal'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.output.resolve()
    if out != DEFAULT_OUT.resolve():
        parser.error("Evidence must use the dedicated articulated_clearance directory")
    out.mkdir(parents=True, exist_ok=False)
    files = [ROOT / "harness/articulated_clearance.py", Path(__file__).resolve(),
             EVIDENCE / "object_geometry_scope/assessment.json", EVIDENCE / "object_geometry_scope/scope.json"]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (out / "source").mkdir()
    for p in files[:2]: (out / "source" / p.name).write_bytes(p.read_bytes())
    start = time.monotonic(); before = resource.getrusage(resource.RUSAGE_SELF)
    memory_samples = [memory_percent()]; stop = threading.Event()
    def monitor():
        while not stop.wait(.02): memory_samples.append(memory_percent())
    watcher = threading.Thread(target=monitor, daemon=True); watcher.start()
    with (out / "tests.log").open("x") as log:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(ClearanceTests)
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    stop.set(); watcher.join(); memory_samples.append(memory_percent())
    usage = resource.getrusage(resource.RUSAGE_SELF)
    unchanged = all(hashlib.sha256(p.read_bytes()).hexdigest() == hashes[str(p)] for p in files)
    passed = result.wasSuccessful() and unchanged and max(memory_samples) < 85
    report = dict(status="PASS" if passed else "FAIL", tests_run=result.testsRun,
        failures=len(result.failures), errors=len(result.errors), skipped=len(result.skipped),
        command=shlex.join([sys.executable, "-B", str(Path(__file__).resolve()), "--output", str(out)]),
        cwd=str(ROOT), source_sha256=hashes, inputs_unchanged=unchanged,
        implementation_lines=len(files[0].read_text().splitlines()),
        resources=dict(elapsed_s=time.monotonic()-start,
            cpu_user_s=usage.ru_utime-before.ru_utime, cpu_system_s=usage.ru_stime-before.ru_stime,
            peak_rss_mib=usage.ru_maxrss/1024, cpu_affinity=sorted(os.sched_getaffinity(0)),
            host_logical_cpus=os.cpu_count(), process_cpu_capacity_fraction=1/os.cpu_count(),
            peak_system_ram_percent=max(memory_samples), memory_sampling_interval_s=.02,
            address_space_limit_bytes=1024**3, cpu_time_limit_s=120,
            thread_environment={k: os.environ[k] for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
            gpu_used=False, simulation_started=False, downloads=0, agents_spawned=0),
        static_facts_reused=STATIC_REUSE, dense_comparisons=COMPARISONS,
        scope="Synthetic CPU mathematics only; no physical labels, no production integration",
        physical_label=None, admitted=False)
    (out / "execution.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "tests_run", "failures", "errors", "implementation_lines", "resources")}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
