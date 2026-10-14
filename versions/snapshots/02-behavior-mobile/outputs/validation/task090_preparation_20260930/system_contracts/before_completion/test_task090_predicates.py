#!/usr/bin/env python3
"""Task90 CPU logic tests, NOT detector calibration or physical validation.

Metric fixtures are metamorphic transforms of saved TRAIN state61/camera observations.
No hardcoded scene/object/test-instance coordinates. RGB is real TRAIN input; constant
synthetic depth and object/reference relations are explicitly diagnostic test doubles.
All positive relation candidates MUST remain unknown/unverified in production output.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image

from harness.task090_predicates import Task090Postconditions, validate_mode, EEF, FINGERS, GRIP_ACTION
from harness.grounding import Catalog, CandidateProgram, Step, compile_programs, parse
from harness.orchestrator import ConditionBank, Orchestrator, _FakeClosure

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "outputs/validation/task090_preparation_20260930/data_runtime/train_observations/manifest.json"


class Fixtures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        manifest = json.loads(MANIFEST.read_text())
        assert manifest["split"] == "train"
        cls.frames = [r for r in manifest["samples"] if "zed" in r["camera"]]
        cls.images = [np.asarray(Image.open(r["rgb"]).convert("RGB")) for r in cls.frames]
        cls.state0 = np.array(cls.frames[0]["state61"], float)
        # Geometry-derived fixture scale: measured TRAIN finger stroke, not a deployment threshold.
        cls.u = float(cls.state0[FINGERS["left"]].sum()) / 10
        u = cls.u
        cls.cal = dict(source_split="geometry", source=f"CPU metamorphic fixture from {MANIFEST}; NOT calibrated",
                       calibration_id="cpu-fixture-not-admitted", uses_validation=False,
                       hold_finger_min=u, hold_finger_max=6*u, open_finger_min=8*u,
                       eef_near_max=2*u, lift_min=2*u, motion_min=2*u, co_motion_max=u,
                       support_start_max=u, detach_min=6*u, settle_max=u,
                       hold_frames=3, settle_frames=3, max_frame_gap=2)

    def make(self, skill="pick up from", target="half_banana", reference="countertop", calibration=True):
        self.p = Task090Postconditions(self.cal if calibration else None)
        self.objects = (target, reference)
        self.skill = skill
        self.p.start(f"{skill}({target},{reference})", skill, self.objects, self.state0, 0, episode=0)
        return self.p

    def row(self, t, *, lift=0, eef_lift=None, opened=False, detach=False, margin=None):
        s = self.state0.copy()
        hand = self.p.hand
        u = self.u
        ref = self.state0[EEF[hand]].copy()
        delta = np.zeros_like(ref); delta[-1] = lift
        obj = ref + delta
        eef = obj.copy()
        if eef_lift is not None:
            eef = ref.copy(); eef[-1] += eef_lift
        if detach:
            eef[0] += 2*self.cal["detach_min"]
        s[EEF[hand]] = eef
        s[FINGERS[hand]] = (9*u if opened else 4*u) / 2
        a = np.zeros(23); a[GRIP_ACTION[hand]] = 1 if opened else -1
        f = self.frames[t % len(self.frames)]
        obs = {"robot_r1::robot_r1:zed_link:Camera:0::rgb": self.images[t % len(self.images)],
               "robot_r1::robot_r1:zed_link:Camera:0::depth_linear": np.full(self.images[0].shape[:2], np.linalg.norm(ref)),
               "robot_r1::cam_rel_poses": np.array(f["camera_pose7"])}
        d = dict(label=self.objects[0], object_id=self.objects[0], track_id="cpu-target-track", hand=hand,
                 visible=True, metric_valid=True, source="rgbd", calibration_id=self.cal["calibration_id"], xyz_base=obj,
                 relation=dict(reference=self.objects[1], reference_track_id="cpu-reference-track", reference_visible=True,
                               reference_xyz_base=ref, kind={"pick up from":"support", "place in":"inside",
                                   "place on next to":"on_next_to"}[self.skill],
                               support_clearance_m=lift, placement_margin_m=self.u if margin is None else margin))
        return s, a, obs, [d]

    def update(self, t, **kw):
        s, a, obs, det = self.row(t, **kw)
        return self.p.update(s, a, t, episode=0, observation=obs, detections=det)

    def assertUnknown(self, ev):
        self.assertFalse(ev["verified"])
        self.assertFalse(ev["satisfied"])
        self.assertFalse(ev["failed"])
        self.assertFalse(ev["admitted"])
        self.assertEqual(ev["status"], "unknown")
        self.assertTrue(ev["admission_blockers"])


class Relations(Fixtures):
    def test_visible_comoving_lift_candidate_is_not_physical_success(self):
        self.make()
        for t in range(3):
            ev = self.update(t, lift=3*self.u*t)
        self.assertTrue(ev["detail"]["candidate_satisfied"], ev)
        self.assertUnknown(ev)

    def test_grip_or_eef_height_alone_cannot_make_pick_candidate(self):
        for which in ("stationary", "eef_only", "object_only", "opened"):
            with self.subTest(which=which):
                self.make()
                for t in range(4):
                    kw = dict(lift=0)
                    if which == "eef_only": kw["eef_lift"] = 3*self.u*t
                    if which in ("object_only", "opened"): kw["lift"] = 3*self.u*t
                    if which == "object_only": kw["eef_lift"] = 0
                    if which == "opened": kw["opened"] = True
                    ev = self.update(t, **kw)
                self.assertFalse(ev["detail"]["candidate_satisfied"], ev)
                self.assertUnknown(ev)

    def test_no_supported_start_or_missing_reference_fails_closed(self):
        self.make()
        ev = self.update(0, lift=10*self.u)
        self.assertEqual(ev["reason"], "no_observed_supported_start")
        s,a,obs,ds = self.row(1); ds[0]["relation"]["reference_visible"] = False
        ev = self.p.update(s,a,1,episode=0,observation=obs,detections=ds)
        self.assertEqual(ev["reason"], "reference_not_observed")
        self.assertUnknown(ev)

    def test_place_requires_bound_carry_release_detach_and_stability(self):
        for skill, target, ref in (("place in","half_banana","trash_can"),
                                    ("place on next to","trash_can","sink")):
            with self.subTest(skill=skill):
                self.make(skill, target, ref)
                for t in range(3): self.update(t, lift=3*self.u*t)
                for t in range(3,7):
                    ev = self.update(t, lift=6*self.u, opened=True, detach=True)
                self.assertTrue(ev["detail"]["candidate_satisfied"], ev)
                self.assertUnknown(ev)

    def test_place_negative_sequences(self):
        for case in ("never_held", "no_motion", "not_released", "still_attached", "outside", "drifting"):
            with self.subTest(case=case):
                self.make("place in", reference="trash_can")
                for t in range(3):
                    self.update(t, lift=0 if case == "no_motion" else 3*self.u*t,
                                opened=case == "never_held")
                for t in range(3,9):
                    ev = self.update(t, lift=(t*3 if case == "drifting" else 6)*self.u,
                                     opened=case != "not_released", detach=case != "still_attached",
                                     margin=-self.u if case == "outside" else self.u)
                self.assertFalse(ev["detail"]["candidate_satisfied"], ev)
                self.assertUnknown(ev)

    def test_missing_or_nontrain_calibration_and_docking_points_are_unknown(self):
        self.make(calibration=False)
        self.assertEqual(self.update(0)["reason"], "missing_relation_calibration")
        self.make(); self.p = Task090Postconditions({**self.cal, "source_split":"validation"})
        self.p.start("k",self.skill,self.objects,self.state0,0,episode=0)
        self.assertEqual(self.update(0)["reason"], "invalid_calibration_provenance")
        self.make(); s,a,obs,ds = self.row(0)
        # This is the current detector interface: a docking point cannot become a verified object relation.
        ev = self.p.update(s,a,0,episode=0,observation=obs,
                           detections=[{"label":self.objects[0], "xyz_base":ds[0]["xyz_base"], "score":1.0}])
        self.assertEqual(ev["reason"], "missing_exact_target_track"); self.assertUnknown(ev)

    def test_target_hand_reference_and_track_binding(self):
        for patch_d in ({"object_id":"half_pomegranate"}, {"hand":"right"}, {"track_id":"other"},
                        {"visible":False}, {"calibration_id":"other"}):
            with self.subTest(patch=patch_d):
                self.make(); self.update(0)
                s,a,obs,ds = self.row(1,lift=3*self.u); ds[0].update(patch_d)
                ev = self.p.update(s,a,1,episode=0,observation=obs,detections=ds)
                self.assertUnknown(ev); self.assertEqual(self.p.hold_n,0)
        self.make(); s,a,obs,ds = self.row(0); ds[0]["relation"]["reference"] = "sink"
        self.assertEqual(self.p.update(s,a,0,episode=0,observation=obs,detections=ds)["reason"],
                         "missing_target_reference_relation")

    def test_frames_episodes_and_reset_do_not_reuse_evidence(self):
        self.make(); self.update(0)
        s,a,obs,ds = self.row(0)
        for key in obs:
            if key.endswith("cam_rel_poses"): obs[key] = obs[key] + self.u
        ev = self.p.update(s,a,1,episode=0,observation=obs,detections=ds)
        self.assertEqual(ev["reason"], "repeated_sensor_frame"); self.assertEqual(self.p.hold_n,0)
        self.make(); self.update(2)
        self.assertEqual(self.update(1)["reason"], "nonmonotonic_frame")
        self.assertEqual(self.update(8)["reason"], "frame_gap")
        self.make(); self.update(0)
        s,a,obs,ds = self.row(1,lift=3*self.u)
        ev = self.p.update(s,a,1,episode=1,observation=obs,detections=ds)
        self.assertEqual(ev["reason"], "episode_mismatch"); self.assertEqual(self.p.hold_n,0)
        self.p.reset_episode(); self.assertIsNone(self.p.update(s,a,2,episode=1))
        self.p.start("new",self.skill,self.objects,self.state0,0,episode=1)
        ev = self.p.update(s,a,1,episode=1,observation=obs,detections=ds)
        self.assertEqual(ev["reason"], "no_observed_supported_start")

    def test_invalid_or_missing_sensor_inputs(self):
        for suffix in ("::rgb","::depth_linear","::cam_rel_poses"):
            self.make(); s,a,obs,ds = self.row(0)
            obs = {k:v for k,v in obs.items() if not k.endswith(suffix)}
            self.assertUnknown(self.p.update(s,a,0,episode=0,observation=obs,detections=ds))
        self.make(); s,a,obs,ds = self.row(0); ds.append(copy.deepcopy(ds[0]))
        self.assertEqual(self.p.update(s,a,0,episode=0,observation=obs,detections=ds)["reason"],
                         "target_missing_or_ambiguous")


class Integration(Fixtures):
    def catalog(self):
        # Exact TRAIN contract forms, not a replacement task configuration.
        data = json.loads((ROOT / "data/skill_vtt_task090/contract_targets.json").read_text())
        cat = Catalog.__new__(Catalog); cat.patterns = {}; cat.classes = {}
        for key in list(data["contracts"]) + list(data["navigation"]):
            name, objects = key[:-1].split("(",1)
            cat.patterns.setdefault(name,set()).add(tuple(objects.split(",")))
        cat.categories = sorted({o for rows in cat.patterns.values() for row in rows for o in row})
        return cat

    def test_actual_task90_train_forms_compile_adjacent_bin_and_fruit_places(self):
        cat = self.catalog()
        steps = [Step("pick up from",("trash_can","floors")), Step("place on next to",("trash_can","sink"))]
        for target in ("half_pomegranate", "half_banana"):
            source = next(p[1] for p in cat.patterns["pick up from"] if p[0] == target)
            steps += [Step("pick up from",(target,source)), Step("place in",(target,"trash_can"))]
        valid, errors = compile_programs([CandidateProgram("all Task90 manipulation effects",steps)],cat)
        self.assertEqual(errors,[])
        self.assertEqual([s for s in valid[0].steps if s.skill != "move to"],steps)
        self.assertIn(Step("move to",("sink",)),valid[0].steps)
        self.assertIn(Step("move to",("trash_can",)),valid[0].steps)
        self.assertTrue(compile_programs([CandidateProgram("bad",[Step("place on next to",("trash_can",))])],cat)[1])

    def test_unknown_cannot_advance_or_promote_memory_even_at_budget(self):
        cat = self.catalog(); skill="pick up from"; objects=("trash_can","floors")
        protos={name:np.ones(1024,dtype=np.float32) for name in cat.patterns}
        o=Orchestrator(catalog=cat,bank=ConditionBank(protos),closure_monitor=_FakeClosure(1),
                       precondition_monitor=None,budgets={"move to(trash_can)":2,"pick up from(trash_can,floors)":3},
                       max_attempts=1,budget_factor=1,navigation_completion="docking",navigation_settle_steps=1,
                       manipulation_completion="postcondition")
        o.reset([CandidateProgram("pick",[Step(skill,objects)])])
        self.make(skill,*objects)
        for t in range(12):
            ev=self.p._evidence(t,"missing_relation_calibration",candidate=True)
            c=o.contract
            nav=None if c is None or not c.is_navigation else dict(docked=True,key=c.key,observation_id=t)
            d=o.tick(np.ones(1024),t,base_velocity=np.zeros(3),postcondition=ev,navigation_evidence=nav)
        self.assertEqual(d.mode,"HOLD"); self.assertEqual(o.index,1)
        self.assertEqual(len(o.completed_steps),1)
        self.assertFalse(any(a.successes or a.failures or a.support_count for arms in o.bank.arms.values() for a in arms))
        self.assertFalse(any(r["success"] is True for r in o.realized if r["key"].startswith(skill)))

    def test_policy_reuses_current_detection_and_preserves_allowlist(self):
        from harness.policy_server import WamPolicy
        self.make(); s,a,obs,ds=self.row(0)
        calls=[]
        def detect(o,st): calls.append(o); return ds,None
        stub=SimpleNamespace(perception=SimpleNamespace(detect=detect),post_task1=self.p,
                             _last_action=a,step_in_episode=0,episode=0)
        WamPolicy._task090_post_update(stub,obs,s,ds)
        self.assertEqual(len(calls),0)
        stub.step_in_episode=1
        WamPolicy._task090_post_update(stub,obs,s,None)
        self.assertEqual(len(calls),1)
        dirty={**obs,"object_scope":{"fake":"success"},"reward":1,"task090_relation":{"verified":True}}
        self.assertEqual(set(WamPolicy._policy_observation(dirty)),set(obs))

    def test_mode_guards_no_ik_proxy_skip_or_candidate_advance(self):
        for bad in ({"postconditions":"task1_physical"},{"postconditions":"patch_candidate"},
                    {"manip_executor":"primitive"},{"skip_failed_objects":True},
                    {"candidate_outcome_policy":"legacy_advance"}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                validate_mode(90,{"postconditions":"task090_observed",**bad})
        for task in (0,1,13):
            with self.assertRaises(ValueError): validate_mode(task,{"postconditions":"task090_observed"})
        validate_mode(0,{"postconditions":"task0_physical"})
        validate_mode(1,{"postconditions":"task1_physical"})
        validate_mode(90,{"postconditions":"task090_observed"})
        from harness.policy_server import WamPolicy
        # Must reject before loading weights/initializing CUDA.
        with patch("harness.policy_server.task_profile.TASK_ID",90):
            with self.assertRaisesRegex(ValueError,"Task90 requires"):
                WamPolicy("nonexistent",skill=None,exec_horizon=8,num_steps=1,device="cpu",cuda_fraction=.1,
                          seed=0,log_dir="unused",orchestrate={"postconditions":"task1_physical"})

    def test_bin_put_down_does_not_change_legacy_task1_carry_logic(self):
        from harness.policy_server import WamPolicy
        done=[Step("pick up from",("trash_can","floors")),Step("place on next to",("trash_can","sink"))]
        stub=SimpleNamespace(postcondition_mode="task090_observed",orch=SimpleNamespace(completed_steps=done))
        self.assertFalse(WamPolicy._carrying_bin(stub))
        stub.postcondition_mode="task1_physical"
        self.assertTrue(WamPolicy._carrying_bin(stub))

    def test_existing_task0_task1_fixed_programs_compile_unchanged(self):
        for task in (0,1):
            cat=Catalog(ROOT / f"configs/task{task:03d}_skill_program.json")
            data=json.loads((ROOT / f"configs/task{task:03d}_fixed_program.json").read_text())
            valid,errors=parse(json.dumps({"candidates":[{"interpretation":"legacy","program":data["program"]}]}),cat,cat.categories)
            self.assertEqual(errors,[])
            self.assertEqual([(s.skill,list(s.objects)) for s in valid[0].steps],
                             [(r["skill"],r["objects"]) for r in data["program"]])


if __name__ == "__main__":
    unittest.main(verbosity=2)
