#!/usr/bin/env python3
"""CPU native-embodiment, padding-isolation and strict-retarget contracts; no robot success claim."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import torch.nn.functional as F
from harness.wam_behavior import proprio_keep_mask, load_upstream
from harness.wam_v3 import (HEAD_CAMERA, LEFT_WRIST, RIGHT_WRIST, V3Context, WamV3Wrapper,
                            bf16_from_bits, load_retargeted_v3_state, v3_model_class)

VIEWS = [HEAD_CAMERA, LEFT_WRIST, RIGHT_WRIST]
RESET = {f"{name}.{part}" for name in ("action_proj", "proprio_proj", "output_proj")
         for part in ("weight", "bias")}


class TinyAction(torch.nn.Module):
    """Cross-token mixing makes padding leaks observable; real Linear interfaces match V3."""
    action_len, proprio_len, num_memory, use_future_feat = 4, 3, 2, True

    def __init__(self, state_dim=14, action_dim=14):
        super().__init__()
        self.action_proj = torch.nn.Linear(action_dim, 8)
        self.proprio_proj = torch.nn.Linear(state_dim, 8)
        self.output_proj = torch.nn.Linear(8, action_dim)
        self.mem_proj = torch.nn.Linear(8, 8)
        self.feat_proj = torch.nn.Linear(4, 8)
        self.gate_head = torch.nn.Linear(8, 3)
        self.phase_head = torch.nn.Linear(8, 4)
        self.dist_head = torch.nn.Linear(8, 1)
        self.future_feat_decoder = torch.nn.Linear(8, 4)

    def forward(self, t, noisy_actions, qpos_history, dino_features_list, task_cond, memory):
        self.last_noisy = noisy_actions.detach().clone()
        self.last_feats = [f.detach().clone() for f in dino_features_list]
        assert all(f.dtype == torch.float32 for f in dino_features_list)
        h = self.action_proj(noisy_actions)
        context = (h.mean(1) + self.proprio_proj(qpos_history).mean(1)
                   + self.mem_proj(memory).mean((1, 2))
                   + self.feat_proj(dino_features_list[0]).mean((1, 2))
                   + task_cond + t[:, None])
        return {"final_pred": self.output_proj(h + context[:, None]),
                "cond_tokens": context[:, None].expand(-1, 6, -1),
                "gate": self.gate_head(context).softmax(-1),
                "phase_logits": self.phase_head(context), "log_dist": self.dist_head(context)[:, 0]}


class EmbodimentTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(41)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.norm_path = Path(self.tmp.name) / "norm.json"

    def make_wrapper(self, state_dim=14, action_dim=14, *, norm_edit=None, stats_edit=None,
                     model=None, **kwargs):
        model = model or TinyAction(state_dim, action_dim)
        ns = {"action_repr": "absolute", "behavior1k": {
            "state": {"min": [-1.] * state_dim, "max": [1.] * state_dim},
            "action": {"min": [-1.] * action_dim, "max": [1.] * action_dim}}}
        ns["native14"] = copy.deepcopy(ns["behavior1k"])
        if norm_edit:
            norm_edit(ns)
        self.norm_path.write_text(json.dumps(ns))
        stats = {"views": VIEWS, "proprio_diff_std": np.ones((2, state_dim)).tolist(),
                 "memory_diff_std": np.ones((3, 2, 8)).tolist()}
        if stats_edit:
            stats_edit(stats)
        if state_dim != 61:
            kwargs.setdefault("proprio_keep_override", [True] * state_dim)
        return WamV3Wrapper(model, views=VIEWS, stats=stats, norm_stats_path=self.norm_path,
                            proprio_mask="positions_no_gripper", device="cpu", **kwargs)

    def batch(self, state_dim=14, action_dim=14):
        return {"feats": torch.randn(2, 3, 3, 5, 4), "future_feats": torch.randn(2, 3, 2, 4),
                "state": torch.randn(2, 3, state_dim), "pool": torch.randn(2, 3, 3, 8),
                "action_sequence": torch.randn(2, 4, action_dim), "task_cond": torch.randn(2, 8),
                "phase": torch.full((2,), -1), "active_hand": torch.full((2,), -1),
                "dist": torch.full((2,), float("nan"))}

    def run_grad(self, wrapper, batch, **kwargs):
        wrapper.zero_grad(set_to_none=True)
        torch.manual_seed(20261004)
        loss, info, out = wrapper(batch, {"future": 1., "phase_ce": 1., "gate_kl": 1., "log_dist": 1.}, **kwargs)
        loss.backward()
        grads = {k: p.grad.clone() for k, p in wrapper.named_parameters() if p.grad is not None}
        return loss.detach(), info, out, grads

    def assert_grads_equal(self, a, b):
        self.assertEqual(a.keys(), b.keys())
        for key in a:
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0, msg=key)

    def test_native14_and_legacy61_normalization(self):
        for dim, action_dim in ((14, 14), (61, 23)):
            with self.subTest(dim=dim):
                w = self.make_wrapper(dim, action_dim)
                s = torch.randn(2, 3, dim)
                expected = torch.cat([s[:, :1], (s[:, :1] - s[:, 1:]).clamp(-5, 5)], 1)
                mask = torch.ones(dim) if dim == 14 else torch.tensor(proprio_keep_mask("positions_no_gripper"))
                torch.testing.assert_close(w.normalize_state(s), expected * mask, atol=2e-7, rtol=1e-6)
                a = torch.randn(2, 4, action_dim)
                torch.testing.assert_close(w.denormalize_action(w.normalize_action(a)), a)
                self.assertEqual((w.state_dim, w.action_dim), (dim, action_dim))

    def test_native_requires_explicit_mask_and_key(self):
        with self.assertRaisesRegex(ValueError, "explicit"):
            self.make_wrapper(proprio_keep_override=None)
        w = self.make_wrapper(normalization_key="native14", norm_edit=lambda n: n.pop("behavior1k"))
        self.assertEqual(w.normalization_key, "native14")
        with self.assertRaises(KeyError):
            self.make_wrapper(normalization_key="absent")

    def test_constructor_rejects_invalid_norms_masks_and_stds(self):
        for section in ("action", "state"):
            for edge in ("min", "max"):
                for value in ([0.] * 13, [[0.] * 14], [float("nan")] * 14, [float("inf")] * 14):
                    with self.subTest(section=section, edge=edge, value=value[:1]), self.assertRaises(ValueError):
                        self.make_wrapper(norm_edit=lambda n: n["behavior1k"][section].update({edge: value}))
            with self.assertRaisesRegex(ValueError, "range"):
                self.make_wrapper(norm_edit=lambda n: n["behavior1k"][section].update(min=[2.] * 14))
        for mask in ([1.] * 13, [[1.] * 14], [.5] * 14, [float("nan")] * 14):
            with self.assertRaisesRegex(ValueError, "mask"):
                self.make_wrapper(proprio_keep_override=mask)
        for key, shape in (("proprio_diff_std", (2, 14)), ("memory_diff_std", (3, 2, 8))):
            for val in (0., -1., float("nan"), float("inf")):
                with self.assertRaisesRegex(ValueError, "std"):
                    self.make_wrapper(stats_edit=lambda s: s.update({key: np.full(shape, val).tolist()}))
            with self.assertRaises(ValueError):
                self.make_wrapper(stats_edit=lambda s: s.update({key: [1.]}))
        model = TinyAction()
        model.output_proj = torch.nn.Linear(8, 13)
        with self.assertRaisesRegex(ValueError, "projection"):
            self.make_wrapper(model=model)

    def test_normalizers_reject_wrong_shapes_and_nonfinite(self):
        w = self.make_wrapper()
        for shape in ((2, 3, 13), (2, 2, 14), (3, 14), (2, 3, 14, 1)):
            with self.assertRaises(ValueError):
                w.normalize_state(torch.zeros(shape))
        for f, shape in ((w.normalize_state, (2, 3, 14)), (w.normalize_action, (2, 4, 14)),
                         (w.normalize_memory, (2, 3, 3, 8))):
            for value in (float("nan"), float("inf")):
                with self.assertRaises(ValueError):
                    f(torch.full(shape, value))
        with self.assertRaises(ValueError):
            w.normalize_action(torch.zeros(2, 4, 13))
        with self.assertRaises(ValueError):
            w.normalize_memory(torch.zeros(2, 3, 2, 8))

    def test_masked_padding_cannot_change_valid_loss_or_any_gradient(self):
        w, b = self.make_wrapper(), self.batch()
        mask = torch.tensor([[True, True, False, False], [True, False, False, False]])
        b["action_valid_mask"] = mask
        b["action_sequence"].requires_grad_()
        loss, info, out, grads = self.run_grad(w, b)
        input_grad = b["action_sequence"].grad.clone()
        self.assertTrue((input_grad[~mask] == 0).all())
        for value in (9999., float("nan")):
            changed = dict(b)
            changed["action_sequence"] = b["action_sequence"].detach().clone().masked_fill(~mask[..., None], value).requires_grad_()
            loss2, info2, out2, grads2 = self.run_grad(w, changed)
            torch.testing.assert_close(loss, loss2, rtol=0, atol=0)
            self.assertEqual(info, info2)
            torch.testing.assert_close(out["final_pred"][mask], out2["final_pred"][mask], rtol=0, atol=0)
            self.assert_grads_equal(grads, grads2)
            torch.testing.assert_close(input_grad, changed["action_sequence"].grad, rtol=0, atol=0)

    def test_exact_valid_element_denominator_and_no_mask_legacy(self):
        w, b = self.make_wrapper(), self.batch()
        mask = torch.tensor([[True, False, False, False], [True, True, True, False]])
        for valid in (None, mask):
            raw = b["action_sequence"] if valid is None else b["action_sequence"].masked_fill(~valid[..., None], 0.)
            x1 = w.normalize_action(raw)
            torch.manual_seed(99)
            x0 = torch.randn_like(x1)
            torch.manual_seed(99)
            _, info, out = w(b, {}, action_valid_mask=valid)
            err = out["final_pred"].float() - (x1 - x0)
            expected = err.square().mean() if valid is None else err[valid].square().sum() / (14 * valid.sum())
            self.assertEqual(info["mse"], expected.item())
        b["action_valid_mask"] = torch.ones(2, 4, dtype=torch.bool)
        masked = self.run_grad(w, b)
        b.pop("action_valid_mask")
        unmasked = self.run_grad(w, b)
        self.assertEqual(masked[1], unmasked[1])
        self.assert_grads_equal(masked[3], unmasked[3])

    def test_invalid_masks_and_action_shapes_fail_closed(self):
        w, b = self.make_wrapper(), self.batch()
        for name, shape in (("action_valid_mask", (2, 4)), ("future_valid_mask", (2,))):
            for mask in (torch.ones(shape), torch.ones(3, dtype=torch.bool), [True, False]):
                with self.assertRaisesRegex(ValueError, name):
                    w(b, {}, **{name: mask})
        with self.assertRaisesRegex(ValueError, "no valid actions"):
            w(b, {}, action_valid_mask=torch.zeros(2, 4, dtype=torch.bool))
        for shape in ((2, 3, 14), (2, 4, 13), (4, 14)):
            with self.assertRaisesRegex(ValueError, "action_sequence"):
                w({**b, "action_sequence": torch.zeros(shape)}, {})

    def test_future_mask_selects_only_its_own_samples(self):
        w, b = self.make_wrapper(), self.batch()
        b["future_valid_mask"] = torch.tensor([True, False])
        first = self.run_grad(w, b)
        pred = w.action_model.future_feat_decoder(first[2]["cond_tokens"])
        upstream, _ = load_upstream()
        expected = upstream.future_feature_cosine_loss(pred[:1], b["future_feats"][:1].reshape(1, 6, 4))
        self.assertEqual(first[1]["future"], expected.item())
        b["future_feats"][1].fill_(float("nan"))
        second = self.run_grad(w, b)
        self.assertEqual(first[1], second[1])
        self.assert_grads_equal(first[3], second[3])

    def test_empty_future_is_differentiable_zero_without_fake_phase(self):
        w, b = self.make_wrapper(), self.batch()
        b["future_valid_mask"] = torch.zeros(2, dtype=torch.bool)
        b.pop("future_feats")  # no target needed for an episode-end-only batch
        loss, info, _, grads = self.run_grad(w, b)
        self.assertTrue(torch.isfinite(loss))
        for name in ("future", "phase_ce", "gate_kl", "log_dist"):
            self.assertEqual(info[name], 0.)
        for part in ("weight", "bias"):
            grad = grads[f"action_model.future_feat_decoder.{part}"]
            self.assertTrue((grad == 0).all())
        self.assertTrue((b["phase"] == -1).all())
        self.assertTrue((b["active_hand"] == -1).all())
        b.pop("phase")
        self.assertEqual(self.run_grad(w, b)[1]["phase_ce"], 0.)

    def test_unknown_phase_not_silently_assigned(self):
        w, b = self.make_wrapper(), self.batch()
        b["phase"] = torch.tensor([9, -1])
        _, info, _, _ = self.run_grad(w, b)
        self.assertEqual(info["phase_ce"], 0.)
        b["phase"] = torch.tensor([0, 2])
        _, info, _, _ = self.run_grad(w, b)  # unknown hand / distance, not legitimate labels
        self.assertEqual(info["phase_ce"], 0.)
        b["active_hand"] = torch.tensor([0, 1])
        b["dist"] = torch.tensor([.3, .05])
        _, info, _, grads = self.run_grad(w, b)
        self.assertGreater(info["phase_ce"], 0.)
        self.assertGreater(grads["action_model.phase_head.weight"].abs().sum().item(), 0.)

    def test_bits_and_actual_floats_have_identical_values_and_gradients(self):
        w, b = self.make_wrapper(), self.batch()
        for key in ("feats", "future_feats"):
            b[key] = b[key].bfloat16().float()
        reference = self.run_grad(w, b)
        for dtype in (torch.int16, torch.uint16, torch.bfloat16, torch.float16, torch.float64):
            other = dict(b)
            for key in ("feats", "future_feats"):
                other[key] = b[key].bfloat16().view(dtype) if dtype in (torch.int16, torch.uint16) else b[key].to(dtype)
            result = self.run_grad(w, other)
            self.assertEqual(reference[1], result[1])
            self.assert_grads_equal(reference[3], result[3])
        h = torch.randn(3, 4).bfloat16()
        for dtype in (np.int16, np.uint16):
            bits = h.view(torch.int16).numpy().view(dtype)
            torch.testing.assert_close(bf16_from_bits(bits), h, rtol=0, atol=0)
        actual = np.array([1.125, -2.5], np.float32)
        np.testing.assert_array_equal(bf16_from_bits(actual).numpy(), actual)
        with self.assertRaises(ValueError):
            bf16_from_bits(torch.ones(3, dtype=torch.int32))

    def test_native_cpu_sampling_decodes_cache_bits(self):
        w, b = self.make_wrapper(), self.batch()
        features = [b["feats"][:, :, i].bfloat16() for i in range(3)]
        outputs = []
        for fs in (features, [f.view(torch.int16) for f in features], [f.float() for f in features]):
            a, _ = w.sample_actions(fs, b["state"], b["pool"], b["task_cond"],
                                    num_steps=2, generator=torch.Generator().manual_seed(3))
            self.assertEqual(a.shape, (2, 4, 14))
            self.assertTrue(torch.isfinite(a).all())
            outputs.append(a)
        for a in outputs[1:]:
            torch.testing.assert_close(a, outputs[0], rtol=0, atol=0)

    def test_retarget_resets_exactly_six_keys_including_same_shape_biases(self):
        for source in (TinyAction(61, 23), TinyAction(14, 14)):
            target = TinyAction()
            with torch.no_grad():
                for p in source.parameters():
                    p.fill_(17.)
                for p in target.parameters():
                    p.fill_(29.)
            sd = {k: v.clone() for k, v in source.state_dict().items()}
            notes = load_retargeted_v3_state(target, sd)
            self.assertEqual(set(notes["reset"]), RESET)
            self.assertEqual(set(notes["loaded"]), set(sd) - RESET)
            for key, value in target.state_dict().items():
                if key in RESET:
                    self.assertFalse((value == 17).any(), key)
                    self.assertFalse((value == 29).any(), key)
                else:
                    torch.testing.assert_close(value, sd[key], rtol=0, atol=0, msg=key)
                self.assertTrue((sd[key] == 17).all(), "source was mutated")
            self.assertTrue((target.gate_head.weight == 17).all())

    def test_retarget_rejects_all_other_differences_without_mutation(self):
        target, source = TinyAction(), TinyAction(61, 23)
        good = source.state_dict()
        bads = [dict(good, unknown=torch.zeros(1)),
                {k: v for k, v in good.items() if k != "gate_head.bias"},
                dict(good, **{"mem_proj.weight": torch.zeros(1)}),
                dict(good, **{"phase_head.bias": None})]
        for bad in bads:
            before = {k: v.clone() for k, v in target.state_dict().items()}
            with self.assertRaises(RuntimeError):
                load_retargeted_v3_state(target, bad)
            for k, v in target.state_dict().items():
                torch.testing.assert_close(v, before[k], rtol=0, atol=0)
        notes = load_retargeted_v3_state(target, {k: v for k, v in good.items() if k not in RESET})
        self.assertEqual(set(notes["reset"]), RESET)

    def test_context14_and_default61_reset_bounded_and_copied(self):
        for dim in (14, 61):
            ctx = V3Context([1, 3], [2], VIEWS, **({"state_dim": 14} if dim == 14 else {}))
            with self.assertRaises(RuntimeError):
                ctx.inputs()
            for t in range(8):
                state, pool = np.full(dim, t), np.full((3, 2, 4), t)
                ctx.push(state, pool)
                state[:] = -999
                s, p = ctx.inputs()
                np.testing.assert_array_equal(s[:, 0], [t, max(0, t-1), max(0, t-3), 0])
                np.testing.assert_array_equal(p[0, :, 0], [t, max(0, t-2), 0])
                self.assertEqual(s.shape, (4, dim))
                self.assertLessEqual(len(ctx.states), 4)
            with self.assertRaises(ValueError):
                ctx.push(np.zeros(dim + 1), np.zeros((3, 2, 4)))
            for pool in (np.array(0), np.zeros((2, 2, 4)), np.zeros((3, 4)), np.zeros((3, 0))):
                with self.assertRaises(ValueError):
                    ctx.push(np.zeros(dim), pool)
            with self.assertRaises(ValueError):
                ctx.push(np.full(dim, np.nan), np.zeros((3, 2, 4)))
            ctx.reset()
            ctx.push(np.full(dim, 100), np.full((3, 2, 4), 100))
            self.assertTrue((ctx.inputs()[0] == 100).all())
            self.assertTrue((ctx.inputs()[1] == 100).all())
        for dim in (0, -1, 14.5):
            with self.assertRaises(ValueError):
                V3Context([1], [1], VIEWS, state_dim=dim)

    def test_real_small_v3_native_forward_backward_and_retarget(self):
        kw = dict(hidden_dim=16, num_heads=4, depth=2, action_len=4, proprio_len=3,
                  num_registers=2, dino_feat_dims=(4, 4, 4), vlm_num_queries=2,
                  adapter_depth=1, task_cond_dim=8, fusion_mode="concat", use_future_feat=True,
                  future_feat_num_queries=2, future_feat_out_dim=4, future_feat_depth=1, future_feat_heads=4)
        cls = v3_model_class()
        source = cls(3, 2, memory_dim=8, phase_hidden=16, action_dim=23, proprio_dim=61, **kw)
        target = cls(3, 2, memory_dim=8, phase_hidden=16, action_dim=14, proprio_dim=14, **kw)
        notes = load_retargeted_v3_state(target, source.state_dict())
        self.assertEqual(set(notes["reset"]), RESET)
        for key, value in target.state_dict().items():
            if key not in RESET:
                torch.testing.assert_close(value, source.state_dict()[key], rtol=0, atol=0)
        w, b = self.make_wrapper(model=target), self.batch()
        target.eval()
        b["feats"] = b["feats"].bfloat16().view(torch.int16)
        b["action_valid_mask"] = torch.tensor([[True, True, False, False], [True, True, True, False]])
        b["future_valid_mask"] = torch.tensor([True, False])
        loss, _, _, grads = self.run_grad(w, b)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(torch.isfinite(g).all() for g in grads.values()))
        self.assertGreater(grads["action_model.action_proj.weight"].abs().sum().item(), 0.)
        self.assertGreater(grads["action_model.output_proj.weight"].abs().sum().item(), 0.)


if __name__ == "__main__":
    unittest.main(verbosity=2)
