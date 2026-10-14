"""Instance-scoped action-loss weighting; frozen v3/Precision sources stay unchanged.

Only locally verified DAgger action labels get weight two. Future-feature and
IDM objectives, model parameters, masks, RNG and deployment remain unchanged.
"""
from types import MethodType
import numpy as np
import torch
import torch.nn.functional as F
from harness.wam_v3 import WamV3Wrapper, bf16_from_bits, gate_prior, PHASES
from harness.wam_behavior import load_upstream

POLICY = dict(schema='expert-recovery-action-weight-v16', original_action_weight=1.0,
    verified_recovery_action_weight=2.0, reduction='sum(weight * valid squared error) / sum(weight * valid action dimensions)',
    applies_to='verified DAgger action labels only; both retained legacy and segmented corrections',
    sampler_unchanged='three original samples plus one verified correction',
    auxiliary_losses_unchanged=True, invalid_and_policy_actions_weight=0.0)


def mixed_weighted_samples(original, corrections, rng):
    from harness.zhuhai_dagger_segmented_training_v7 import mixed_samples
    samples = mixed_samples(original, corrections, rng)
    if len(samples) != 4:
        raise ValueError('Weighted sampler requires the unchanged three-plus-one batch')
    return [dict(sample, action_loss_weight=np.float32(1.0 if i < 3 else 2.0))
            for i, sample in enumerate(samples)]


def weighted_action_mse(prediction, target, valid, sample_weights):
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError('Action predictions and targets must have equal B x T x D shapes')
    b, t, d = prediction.shape
    weights = sample_weights.to(device=prediction.device, dtype=torch.float32).detach()
    if weights.shape != (b,) or not torch.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError('Positive finite per-sample action weights required')
    if valid is None:
        valid = torch.ones((b, t), dtype=torch.bool, device=prediction.device)
    elif valid.dtype != torch.bool or valid.shape != (b, t):
        raise ValueError('Boolean B x T action validity mask required')
    else:
        valid = valid.to(prediction.device)
    if not valid.any():
        raise ValueError('No valid expert action labels')
    # Select before arithmetic: invalid targets may contain arbitrary padding.
    error = prediction.float()[valid] - target.float()[valid]
    selected = weights[:, None].expand(b, t)[valid]
    loss = (error.square().sum(-1) * selected).sum() / (d * selected.sum())
    counts = valid.sum(-1).float()
    mass = counts * weights
    info = dict(action_weighted=True, action_sample_weights=weights.cpu().tolist(),
        action_valid_counts=counts.to(dtype=torch.int64).cpu().tolist(),
        recovery_action_loss_mass_fraction=float(mass[weights > 1].sum() / mass.sum()))
    return loss, info


def install_recovery_weighting(wrapper):
    base = wrapper.base
    if not isinstance(base, WamV3Wrapper) or getattr(base, '_recovery_weighting_installed', False):
        raise ValueError('Install once on a freshly constructed Precision/H-JEPA wrapper')
    base.forward = MethodType(weighted_forward, base)
    base._recovery_weighting_installed = True
    return wrapper


def weighted_forward(self, batch, lambdas, action_valid_mask=None, future_valid_mask=None):
    """Optional masks may be supplied as arguments or batch fields; absent masks retain legacy reductions.

    Canonicalize invalid raw actions BEFORE noising: attention can mix action tokens, so
    masking only the loss would leak arbitrary padding into predictions of valid tokens.
    Future masks select samples, never borrow targets from another episode.
    """
    if 'action_loss_weight' not in batch:
        return WamV3Wrapper.forward(self, batch, lambdas, action_valid_mask, future_valid_mask)
    dev = self.device
    raw = batch["action_sequence"].to(dev, torch.float32)
    if raw.ndim != 3 or raw.shape[1:] != (self.action_model.action_len, self.action_dim):
        raise ValueError(f"action_sequence must have shape (B, {self.action_model.action_len}, {self.action_dim})")
    B = raw.shape[0]
    action_valid_mask = self._valid_mask(
        batch.get("action_valid_mask") if action_valid_mask is None else action_valid_mask,
        raw.shape[:2], dev, "action_valid_mask")
    future_valid_mask = self._valid_mask(
        batch.get("future_valid_mask") if future_valid_mask is None else future_valid_mask,
        (B,), dev, "future_valid_mask")
    if action_valid_mask is not None:
        if not action_valid_mask.any():
            raise ValueError("action_valid_mask has no valid actions")
        raw = raw.masked_fill(~action_valid_mask[..., None], 0.0)
    x1 = self.normalize_action(raw)
    feats_all = bf16_from_bits(batch["feats"].to(dev, non_blocking=True))          # (B, V, L, T, D)
    if dev.type == "cpu":
        feats_all = feats_all.float()
    feats = [feats_all[:, :, i] for i in range(feats_all.shape[2])]
    qpos = self.normalize_state(batch["state"].to(dev, torch.float32))
    memory = self.normalize_memory(batch["pool"].to(dev, torch.float32))
    tc = batch["task_cond"].to(dev, torch.float32)
    if any(x.shape[0] != B for x in (feats_all, qpos, memory, tc)):
        raise ValueError("all model inputs must have the same batch size")
    x0 = torch.randn_like(x1)
    t = torch.sigmoid(torch.randn(B, device=dev) * float(self.train_cfg.get("time_sigma", 1.0)) + float(self.train_cfg.get("time_mu", 0.0)))
    x_t = (1 - t.view(B, 1, 1)) * x0 + t.view(B, 1, 1) * x1
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
        out = self.action_model(t, noisy_actions=x_t, qpos_history=qpos, dino_features_list=feats, task_cond=tc, memory=memory)
    if out["final_pred"].shape != x1.shape:
        raise ValueError("action prediction shape does not match action_sequence")
    loss_mse, weight_info = weighted_action_mse(
        out['final_pred'], x1 - x0, action_valid_mask, batch['action_loss_weight'])
    losses = {"mse": loss_mse}
    if self.action_model.use_future_feat:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            pred = self.action_model.future_feat_decoder(out["cond_tokens"])
        if future_valid_mask is not None and not future_valid_mask.any():
            losses["future"] = pred.float().sum() * 0.0
        else:
            fut = bf16_from_bits(batch["future_feats"].to(dev, non_blocking=True))   # (B, V, P, D)
            if fut.ndim != 4 or fut.shape[:2] != (B, len(self.views)):
                raise ValueError("future_feats must have shape (B, V, P, D)")
            fut = fut.reshape(B, -1, fut.shape[-1])
            if pred.shape != fut.shape:
                raise ValueError("future feature prediction and target shapes must match")
            if future_valid_mask is not None:
                pred, fut = pred[future_valid_mask], fut[future_valid_mask]
            if not torch.isfinite(fut).all():
                raise ValueError("valid future features must be finite")
            vla, _ = load_upstream()
            losses["future"] = vla.future_feature_cosine_loss(pred, fut)
    phase = batch.get("phase", torch.full((B,), -1, dtype=torch.long)).to(dev)
    if phase.shape != (B,):
        raise ValueError("phase must have shape (B,)")
    valid = (phase >= 0) & (phase < len(PHASES)) & (phase == phase.long())
    gate = out["gate"]
    if valid.any():
        hand = batch["active_hand"].to(dev)
        dist = batch["dist"].to(dev, torch.float32)
        if hand.shape != (B,) or dist.shape != (B,):
            raise ValueError("active_hand and dist must have shape (B,)")
        valid = valid & ((hand == 0) | (hand == 1)) & torch.isfinite(dist) & (dist >= 0)
    if valid.any():
        labelled_phase = torch.where(valid, phase, -1).long()
        prior = torch.from_numpy(gate_prior(labelled_phase.cpu().numpy(), hand.cpu().numpy(), self.views)).to(dev)
        kl = (prior * (torch.log(prior.clamp_min(1e-8)) - torch.log(gate.clamp_min(1e-8)))).sum(-1)
        losses["gate_kl"] = kl[valid].mean()
        losses["phase_ce"] = F.cross_entropy(out["phase_logits"][valid], phase[valid].long())
        losses["log_dist"] = F.smooth_l1_loss(out["log_dist"][valid], torch.log(dist[valid] + 0.02))
    else:
        zero = loss_mse * 0
        losses.update(gate_kl=zero, phase_ce=zero, log_dist=zero)
    total = loss_mse + sum(float(lambdas.get(k, 0.0)) * v for k, v in losses.items() if k != "mse")
    info = {k: float(v.detach()) for k, v in losses.items()}
    info.update(weight_info)
    info["gate_mean"] = gate.detach().mean(0).tolist()
    return total, info, out
