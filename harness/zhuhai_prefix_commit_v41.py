"""Hard prefix commitment (flow inpainting) for the unchanged native16 flow policy.

Candidate v41 mechanism. Synchronous (simulation does not advance during
inference, d=0). The last chunk's unexecuted indices 8:32 are PINNED as the new
chunk's indices 0:24 at every Euler step in normalized action space, exactly
along the rectified-flow interpolant x_tau = tau*a + (1-tau)*eps; only indices
24:32 are generated fresh, conditioned through the model input on the pinned
prefix. Consequences, stated as falsifiable predictions rather than claims:
 * the raw cross-chunk overlap mismatch becomes exactly zero by construction;
 * any discrete gripper event the model placed at chunk index j is executed
   after ceil(j/8) replans instead of being regenerated and postponed;
 * stale commitments are the price: an action executed at replan k was first
   generated at replan k-3 (2.4s earlier) and never revised.
No weights change, no task-state oracle, no action averaging, no training.
"""
from pathlib import Path
import hashlib
import math
import time
import numpy as np
import torch
from harness.wam_v3 import bf16_from_bits

HORIZON, EXECUTION, DENOISE_STEPS = 32, 8, 10
PIN = HORIZON - EXECUTION
VERSION = 'precision-v2-hard-prefix-commit-c1-fine250-v41'
SOURCES = ['https://arxiv.org/html/2506.07339v1', 'https://sjtu-deng-lab.github.io/Streaming-WAM/']


def pinned(target, noise, tau):
    """Interpolant of the committed prefix at flow time tau (tau=0 noise, tau=1 data)."""
    if not 0 <= tau <= 1:
        raise ValueError('flow tau outside [0,1]')
    return tau*target + (1-tau)*noise


def integrate(velocity, x0, target=None, pin=PIN, num_steps=DENOISE_STEPS):
    """Euler flow integration with replacement inpainting of the first `pin` rows.

    target: (B,HORIZON,D) normalized actions whose first `pin` rows are committed;
    rows >= pin are ignored. With target=None this is exactly the plain sampler.
    """
    if num_steps != DENOISE_STEPS:
        raise ValueError('bound ten-step integration required')
    if target is not None and (target.shape != x0.shape or not 0 < pin < x0.shape[1]):
        raise ValueError('committed prefix must match chunk geometry')
    x = x0.clone()
    traces = []
    for i in range(num_steps):
        tau = i/num_steps
        if target is not None:
            x[:, :pin] = pinned(target[:, :pin], x0[:, :pin], tau)
        with torch.no_grad():
            v = velocity(x, tau).float()
        if not torch.isfinite(v).all():
            raise ValueError('nonfinite flow velocity; never silently fall back')
        x = (x + v/num_steps).detach()
        traces.append(dict(tau=tau, flow_rms=float(v.square().mean().sqrt()),
            suffix_flow_rms=float(v[:, pin:].square().mean().sqrt()) if target is not None else None,
            pinned_prefix_flow_rms=float(v[:, :pin].square().mean().sqrt()) if target is not None else None))
    if target is not None:
        x[:, :pin] = target[:, :pin]
    return x, traces


class PrefixCommitSampler:
    """Drop-in replacement for the frozen sampler: identical first chunk, pinned later chunks."""

    def __init__(self, wrapper, condition, verify_owner):
        self.wrapper = wrapper
        self.condition = condition
        self.verify_owner = verify_owner
        self.previous = None
        self.previous_step = None
        self.seed = None
        self.last_diagnostic = None
        wrapper.eval().requires_grad_(False)

    def __call__(self, feats, inputs, seed, step, event_state):
        self.verify_owner()
        start = time.monotonic()
        if step == 0:
            self.previous = None
            self.previous_step = None
            self.seed = seed
        if self.seed != seed or (self.previous_step is not None and step != self.previous_step+EXECUTION):
            raise ValueError('commitment history is not this episode or not exec8 aligned')
        wrapper = self.wrapper
        device = wrapper.device
        state, pool = inputs
        seed_value = int(hashlib.sha256(f'{seed}:{step}'.encode()).hexdigest()[:15], 16)
        generator = torch.Generator(device=device).manual_seed(seed_value)
        if self.previous is None:
            # First chunk is exactly the old sampler, including attention backend.
            with torch.inference_mode():
                actions, _ = wrapper.sample_actions(feats, torch.from_numpy(state)[None],
                    torch.from_numpy(pool)[None], torch.from_numpy(self.condition)[None],
                    num_steps=DENOISE_STEPS, generator=generator,
                    event_state=torch.from_numpy(event_state.copy())[None])
            value = actions[0].float().cpu().numpy()
            traces = []
            noise_sha = None
            boundary = None
        else:
            features = [bf16_from_bits(f.to(device)).detach().clone() for f in feats]
            if device.type == 'cpu':
                features = [f.float() for f in features]
            qpos = wrapper.normalize_state(torch.from_numpy(state)[None].to(device, dtype=torch.float32)).detach()
            memory = wrapper.normalize_memory(torch.from_numpy(pool)[None].to(device, dtype=torch.float32)).detach()
            condition = torch.from_numpy(self.condition)[None].to(device, dtype=torch.float32)
            clock = torch.from_numpy(event_state.copy())[None].to(device, dtype=torch.float32)
            previous = torch.from_numpy(self.previous)[None].to(device, dtype=torch.float32)
            target = torch.zeros((1, HORIZON, 16), device=device)
            target[:, :PIN] = wrapper.normalize_action(previous[:, EXECUTION:])
            x0 = torch.randn((1, HORIZON, 16), device=device, generator=generator)
            noise_sha = hashlib.sha256(x0.cpu().numpy().tobytes()).hexdigest()

            def velocity(noisy, tau):
                t = torch.full((1,), tau, device=device)
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                    return wrapper.action_model(t, noisy_actions=noisy, qpos_history=qpos,
                        dino_features_list=features, task_cond=condition, memory=memory,
                        event_state=clock)['final_pred']
            with torch.inference_mode():
                x, traces = integrate(velocity, x0, target)
            value = wrapper.denormalize_action(x)[0].float().cpu().numpy()
            if not np.array_equal(value[:PIN], self.previous[EXECUTION:]) and \
                    not np.allclose(value[:PIN], self.previous[EXECUTION:], rtol=0, atol=2e-6):
                raise ValueError('committed prefix was not reproduced after denormalization')
            # Force bit-exact native commitment: the pinned rows ARE the previous actions.
            value[:PIN] = self.previous[EXECUTION:]
            joint = value[:, [*range(7), *range(8, 15)]]
            boundary = dict(position_jump_rad=float(np.max(np.abs(joint[PIN]-joint[PIN-1]))),
                within_prefix_step_rad=float(np.max(np.abs(np.diff(joint[:PIN], axis=0)))),
                within_suffix_step_rad=float(np.max(np.abs(np.diff(joint[PIN:], axis=0)))) )
        if value.shape != (HORIZON, 16) or not np.isfinite(value).all():
            raise ValueError('invalid committed native16 chunk')
        self.last_diagnostic = dict(version=VERSION, sources=SOURCES, previous_query_step=self.previous_step,
            execution_horizon=EXECUTION, committed_actions=0 if self.previous is None else PIN,
            physical_inference_delay_actions=0, commitment='hard_replacement_inpainting_normalized_space',
            previous_unexecuted_native_chunk=None if self.previous is None else self.previous[EXECUTION:].tolist(),
            denoising_trace=traces, initial_noise_sha256=noise_sha, query_seed=seed_value,
            suffix_boundary=boundary, inference_seconds=time.monotonic()-start,
            model_weights_updated=False, same_time_overlap=True, stage_progress_or_grasp_oracle=False,
            fresh_rows=[0, HORIZON] if self.previous is None else [PIN, HORIZON])
        self.previous = value.copy()
        self.previous_step = step
        return value


def implementation_sha256():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
