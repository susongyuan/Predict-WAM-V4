#!/usr/bin/env python3
"""Measure full-size synthetic training throughput without changing run artifacts."""
import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from harness import zhuhai_precision_v4_future16 as future
from harness import zhuhai_precision_v2_training as v2
from scripts import train_zhuhai_wam_precision_v4 as trainer
from scripts.check_precision_v4_runtime import synthetic_batch
from scripts import gpu_lock

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--gpu-lock-owner', required=True)
    parser.add_argument('--batches', type=int, nargs='+', default=[4, 8, 12, 16, 20, 24, 32])
    parser.add_argument('--activation-checkpointing', action='store_true')
    parser.add_argument('--verify-checkpoint-gradients', action='store_true')
    args = parser.parse_args()
    assert gpu_lock.owner() == args.gpu_lock_owner
    assert not args.out.exists()
    torch.set_num_threads(8)
    torch.cuda.set_per_process_memory_fraction(.95)
    future.install()
    contract = v2.read_json(args.training / 'manifest.json')['contract']
    v2.verify_code_hashes(contract['source_files'])
    saved = trainer._load_resume_path(args.training / 'checkpoint_last.pt', contract)
    cfg = contract['config']
    wrapper, _ = future.build_wrapper(cfg, contract['statistics'], variant=contract['hjepa']['variant'],
        idm_statistics=contract['idm_statistics'], device='cuda', state_dict=saved['model_state_dict'])
    checkpointed_layers = []
    schedule = contract['v4_training']['schedule']
    optimizer = trainer.make_optimizer(wrapper.action_model, cfg['training'], schedule)
    optimizer.load_state_dict(saved['optimizer_state_dict'])
    trainer.configure_phase(optimizer, 'joint', saved['updates'], schedule)
    # Keep the same real frozen encoder resident as production training.
    from harness.wam_behavior import load_upstream
    _, runner = load_upstream()
    encoder, _, _, _ = runner.ModelFactory.create_vision_encoder(
        str(v2.resolve(cfg['model']['vision_encoder']['checkpoint_path'])),
        dtype=torch.bfloat16, device='cuda')
    encoder.eval().requires_grad_(False)
    single = synthetic_batch(wrapper, contract['conditions'][v2.TASK])
    single['future_feats_16'] = single.pop('future_feats_8')
    single['future_valid_mask_16'] = single.pop('future_valid_mask_8')
    gradient_validation = None
    if args.verify_checkpoint_gradients:
        assert args.activation_checkpointing
        wrapper.train()
        optimizer.zero_grad(set_to_none=True)
        cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
        loss, _, outputs = wrapper(single, cfg['training']['auxiliary_loss'])
        loss.backward()
        original_loss = float(loss.detach())
        reference_outputs = {key: outputs[key].detach().cpu() for key in ('final_pred','cond_tokens','chunk_vtt_prediction')}
        gradients = {name: p.grad.detach().cpu() if p.grad is not None else None
                     for name,p in wrapper.action_model.named_parameters()}
        cpu_after, cuda_after = torch.get_rng_state(), torch.cuda.get_rng_state()
        del loss, outputs
        optimizer.zero_grad(set_to_none=True)
        torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state(cuda_rng)
        control_loss, _, control_outputs = wrapper(single, cfg['training']['auxiliary_loss'])
        control_loss.backward()
        assert float(control_loss.detach()) == original_loss
        repeat_maximum_difference = 0.
        repeat_error_square, repeat_reference_square = 0., 0.
        for name,p in wrapper.action_model.named_parameters():
            if gradients[name] is not None:
                difference = p.grad.detach().cpu()-gradients[name]
                repeat_error_square += float(difference.square().sum())
                repeat_reference_square += float(gradients[name].square().sum())
                repeat_maximum_difference = max(repeat_maximum_difference,
                    float((p.grad.detach().cpu()-gradients[name]).abs().max()))
        del control_loss, control_outputs
        optimizer.zero_grad(set_to_none=True)
    if args.activation_checkpointing:
        from scripts.precision_activation_checkpoint import enable
        checkpointed_layers = enable(wrapper.action_model)
    if args.verify_checkpoint_gradients:
        torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state(cuda_rng)
        loss, _, outputs = wrapper(single, cfg['training']['auxiliary_loss'])
        loss.backward()
        assert float(loss.detach()) == original_loss
        assert torch.equal(torch.get_rng_state(),cpu_after) and torch.equal(torch.cuda.get_rng_state(),cuda_after)
        maximum_difference = 0.
        error_square, reference_square = 0., 0.
        strict_mismatches, bf16_mismatches = [], []
        checked = 0
        for name,p in wrapper.action_model.named_parameters():
            expected = gradients[name]
            assert (p.grad is None) == (expected is None), name
            if expected is not None:
                actual = p.grad.detach().cpu()
                difference = actual-expected
                error_square += float(difference.square().sum())
                reference_square += float(expected.square().sum())
                if not torch.allclose(actual,expected,rtol=1e-4,atol=2e-6):
                    strict_mismatches.append(name)
                if not torch.allclose(actual,expected,rtol=.02,atol=2e-5):
                    bf16_mismatches.append(name)
                maximum_difference = max(maximum_difference,float((actual-expected).abs().max()))
                checked += 1
        output_exact = all(torch.equal(outputs[key].detach().cpu(),value) for key,value in reference_outputs.items())
        relative_l2 = (error_square/max(reference_square,1e-30))**.5
        repeat_relative_l2 = (repeat_error_square/max(repeat_reference_square,1e-30))**.5
        admitted = (output_exact and relative_l2 <= max(.01,2*repeat_relative_l2)
                    and maximum_difference <= max(2e-6,2*repeat_maximum_difference))
        gradient_validation = dict(status='passed' if admitted else 'failed', parameter_gradients_checked=checked,
            maximum_absolute_difference=maximum_difference, loss_exact=True, cpu_and_cuda_rng_exact=True,
            selected_forward_outputs_exact=output_exact, gradient_relative_l2=relative_l2,
            original_repeat_maximum_difference=repeat_maximum_difference,
            original_repeat_gradient_relative_l2=repeat_relative_l2,
            strict_fp32_tolerance_mismatches=strict_mismatches,bf16_tolerance_mismatches=bf16_mismatches,
            admission='forward/loss/RNG exact; gradient relative L2 <=max(1%,2x original-repeat error), max absolute <=max(2e-6,2x original-repeat error); full pointwise mismatch diagnostics retained; no bitwise-gradient claim',
            state_dict_keys_unchanged=True, inputs='full-size synthetic batch1; real trained model')
        with args.out.with_suffix('.gradient.json').open('x') as stream:
            json.dump(gradient_validation,stream,indent=2)
        assert gradient_validation['status']=='passed', gradient_validation
        del gradients, loss, outputs, actual, expected
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        print(json.dumps(dict(gradient_validation=gradient_validation)),flush=True)
    rows, synthetic_updates = [], 0
    samples, stop = [], threading.Event()
    def measure_gpu():
        while not stop.wait(.5):
            result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used',
                '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5)
            try:
                util, memory = map(int, result.stdout.strip().split(','))
                samples.append(dict(time=time.monotonic(), utilization=util, memory_mib=memory))
            except ValueError:
                pass
    monitor = threading.Thread(target=measure_gpu, daemon=True)
    monitor.start()
    candidates = args.batches
    wrapper.train()
    try:
        for size in candidates:
            batch = {key: value.repeat((size,) + (1,) * (value.ndim-1)) for key, value in single.items()}
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            durations = []
            begin = time.monotonic()
            try:
                for iteration in range(7):
                    optimizer.zero_grad(set_to_none=True)
                    torch.cuda.synchronize()
                    tick = time.monotonic()
                    loss, info, outputs = wrapper(batch, cfg['training']['auxiliary_loss'])
                    assert torch.isfinite(loss)
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(
                        [p for group in optimizer.param_groups for p in group['params'] if p.requires_grad],
                        cfg['training']['grad_clip_norm'])
                    assert torch.isfinite(norm)
                    optimizer.step()
                    synthetic_updates += 1
                    torch.cuda.synchronize()
                    if iteration >= 2:
                        durations.append(time.monotonic() - tick)
                    del loss, info, outputs
                optimizer.zero_grad(set_to_none=True)
                free, total = torch.cuda.mem_get_info()
                avg = sum(durations) / len(durations)
                measured = [s['utilization'] for s in samples if begin <= s['time'] <= time.monotonic()]
                row = dict(batch_size=size, status='passed', seconds_per_update=avg,
                    samples_per_second=size/avg, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(), free_bytes_after_update=free,
                    mean_gpu_utilization=sum(measured)/len(measured) if measured else None)
                rows.append(row)
                print(json.dumps(row), flush=True)
                if free < 768 * 1024**2:
                    break
            except torch.cuda.OutOfMemoryError as exc:
                row = dict(batch_size=size, status='oom', error=str(exc),
                    peak_allocated_bytes=torch.cuda.max_memory_allocated())
                rows.append(row)
                print(json.dumps(row), flush=True)
                optimizer.zero_grad(set_to_none=True)
                gc.collect(); torch.cuda.empty_cache()
                break
            finally:
                del batch
    finally:
        stop.set(); monitor.join(timeout=6)
    passing = [r for r in rows if r['status']=='passed' and r['free_bytes_after_update'] >= 768*1024**2]
    # Choose the largest stable batch whose throughput is within 5% of the fastest.
    fastest = max((r['samples_per_second'] for r in passing), default=0.)
    selected = max((r for r in passing if r['samples_per_second'] >= .95*fastest), key=lambda r:r['batch_size']) if passing else None
    report = dict(status='passed' if selected else 'no_stable_candidate', results=rows,
        selected_batch_size=selected['batch_size'] if selected else None,
        activation_checkpointing=args.activation_checkpointing, checkpointed_layers=checkpointed_layers,
        gradient_validation=gradient_validation,
        gpu_samples=samples, threads=8, starting_step=saved['step'],
        synthetic_optimizer_updates=synthetic_updates, formal_optimizer_updates=0,
        original_checkpoint_unchanged=True, inputs='synthetic full-size tensors; actual model/optimizer/encoder',
        includes_real_data_io=False, selection='largest passing batch within 5% of maximum throughput; >=768MiB free')
    with args.out.open('x') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps({k:v for k,v in report.items() if k not in ('gpu_samples','results')}), flush=True)

if __name__ == '__main__':
    main()
