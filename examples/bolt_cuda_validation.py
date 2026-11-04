"""GPU validation for the MLBricks BOLT accelerated state-memory implementation.

Run after installing MLBricks with optional CUDA recurrence extensions:

    MLBRICKS_BUILD_BOLT_ACCEL=1 pip install -v .
    python examples/bolt_cuda_validation.py

This compares matched weights, forward outputs, input gradients and CUDA
latency. It is a smoke test, not a full model-quality benchmark.
"""
from dataclasses import replace
import time
import torch
from mlbricks import BoltConfig, BoltModel
from mlbricks.bolt import accelerated


def run(profile, batch=2, sequence=256, *, state_memory=True):
    cfg = BoltConfig(
        vocab_size=512, context=sequence, layers=2, dim=384, heads=12,
        profile=profile, state_memory=state_memory,
        device='cuda', precision='fp16',
        backend='pytorch', projection='separate',
    )
    torch.manual_seed(1729)
    reference = BoltModel(cfg).train()
    optimized = BoltModel(replace(cfg, projection='packed', backend='auto')).train()
    optimized.load_state_dict(reference.state_dict(), strict=True)
    for block in optimized.blocks:
        assert (block.attn._modern.avq is not None) is state_memory
    if not state_memory:
        assert not any('.avq.' in key for key in optimized.state_dict())
    tokens = torch.randint(0, 512, (batch, sequence), device='cuda')

    logits_ref, loss_ref = reference(tokens, targets=tokens)
    logits_opt, loss_opt = optimized(tokens, targets=tokens)
    forward_error = float((logits_ref.float()-logits_opt.float()).abs().max())
    loss_ref.backward()
    loss_opt.backward()
    grad_ref = reference.token_embedding.weight.grad
    grad_opt = optimized.token_embedding.weight.grad
    gradient_error = float((grad_ref-grad_opt).abs().max())
    assert torch.isfinite(logits_opt).all() and torch.isfinite(grad_opt).all()
    assert forward_error < 0.06, f'forward numerical mismatch: {forward_error}'
    assert gradient_error < 0.06, f'gradient numerical mismatch: {gradient_error}'

    def measure(model):
        optim = torch.optim.AdamW(model.parameters(), lr=3e-4)
        for _ in range(5):
            optim.zero_grad(set_to_none=True)
            _, loss = model(tokens, targets=tokens)
            loss.backward(); optim.step()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(20):
            optim.zero_grad(set_to_none=True)
            _, loss = model(tokens, targets=tokens)
            loss.backward(); optim.step()
        torch.cuda.synchronize()
        return 1000*(time.perf_counter()-start)/20

    ms_ref=measure(reference)
    ms_opt=measure(optimized)
    print(profile, dict(
        state_memory=state_memory,
        output_max_abs=forward_error, gradient_max_abs=gradient_error,
        reference_ms=round(ms_ref,3), optimized_ms=round(ms_opt,3),
        speedup=round(ms_ref/ms_opt,3),
    ))


if __name__ == '__main__':
    if not torch.cuda.is_available():
        raise SystemExit('CUDA GPU required to run the native-kernel smoke test')
    print('CUDA device:', torch.cuda.get_device_name(),
          'Accelerated state kernels:',accelerated.available())
    if not accelerated.available():
        print('Native kernels not installed; reference parity works but no accelerated training expected.')
    for profile in ('compact','standard'):
        for state_memory in (True, False):
            run(profile, state_memory=state_memory)
