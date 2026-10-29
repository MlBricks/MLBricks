# BOLT v0.2 — Experimental optimized inference patch

This opt-in patch changes **modern** BOLT only; it does not change SOUP, legacy
BOLT, its training path, or `state_dict` parameter names. It uses **PyTorch only**;
there is no Triton dependency and no `torch.compile` requirement.

## Changes

- Packed Q/C/G linear inference projection, Q scale **P folded into Q's weight
  and bias**, with automatic invalidation on changes to any source parameter.
- Fold AVQ's RMSNorm learned gain and `gamma` into its output projection for
  inference only. The input-dependent RMS normalization and read gate remain.
- Use the existing `BoltPresetCache` fixed-capacity KV cache during
  `BoltModel.generate()`; in optimized mode update AVQ state *in-place*.
- Cache rotated RoPE keys across subsequent fixed-buffer decode steps instead
  of re-rotating the full history each time.
- Optional **experimental** CUDA Graph capture for whole-model greedy decode,
  using PyTorch SDPA with a masked, full-capacity cache and a device-side
  position. No custom CUDA/Triton kernels.

## Usage: PyTorch-only eager mode (CPU or GPU)

```python
import torch
from mlbricks import BoltModel

model = BoltModel(vocab_size=2048, dim=384, heads=12, layers=6,
                  context=2048, bolt_mode='quality', bolt_cache='tight',
                  avq_dim=198, backend='pytorch').eval()
model.enable_bolt_optimized_inference(True)
with torch.inference_mode():
    out = model.generate(torch.randint(0, 2048, (1, 32)),
                         32, temperature=0)
# To restore exact old code route:
model.enable_bolt_optimized_inference(False)
```

At single-layer scope: `Bolt(...).enable_optimized_inference(True)`.
Training keeps the original unpacked projection and explicit P scaling, even
when inference optimization has been toggled on. Packing/folding outputs may
vary slightly from eager PyTorch due to FP rounding.

## Usage: CUDA Graph greedy decode (experimental)

```python
model = BoltModel(vocab_size=2048, dim=384, heads=12, layers=6,
                  context=2048, bolt_mode='quality', bolt_cache='tight',
                  avq_dim=198, backend='pytorch', position='none',
                  dropout=0).cuda().half().eval()
model.enable_bolt_optimized_inference(True)
prompt = torch.randint(0, 2048, (1, 512), device='cuda')
with torch.inference_mode():
    session = model.prepare_cuda_graph_generation(prompt, 64)
    output_ids = session.generate()  # [1, 576]
```

The graph generator is **one-shot** and supports modern BOLT-16/32 with
`backend='pytorch'`, standard residual/FFN, no positional encoding, dropout=0,
fixed sequence length and greedy token selection. A new session must be made
for every prompt; models and weights must not change after capture. There is
no guarantee that graph replay is faster for all model sizes/context lengths.
**CUDA Graph capture is untested in CPU-only CI**, and requires GPU validation
for kernels, memory use, concurrency and all-token numerical parity.

Graph capture and prompt prefill cost are excluded from replay-only timing;
include them in any real end-to-end benchmark. The masked full-capacity SDPA
attention may become expensive for long contexts. Optimized eager decode
supports RoPE, but graph mode currently does not.

## Tests

```bash
python -m pytest -q tests/test_bolt_optimized_inference.py tests/test_bolt_presets.py
python examples/bolt_optimized_inference.py             # CPU or CUDA
python examples/bolt_optimized_inference.py --graph     # CUDA GPU only
```

This is an experimental feature. Do not promote to default until GPU
correctness and performance have been reproduced on target hardware.
