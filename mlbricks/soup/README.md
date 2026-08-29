# SOUP

SOUP is an MLBricks architecture package.

## API

```python
from mlbricks import soup

model = soup(
    dim=512,
    width=1116,
    depth=2,
    mixer="esa",
    ffn="saffn",
    backend="auto",
    precision="fp16",
)
```

SOUP always includes its built-in Observer State Memory and SOUP Fusion blocks. They are not public interchangeable slots.

## Training fast path

The v0.1.0a1 training optimizations are preserved in v0.1.0a3:

- repeated tensor-contract checks are kept out of `forward()` and available through `model.validate(example)`;
- equal-width models skip registered `Identity` state bridges entirely;
- non-uniform widths use a construction-resolved bridged path;
- custom FFN calling convention is resolved once at construction instead of branching every forward.

These are execution-path optimizations only; SOUP equations, trainable parameters, and checkpoint keys are unchanged.

## Fast recurrent generation

The training and normal `forward()` path are unchanged. For generation, move the model to its final device, switch to eval mode, then prepare the inference-only packed plan once:

```python
model = model.cuda().eval()
model.prepare_generation()

prompt_hidden = torch.randn(batch, prompt_tokens, 512, device="cuda", dtype=torch.float16)
prompt_out, cache = model.prefill(prompt_hidden)

next_hidden = torch.randn(batch, 1, 512, device="cuda", dtype=torch.float16)
next_out, cache = model.decode_step(next_hidden, cache)
```

`prepare_generation()`:

- packs ESA `qgv` and SAFFN `x_proj` into one inference GEMM per layer;
- caches depth projections and scalar transition/residual constants;
- removes mathematically-zero first-layer state-projection work;
- preserves the original model parameters and checkpoint keys;
- adds no trainable parameters.

For SOUP it also packs Observer `state_write + write_gate` and `key + value` projections.

If a custom mixer/FFN is not compatible with the optimized plan, use recurrent `prefill/decode_step` without calling `prepare_generation()`.

### Recommended compilation

Compile the surrounding one-token generation step after `prepare_generation()`. For PyTorch/Inductor, a safe starting point is `fullgraph=True`, `dynamic=False`, and `options={"shape_padding": True}`. Avoid `mode="reduce-overhead"` when returned recurrent cache tensors are directly fed into the next call unless you explicitly manage CUDA-Graph buffer lifetimes.

## License

PolyForm Noncommercial License 1.0.0. Commercial use requires a separate written license. See `LICENSE`.
