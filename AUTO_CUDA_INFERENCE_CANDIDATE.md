# MLBricks automatic CUDA inference — experimental candidate

This patch introduces `AttentionModel` (QKV language-model convenience), a
stock-arithmetic fixed-K/V graph generator, and a numerical verification
notebook for both `AttentionModel` and the previously integrated `BoltModel`.
It does **not** merge to `main` or enable unvalidated GPU configurations.

## Normal usage

```python
from mlbricks import BoltModel, AttentionModel, Bolt, Attention

bolt_model = BoltModel(vocab_size=2048, dim=384, heads=12, layers=6,
                       context=1024, bolt_cache='tight', avq_dim=198)
qkv_model = AttentionModel(vocab_size=2048, dim=384, heads=12,
                           layers=6, context=1024)

bolt_model.eval()
qkv_model.eval()
a = bolt_model.generate(prompt, 64, temperature=0)
b = qkv_model.generate(prompt, 64, temperature=0)
```

`Bolt` and `Attention` are standalone mix layers with existing forward APIs.
Their public constructor signatures remain unchanged.

## Guardrails (v3 FP32-master / FP16-inference candidate)

- `train()` and normal forward retain FP32 parameters and autograd behavior.
- CUDA model placement happens automatically where available.
- `BoltModel.generate` and `AttentionModel.generate` default to greedy decoding.
  Sampling via `temperature > 0` remains supported, using eager inference.
- On supported Tesla T4 / batch-one / greedy / no-dropout settings, a FP32
  model uses a cached *separate* FP16 inference mirror and Strict CUDA Graph.
- Mirror is not a PyTorch registered submodule: original `state_dict()` /
  `parameters()` and optimizer training remain FP32. Training mode or device
  changes drop the cached mirror; weight/buffer version changes rebuild it.
- Other GPU, CPU, unsupported architecture modes and sampling stay on original
  eager inference. CUDA OOM during mirror construction attempts fallback.
- Full FP16 graph parity must be measured against the **FP16 eager** math;
  casting FP32 weights to FP16 can change model outputs and token choices.
- Mirror construction and graph capture increase first-request latency and
  memory; no reusable graph pool is implemented yet.
- Current strict graph is **experimental**; real T4 v3 end-to-end validation
  and trained checkpoint evaluation must precede production approval.

## Files

- `mlbricks/attention_model.py`: QKV model and manual CUDA Graph generator.
- `mlbricks/gaussian.py`: add explicit dtype/fallback reporting (no weight cast).
- `tests/test_attention_model_auto_graph.py`: autograd/cache/precision regressions.
- `examples/mlbricks_auto_inference_gpu_audit.py`: stock parity + GPU audit.
