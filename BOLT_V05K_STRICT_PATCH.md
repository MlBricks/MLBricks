# BOLT v0.5k opt-in strict inference

This snapshot adds supported `enable_bolt_strict_inference(True)` on modern `BoltModel`, `Bolt`, and `BoltPreset`. It replaces the v0.5j experimental monkeypatch with a real class API. It **does not** alter default inference, checkpoint keys, or the training path.

```python
model.eval().cuda().half()
model.enable_bolt_strict_inference(True)
with torch.inference_mode(), torch.autocast('cuda',dtype=torch.float16):
    session = model.prepare_cuda_graph_generation(prompt, max_new_tokens=64)
    output_ids = session.generate()
```

`strict` retains separate Q/U/G projections, multiplies P into Q at the original location, and keeps the original AVQ norm/gamma/back arithmetic. It reuses fixed-capacity masked SDPA and CUDA Graph routing. Strict mode rejects v0.5 native kernels. Toggling to approximate `enable_bolt_optimized_inference(True)` explicitly disables strict.

**Not yet production approved:** T4 graph parity established for the v0.5j method-override prototype (BOLT-16/32, FP16, batch 1, prompt 512, 64 tokens). This dedicated v0.5k class integration has not yet been GPU-validated. RoPE, BF16, trained checkpoints, concurrency, and deployment lifecycle remain unvalidated.
