# BOLT public API patch

The primary public classes are `Bolt`, `BoltConfig`, `BoltModel`,
`BoltStateMemory`, and `BoltCache`. New models use a modern, quality-selected
configuration. `BoltPreset`, `BoltPresetCache`, and `PostSoftmaxAVQ` remain importable.

## Automatic device and precision

`device="auto"` chooses CUDA when available and CPU otherwise. Explicit
`device="cuda:1"` selects GPU 1; explicit unsupported CUDA raises an error.
Model inputs/targets automatically follow current model device. `precision="auto"`
uses CUDA autocast (BF16 when supported, otherwise FP16), without changing
FP32 master weights. On CPU it runs FP32. For high-performance data pipelines,
place batches directly on the target device to avoid repeated copies.

## Architecture profiles

`profile="standard"`: latent width 32. `profile="compact"`: latent width 16.
Both default to AVQ state width 198, **`state_memory=True`**, compact-dimension
attention scaling, and balanced state initialization (projection reset with
independent RNG preservation and gamma=0.25). Set `attention_scale="head"`
or `memory_init="legacy"` to reproduce historical mathematical settings.
Set `state_memory=False` on `BoltConfig` or standalone `Bolt` for attention-only
BOLT. No AVQ parameters, backward kernels, or recurrent state are allocated in
that configuration. The historical `bolt_mode="fast"` and layer `mode="fast"`
still work for old code. The two variants have different checkpoint structure.

## Checkpoint compatibility

Packed QUG is trained with a single parameter but saves and loads **canonical
separate** `q_proj`, `c_proj`, `g_proj` checkpoint keys. Existing model state
dictionaries can load without editing. Changing a trained model's math profile
is **not** an implicit checkpoint conversion. Optimizer state uses different
parameter layouts; rebuild or migrate it explicitly when enabling packing.
The old `Gaussian` and `GaussianConfig` names have been removed. For legacy
checkpoint weights, use `BoltModel` and explicitly select matching math settings.

## CUDA acceleration

`MLBRICKS_BUILD_BOLT_ACCEL=1 pip install -v .` builds the segmented (256-token)
and short-context native CUDA state-memory kernels. It is opt-in. Prebuilt
wheels can bundle these extensions. With `backend="auto"`, BOLT selects the
accelerated state-memory path when available, otherwise falls back to the
original modern native extension or mathematical PyTorch implementation.
No Triton or `torch.compile` is required. CUDA compilation and kernel speed
must be verified on an NVIDIA GPU; the source tree is not a prebuilt wheel.
