"""Generic real-model test adapter for ElasticBit.

The native runtime itself has no PyTorch dependency. This optional adapter uses
PyTorch only to inspect an existing model, collect representative inputs from
``torch.nn.Linear`` layers, export exact-bit MLB3 matrices, evaluate a
weight-reconstructed model's loss/perplexity, and benchmark the native compact,
fast, and FP16 matrix kernels.

Architecture-specific fully device-resident replacement is intentionally left
to MLBricks model backends; this adapter never claims Python-chained timings are
full-model production latency.
"""

from __future__ import annotations

import contextlib
import copy
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np


@dataclass
class LayerResult:
    name: str
    rows: int
    cols: int
    storage_bits: int
    selected_error: float
    compute_type: str
    file_bytes: int
    compact_gpu_bytes: int
    fast_gpu_bytes: int
    fp16_gpu_bytes: int
    compact_ms: float
    fast_ms: float
    fp16_ms: float
    compact_fast_error: float
    file_path: str


def _call_model(model: Any, batch: Any) -> Any:
    if isinstance(batch, Mapping):
        return model(**batch)
    if isinstance(batch, (tuple, list)):
        return model(*batch)
    return model(batch)


def _move_batch(batch: Any, device: Any) -> Any:
    import torch

    if torch.is_tensor(batch):
        return batch.to(device)
    if isinstance(batch, Mapping):
        return {key: _move_batch(value, device) for key, value in batch.items()}
    if isinstance(batch, tuple):
        return tuple(_move_batch(value, device) for value in batch)
    if isinstance(batch, list):
        return [_move_batch(value, device) for value in batch]
    return batch


def _extract_loss(output: Any, batch: Any) -> Any:
    import torch
    import torch.nn.functional as F

    if hasattr(output, "loss") and output.loss is not None:
        return output.loss
    if isinstance(output, Mapping) and output.get("loss") is not None:
        return output["loss"]

    logits = output.logits if hasattr(output, "logits") else output
    labels = batch.get("labels") if isinstance(batch, Mapping) else None
    if labels is None:
        raise ValueError(
            "The model output has no loss and the batch has no 'labels'. "
            "Supply batches with labels or a model that returns loss."
        )

    if logits.ndim < 2:
        raise ValueError("Expected logits with at least two dimensions")

    # Causal-LM convention: predict token t+1 from token t.
    if logits.ndim == 3 and labels.ndim == 2:
        return F.cross_entropy(
            logits[:, :-1, :].reshape(-1, logits.shape[-1]),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
        )

    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=-100,
    )


def evaluate_loss_ppl(
    model: Any,
    batches: Sequence[Any],
    *,
    device: Any,
) -> dict[str, float]:
    import torch

    model.eval()
    losses: list[float] = []
    start = time.perf_counter()
    with torch.inference_mode():
        for raw_batch in batches:
            batch = _move_batch(raw_batch, device)
            output = _call_model(model, batch)
            loss = _extract_loss(output, batch)
            losses.append(float(loss.detach().float().cpu()))
    seconds = time.perf_counter() - start
    if not losses:
        raise ValueError("No validation batches were supplied")
    mean_loss = float(np.mean(losses))
    return {
        "loss": mean_loss,
        "perplexity": float(math.exp(min(mean_loss, 50.0))),
        "seconds": seconds,
        "batches_per_second": len(losses) / seconds,
    }


def collect_linear_calibration(
    model: Any,
    batches: Sequence[Any],
    *,
    device: Any,
    max_rows_per_layer: int = 32,
) -> dict[str, np.ndarray]:
    import torch
    import torch.nn as nn

    collected: dict[str, list[np.ndarray]] = {}
    handles = []

    def make_hook(name: str) -> Callable[..., None]:
        def hook(_module: Any, inputs: tuple[Any, ...]) -> None:
            if not inputs:
                return
            tensor = inputs[0]
            if not torch.is_tensor(tensor):
                return
            rows = tensor.detach().float().reshape(-1, tensor.shape[-1])
            current = sum(item.shape[0] for item in collected.get(name, []))
            remaining = max_rows_per_layer - current
            if remaining <= 0:
                return
            sample = rows[:remaining].cpu().numpy().astype(np.float32, copy=True)
            collected.setdefault(name, []).append(sample)
        return hook

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            handles.append(module.register_forward_pre_hook(make_hook(name)))

    model.eval()
    try:
        with torch.inference_mode():
            for raw_batch in batches:
                batch = _move_batch(raw_batch, device)
                _call_model(model, batch)
                if collected and all(
                    sum(item.shape[0] for item in values) >= max_rows_per_layer
                    for values in collected.values()
                ):
                    break
    finally:
        for handle in handles:
            handle.remove()

    return {
        name: np.ascontiguousarray(np.concatenate(values, axis=0), dtype=np.float32)
        for name, values in collected.items()
        if values
    }


def relative_error(actual: np.ndarray, reference: np.ndarray) -> float:
    return float(
        np.linalg.norm(actual - reference)
        / max(float(np.linalg.norm(reference)), 1.0e-12)
    )


def export_and_benchmark_model(
    model: Any,
    calibration_batches: Sequence[Any],
    validation_batches: Sequence[Any],
    eb: Any | None = None,
    *,
    output_dir: str | Path,
    device: Any | None = None,
    threshold: float = 0.01,
    min_bits: int = 4,
    max_bits: int = 32,
    calibration_rows_per_layer: int = 16,
    benchmark_iterations: int = 500,
    deepcopy_for_quality: bool = True,
) -> dict[str, Any]:
    """Export all ``nn.Linear`` matrices and benchmark ElasticBit.

    Full-model loss/PPL after export is evaluated by temporarily replacing each
    linear weight with the exact reconstructed weight. This validates storage
    degradation on the real model. Native compact/fast promoted arithmetic is
    separately validated per layer using held calibration vectors.
    """

    import torch
    import torch.nn as nn

    if eb is None:
        from . import _C as eb

    if device is None:
        device = next(model.parameters()).device

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    matrices_dir = destination / "matrices"
    matrices_dir.mkdir(parents=True, exist_ok=True)

    calibration = collect_linear_calibration(
        model,
        calibration_batches,
        device=device,
        max_rows_per_layer=calibration_rows_per_layer,
    )

    original_metrics = evaluate_loss_ppl(model, validation_batches, device=device)
    quality_model = copy.deepcopy(model) if deepcopy_for_quality else model
    quality_modules = dict(quality_model.named_modules())

    Matrix = eb.RuntimeMatrix
    FP16 = eb.NativeFP16Matrix
    results: list[LayerResult] = []

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if name not in calibration:
            print(f"Skipping {name}: no calibration input captured")
            continue

        weights = np.ascontiguousarray(
            module.weight.detach().float().cpu().numpy(), dtype=np.float32
        )
        layer_calibration = calibration[name]
        analysis = eb.bitsAnaliser(
            weights, layer_calibration, threshold, min_bits, max_bits
        )
        bits = int(analysis["selected_bits"])
        selected_error = float(analysis["selected_error"])
        compute_type = str(analysis["selected_compute_type"])

        compact = Matrix(weights, bits, "compact")
        fast = Matrix(weights, bits, "fast")
        fp16 = FP16(weights)

        representative = np.ascontiguousarray(layer_calibration[0], dtype=np.float32)
        compact_output = np.asarray(compact.forward(representative))
        fast_output = np.asarray(fast.forward(representative))
        compact_fast_error = relative_error(fast_output, compact_output)

        if bits <= 8 and compact_fast_error > 1.0e-6:
            raise RuntimeError(
                f"Direct widening mismatch in {name}: {compact_fast_error}"
            )

        safe_name = name.replace(".", "__") or "root"
        path = matrices_dir / f"{safe_name}.mlb"
        compact.save(str(path))

        # Validate both loader modes immediately.
        loaded_compact = Matrix.load(str(path), "compact")
        loaded_fast = Matrix.load(str(path), "fast")
        if relative_error(
            np.asarray(loaded_compact.forward(representative)), compact_output
        ) > 1.0e-7:
            raise RuntimeError(f"Compact MLB3 round trip failed for {name}")
        if relative_error(
            np.asarray(loaded_fast.forward(representative)), fast_output
        ) > 1.0e-7:
            raise RuntimeError(f"Fast MLB3 round trip failed for {name}")

        # Replace only the quality-copy weight with exact reconstructed values.
        reconstructed = torch.from_numpy(np.asarray(compact.dequantize()))
        target_module = quality_modules[name]
        target_module.weight.data.copy_(
            reconstructed.to(
                device=target_module.weight.device,
                dtype=target_module.weight.dtype,
            )
        )

        compact_ms = float(compact.benchmark(representative, benchmark_iterations))
        fast_ms = float(fast.benchmark(representative, benchmark_iterations))
        fp16_ms = float(fp16.benchmark(representative, benchmark_iterations))

        result = LayerResult(
            name=name,
            rows=weights.shape[0],
            cols=weights.shape[1],
            storage_bits=bits,
            selected_error=selected_error,
            compute_type=compute_type,
            file_bytes=int(compact.file_weight_bytes),
            compact_gpu_bytes=int(compact.gpu_weight_bytes),
            fast_gpu_bytes=int(fast.gpu_weight_bytes),
            fp16_gpu_bytes=int(fp16.gpu_weight_bytes),
            compact_ms=compact_ms,
            fast_ms=fast_ms,
            fp16_ms=fp16_ms,
            compact_fast_error=compact_fast_error,
            file_path=str(path),
        )
        results.append(result)
        print(
            f"{name:48s} bits={bits:2d} {compute_type:4s} "
            f"error={selected_error:.6f} "
            f"compact={compact_ms:.6f}ms fast={fast_ms:.6f}ms "
            f"fp16={fp16_ms:.6f}ms"
        )

    reconstructed_metrics = evaluate_loss_ppl(
        quality_model, validation_batches, device=device
    )

    total_file = sum(item.file_bytes for item in results)
    total_compact_gpu = sum(item.compact_gpu_bytes for item in results)
    total_fast_gpu = sum(item.fast_gpu_bytes for item in results)
    total_fp16 = sum(item.fp16_gpu_bytes for item in results)
    compact_ms = sum(item.compact_ms for item in results)
    fast_ms = sum(item.fast_ms for item in results)
    fp16_ms = sum(item.fp16_ms for item in results)

    manifest = {
        "format": "MLB3",
        "technology": "ElasticBit",
        "threshold": threshold,
        "quality_scope": (
            "Full-model loss/PPL uses exact reconstructed weights in the "
            "original framework. Native promoted compute is checked and "
            "benchmarked per linear layer."
        ),
        "original": original_metrics,
        "reconstructed": reconstructed_metrics,
        "loss_change": reconstructed_metrics["loss"] - original_metrics["loss"],
        "ppl_change": (
            reconstructed_metrics["perplexity"] - original_metrics["perplexity"]
        ),
        "memory": {
            "exact_file_bytes": total_file,
            "compact_gpu_bytes": total_compact_gpu,
            "fast_gpu_bytes": total_fast_gpu,
            "fp16_gpu_bytes": total_fp16,
            "file_reduction_vs_fp16": 1.0 - total_file / total_fp16,
            "compact_gpu_reduction_vs_fp16": 1.0 - total_compact_gpu / total_fp16,
            "fast_gpu_reduction_vs_fp16": 1.0 - total_fast_gpu / total_fp16,
        },
        "kernel_benchmark": {
            "compact_total_ms": compact_ms,
            "fast_total_ms": fast_ms,
            "fp16_total_ms": fp16_ms,
            "compact_speedup_vs_fp16": fp16_ms / compact_ms,
            "fast_speedup_vs_fp16": fp16_ms / fast_ms,
        },
        "layers": [item.__dict__ for item in results],
    }

    manifest_path = destination / "elasticbit_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
