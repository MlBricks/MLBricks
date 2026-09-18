"""Research helpers for validating the clean ElasticBit API on real models.

This module is intentionally outside the core public surface.  It uses the same
threshold-driven analyzer and the same RuntimeMatrix implementation as
``ElasticBit.compress``; it does not contain an alternate quantization path.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence
import numpy as np


def _callModel(model: Any, batch: Any) -> Any:
    if isinstance(batch, Mapping):
        return model(**batch)
    if isinstance(batch, tuple):
        return model(*batch)
    return model(batch)


def _moveBatch(batch: Any, device: Any) -> Any:
    import torch
    if torch.is_tensor(batch):
        return batch.to(device)
    if isinstance(batch, Mapping):
        return {key: _moveBatch(value, device) for key, value in batch.items()}
    if isinstance(batch, tuple):
        return tuple(_moveBatch(value, device) for value in batch)
    if isinstance(batch, list):
        return [_moveBatch(value, device) for value in batch]
    return batch


def collectLinearCalibration(
    model: Any,
    calibrationData: Sequence[Any],
    *,
    device: Any,
    maxRowsPerLayer: int = 32,
) -> dict[str, np.ndarray]:
    import torch
    import torch.nn as nn

    collected: dict[str, list[np.ndarray]] = {}
    handles = []

    def makeHook(name: str):
        def hook(_module: Any, inputs: tuple[Any, ...]) -> None:
            if not inputs or not torch.is_tensor(inputs[0]):
                return
            rows = inputs[0].detach().float().reshape(-1, inputs[0].shape[-1])
            have = sum(item.shape[0] for item in collected.get(name, []))
            remaining = maxRowsPerLayer - have
            if remaining <= 0:
                return
            collected.setdefault(name, []).append(
                rows[:remaining].cpu().numpy().astype(np.float32, copy=True)
            )
        return hook

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            handles.append(module.register_forward_pre_hook(makeHook(name)))

    model.eval()
    try:
        with torch.inference_mode():
            for batch in calibrationData:
                _callModel(model, _moveBatch(batch, device))
    finally:
        for handle in handles:
            handle.remove()

    return {
        name: np.ascontiguousarray(np.concatenate(values, axis=0), dtype=np.float32)
        for name, values in collected.items()
        if values
    }


def analyzeModel(model: Any, calibrationData: Sequence[Any], threshold: float) -> list[dict[str, Any]]:
    import torch.nn as nn
    from . import ElasticBit

    device = next(model.parameters()).device
    calibration = collectLinearCalibration(model, calibrationData, device=device)
    rows: list[dict[str, Any]] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear) or name not in calibration:
            continue
        weights = np.ascontiguousarray(module.weight.detach().float().cpu().numpy(), dtype=np.float32)
        analysis = ElasticBit.analyze(weights, calibration[name], threshold)
        rows.append({
            "name": name,
            "selectedBits": analysis.selectedBits,
            "selectedError": analysis.selectedError,
        })
    return rows


__all__ = ["collectLinearCalibration", "analyzeModel"]
