# Copyright 2026 Zameer Hussain and Akhtar Hussain
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# See LICENSE.md and LICENSING_NOTICE.md; commercial use requires a separate written license.

"""Unified model lifecycle helpers for every MLBricks architecture."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn


FORMAT_NAME = "mlbricks.model"
FORMAT_VERSION = 1
MODEL_FILE = "model.pt"
METADATA_FILE = "metadata.json"


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is None or str(device).lower() == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def _model_device(model: nn.Module) -> torch.device:
    for parameter in model.parameters():
        return parameter.device
    for buffer in model.buffers():
        return buffer.device
    return torch.device("cpu")


def _model_dtype(model: nn.Module) -> str | None:
    for parameter in model.parameters():
        if parameter.is_floating_point():
            return str(parameter.dtype).replace("torch.", "")
    for buffer in model.buffers():
        if buffer.is_floating_point():
            return str(buffer.dtype).replace("torch.", "")
    return None


def _architecture_name(model: nn.Module) -> str:
    cls = type(model)
    return f"{cls.__module__}.{cls.__qualname__}"


def _config_snapshot(model: nn.Module) -> Any | None:
    """Return JSON-friendly model configuration when one is exposed."""
    config = getattr(model, "config", None)
    if config is None:
        return None
    if is_dataclass(config):
        return asdict(config)
    to_dict = getattr(config, "to_dict", None)
    if callable(to_dict):
        try:
            return to_dict()
        except Exception:
            return None
    if isinstance(config, dict):
        return config
    return None


def inspect(model_or_path: nn.Module | str | Path) -> dict[str, Any]:
    """Inspect a live model or a saved MLBricks artifact without training it."""
    if isinstance(model_or_path, nn.Module):
        model = model_or_path
        return {
            "format": FORMAT_NAME,
            "format_version": FORMAT_VERSION,
            "architecture": _architecture_name(model),
            "parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "device": str(_model_device(model)),
            "dtype": _model_dtype(model),
            "training": bool(model.training),
            "config": _config_snapshot(model),
        }

    path = Path(model_or_path)
    metadata_path = path / METADATA_FILE if path.is_dir() else path.with_suffix(path.suffix + ".json")
    if not metadata_path.exists():
        raise FileNotFoundError(f"MLBricks metadata not found: {metadata_path}")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def save(
    model: nn.Module,
    path: str | Path,
    *,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Save any MLBricks/PyTorch module as one self-describing model artifact.

    The artifact stores the complete module graph and its parameters. This is
    what allows a future or mixed architecture (for example ESA + Bolt + SOUP)
    to reload without an architecture-specific loader.
    """
    if not isinstance(model, nn.Module):
        raise TypeError("mlbricks.save() expects a torch.nn.Module")

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    model_path = path / MODEL_FILE
    torch.save(model, model_path)

    info = inspect(model)
    info.update(
        {
            "format": FORMAT_NAME,
            "format_version": FORMAT_VERSION,
            "artifact": MODEL_FILE,
            "mlbricks_version": _mlbricks_version(),
        }
    )

    try:
        from .elasticbit import elasticbit_manifest
        quantization = elasticbit_manifest(model)
        if quantization is not None:
            info["quantization"] = quantization
    except Exception:
        pass

    if metadata:
        info["metadata"] = dict(metadata)

    (path / METADATA_FILE).write_text(
        json.dumps(info, indent=2, default=str),
        encoding="utf-8",
    )
    return path


def _load_legacy_esa(path: Path, device: torch.device, strict: bool) -> nn.Module:
    """Read pre-unified ESA artifacts through the new public loader."""
    from .esa.model import ESAModel, ESAModelConfig

    config_path = path / "config.json"
    state_path = path / "model.pt"
    config = ESAModelConfig.from_dict(json.loads(config_path.read_text(encoding="utf-8")))
    model = ESAModel(config, device=device)

    metadata_path = path / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        quantization = metadata.get("quantization")
        if quantization is not None:
            from .elasticbit import restore_elasticbit_modules
            restore_elasticbit_modules(model, quantization)
            model.to(device=device)

    state = torch.load(state_path, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=strict)
    return model


def load(
    path: str | Path,
    *,
    device: str | torch.device | None = "auto",
    strict: bool = True,
) -> nn.Module:
    """Load any model saved with :func:`mlbricks.save`.

    Only load artifacts you trust. Unified artifacts contain a serialized
    Python module graph and therefore use ``torch.load(..., weights_only=False)``.
    """
    path = Path(path)
    target_device = _resolve_device(device)

    metadata_path = path / METADATA_FILE
    model_path = path / MODEL_FILE
    if not model_path.exists():
        raise FileNotFoundError(f"MLBricks model artifact not found: {model_path}")

    metadata: dict[str, Any] = {}
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            metadata = {}

    # Pre-unified ESA artifacts used config.json + a weights-only model.pt.
    if metadata.get("format") != FORMAT_NAME and (path / "config.json").exists():
        return _load_legacy_esa(path, target_device, strict)

    model = torch.load(model_path, map_location=target_device, weights_only=False)
    if not isinstance(model, nn.Module):
        raise TypeError(f"Saved object is not a torch.nn.Module: {type(model)!r}")
    model.to(target_device)
    return model


def predict(model: nn.Module, *args: Any, **kwargs: Any) -> Any:
    """Run inference through a model's optimized predictor when available."""
    predictor = getattr(model, "predict", None)
    if callable(predictor):
        return predictor(*args, **kwargs)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            return model(*args, **kwargs)
    finally:
        model.train(was_training)


def generate(model: nn.Module, *args: Any, **kwargs: Any) -> Any:
    """Call the model's generation API through one package-level entry point."""
    generator = getattr(model, "generate", None)
    if not callable(generator):
        raise TypeError(f"{type(model).__name__} does not expose generate()")
    return generator(*args, **kwargs)


def compile(
    model: nn.Module,
    *,
    mode: str = "default",
    dynamic: bool | None = None,
    fullgraph: bool = False,
    strict: bool = False,
) -> nn.Module:
    """Compile a model using its MLBricks compile hook or torch.compile."""
    method = getattr(model, "compile", None)
    if callable(method):
        import inspect as _inspect
        params = _inspect.signature(method).parameters
        options = {
            "mode": mode,
            "dynamic": dynamic,
            "fullgraph": fullgraph,
            "strict": strict,
        }
        supported = {k: v for k, v in options.items() if k in params}
        result = method(**supported)
        return model if result is None else result
    if not hasattr(torch, "compile"):
        if strict:
            raise RuntimeError("torch.compile is not available in this PyTorch build")
        return model
    return torch.compile(model, mode=mode, dynamic=dynamic, fullgraph=fullgraph)


def quantize(
    model: nn.Module,
    *,
    method: str = "elasticbit",
    bits: int = 4,
    include_embeddings: bool = False,
    **kwargs: Any,
) -> nn.Module:
    """Quantize a model through the unified MLBricks optimization API."""
    name = str(method).strip().lower().replace("-", "")
    if name not in {"elasticbit", "eb"}:
        raise ValueError("method must currently be 'elasticbit'")
    from .elasticbit import ElasticBitConfig, quantize_module

    config = ElasticBitConfig(bits=int(bits), **kwargs)
    return quantize_module(model, config, include_embeddings=include_embeddings)


def _mlbricks_version() -> str:
    try:
        from importlib.metadata import version
        return version("mlbricks")
    except Exception:
        return "unknown"


__all__ = [
    "FORMAT_NAME",
    "FORMAT_VERSION",
    "save",
    "load",
    "inspect",
    "predict",
    "generate",
    "compile",
    "quantize",
]
