# Copyright 2026 Zameer Hussain and Akhtar Hussain
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# See LICENSE and LICENSING_NOTICE.md; commercial use requires a separate written license.

from __future__ import annotations

from typing import Any, Callable

import torch
import torch.nn as nn


def _cfg_value(name: str, default: Any) -> Any:
    """
    Read benchmark defaults from mlbricks.benchmark.

    Supports both:
        DEFAULT_BENCHMARK_CONFIG.name
        DEFAULT_BENCHMARK_CONFIG["name"]
    """
    try:
        from .benchmark import DEFAULT_BENCHMARK_CONFIG

        cfg = DEFAULT_BENCHMARK_CONFIG

        if isinstance(cfg, dict):
            return cfg.get(name, default)

        return getattr(cfg, name, default)

    except Exception:
        return default


def _module_device(module: nn.Module) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _move_to_device(data: Any, device: torch.device) -> Any:
    if torch.is_tensor(data):
        return data.to(device, non_blocking=True)

    if isinstance(data, tuple):
        return tuple(_move_to_device(x, device) for x in data)

    if isinstance(data, list):
        return [_move_to_device(x, device) for x in data]

    if isinstance(data, dict):
        return {k: _move_to_device(v, device) for k, v in data.items()}

    return data


def _call_module(module: nn.Module, warmup_batch: Any) -> Any:
    if isinstance(warmup_batch, tuple):
        return module(*warmup_batch)

    if isinstance(warmup_batch, dict):
        return module(**warmup_batch)

    return module(warmup_batch)


def _loss_from_output(output: Any) -> torch.Tensor:
    """
    Infer scalar loss from common outputs.

    Supported:
        Tensor
        (logits, loss)
        {"loss": loss}
    """
    if torch.is_tensor(output):
        return output.float().pow(2).mean()

    if isinstance(output, dict):
        if "loss" in output and torch.is_tensor(output["loss"]):
            return output["loss"].float()

        for value in output.values():
            if torch.is_tensor(value):
                return value.float().pow(2).mean()

    if isinstance(output, (tuple, list)):
        # Prefer scalar loss, e.g. TinyLM returns (logits, loss).
        for value in output:
            if torch.is_tensor(value) and value.ndim == 0:
                return value.float()

        # Otherwise use first tensor.
        for value in output:
            if torch.is_tensor(value):
                return value.float().pow(2).mean()

    raise ValueError(
        "Could not infer loss from module output. "
        "Pass loss_fn=... to thunderBoost()."
    )


def _has_token_lm_shape(module: nn.Module) -> bool:
    """
    Detect TinyLM-like token models.

    These models expect token IDs [B, T], not hidden vectors [B, T, C].
    """
    names = set(dict(module.named_modules()).keys())

    if "token_emb" in names or "tok_emb" in names or "lm_head" in names:
        return True

    if hasattr(module, "token_emb") or hasattr(module, "tok_emb") or hasattr(module, "lm_head"):
        return True

    return False


def _shape_from_module(module: nn.Module):
    """
    Try to find batch/block/embd from module or nested ESA layers.
    """
    candidates = [module] + list(module.modules())

    for m in candidates:
        batch_size = getattr(m, "batch", None)
        block = getattr(m, "block", None)
        embd = getattr(m, "embd", None)

        if batch_size is None:
            batch_size = getattr(m, "batch_size", None)

        if block is None:
            block = getattr(m, "block_size", None)

        if embd is None:
            embd = getattr(m, "n_embd", None)

        if batch_size is not None and block is not None and embd is not None:
            return int(batch_size), int(block), int(embd)

    return None


def _auto_vector_input(
    module: nn.Module,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if _has_token_lm_shape(module):
        raise ValueError(
            "This looks like a token language model. "
            "Use thunderBoost(model, batch=get_batch)."
        )

    shape = _shape_from_module(module)

    if shape is None:
        raise ValueError(
            "Could not auto-create input for this module. "
            "Use thunderBoost(model, batch=x) for vector-input models, "
            "or thunderBoost(model, batch=get_batch) for TinyLM/token models."
        )

    batch_size, block, embd = shape

    if device.type != "cuda":
        dtype = torch.float32

    return torch.randn(
        batch_size,
        block,
        embd,
        device=device,
        dtype=dtype,
    )


def _make_input_fn_from_batch_fn(batch_fn: Callable[..., Any]) -> Callable[[], Any]:
    """
    Convert a user batch function into a zero-argument input function.

    Supports both:
        batch_fn("train")
        batch_fn()
    """

    def input_fn():
        try:
            return batch_fn("train")
        except TypeError:
            return batch_fn()

    return input_fn


def _disable_esa_auto_move(module: nn.Module) -> int:
    """
    Disable ESA.forward() internal x.to(device) for boosted/compiled models.

    Why:
        DeviceCopy ops inside compiled/CUDA graph paths can cause graph
        partitioning. After thunderBoost, users should pass inputs already
        on the module device.

    Normal non-boosted ESA keeps auto_move_input=True.
    """
    count = 0

    for m in module.modules():
        if hasattr(m, "auto_move_input"):
            try:
                m.auto_move_input = False
                count += 1
            except Exception:
                pass

    return count


def thunderBoost(
    module: nn.Module,
    example_input: Any | None = None,
    *,
    batch: Any | Callable[..., Any] | None = None,
    state: bool = False,
    get_batch: Callable[..., Any] | None = None,
    input_fn: Callable[[], Any] | None = None,
    loss_fn: Callable[[Any], torch.Tensor] | None = None,
    steps: int | None = None,
    compile: bool = True,
    compile_mode: str | None = None,
    backward: bool = True,
    amp: bool = True,
    dtype: torch.dtype = torch.float16,
    device: str | torch.device | None = "auto",
    return_stats: bool | None = None,
):
    """
    Compile and warm up an ESA layer or full model.

    thunderBoost runs warmup ONCE at module/model level.
    It does not call optimizer.step(), so it does not train/update weights.

    Clean API:

        Single ESA layer:
            layer = thunderBoost(layer)

        Single ESA layer with state:
            layer, boost_state = thunderBoost(layer, state=True)

        TinyLM/token model:
            model = thunderBoost(model, batch=get_batch)

        TinyLM/token model with state:
            model, boost_state = thunderBoost(model, batch=get_batch, state=True)

        Vector-input model:
            model = thunderBoost(model, batch=x)

        Vector-input model with state:
            model, boost_state = thunderBoost(model, batch=x, state=True)

    Backward-compatible old API:

        model = thunderBoost(model, get_batch=get_batch)
        model, stats = thunderBoost(model, return_stats=True)

    Important after thunderBoost:
        Inputs should already be on the model device.

        Example:
            device = next(model.parameters()).device
            x = torch.randn(B, T, C, device=device)
            y = model(x)
    """
    if not isinstance(module, nn.Module):
        raise TypeError("thunderBoost() expects an nn.Module.")

    was_training = bool(module.training)

    # Backward compatibility:
    # return_stats=True means state=True.
    if return_stats is not None:
        state = bool(return_stats)

    if steps is None:
        steps = int(_cfg_value("compile_warmup_steps", 2))

    if compile_mode is None:
        compile_mode = str(_cfg_value("compile_mode", "reduce-overhead"))

    # Backward compatibility:
    # get_batch=get_batch becomes batch=get_batch.
    if get_batch is not None:
        if batch is not None:
            raise ValueError("Use either batch=... or get_batch=..., not both.")
        batch = get_batch

    if input_fn is not None and batch is not None:
        raise ValueError("Use either input_fn=... or batch=..., not both.")

    # New API:
    # batch can be either:
    #   1. a callable batch function, e.g. batch=get_batch
    #   2. an actual example batch, e.g. batch=x or batch=(xb, yb)
    if batch is not None:
        if callable(batch):
            input_fn = _make_input_fn_from_batch_fn(batch)
        else:
            if example_input is not None:
                raise ValueError("Use either example_input=... or batch=..., not both.")
            example_input = batch

    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device is None:
        device = _module_device(module)
    else:
        device = torch.device(device)

    module = module.to(device)

    # Prepare input before torch.compile because compiled wrappers may hide
    # custom attributes like batch/block/embd.
    if example_input is None and input_fn is None:
        example_input = _auto_vector_input(
            module=module,
            device=device,
            dtype=dtype,
        )

    # Move provided example input to device before compile/warmup.
    if example_input is not None:
        example_input = _move_to_device(example_input, device)

    # Disable internal ESA x.to(device) before compile.
    # This avoids DeviceCopy ops inside compiled graphs.
    auto_move_disabled_count = _disable_esa_auto_move(module)

    compiled = False

    if compile:
        try:
            module = torch.compile(
                module,
                mode=compile_mode,
                fullgraph=False,
            )
            compiled = True
        except Exception:
            compiled = False

    module.train()

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    last_loss_value = None

    for _ in range(int(steps)):
        if input_fn is not None:
            warmup_batch = input_fn()
        else:
            warmup_batch = example_input

        warmup_batch = _move_to_device(warmup_batch, device)

        module.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type="cuda",
            dtype=dtype,
            enabled=(amp and device.type == "cuda"),
        ):
            output = _call_module(module, warmup_batch)
            loss = loss_fn(output) if loss_fn is not None else _loss_from_output(output)

        if torch.is_tensor(loss):
            last_loss_value = float(loss.detach().float().item())

        if backward and torch.is_tensor(loss) and loss.requires_grad:
            loss.backward()

        # No optimizer.step() here.
        # thunderBoost warms compile/runtime path but does not train.
        module.zero_grad(set_to_none=True)

    if device.type == "cuda":
        torch.cuda.synchronize()
        peak_mem_mb = torch.cuda.max_memory_allocated() / 1024**2
    else:
        peak_mem_mb = 0.0

    boost_state = {
        "compiled": compiled,
        "steps": int(steps),
        "compile_mode": compile_mode,
        "device": str(device),
        "backward": bool(backward),
        "amp": bool(amp),
        "peak_mem_mb": peak_mem_mb,
        "last_loss": last_loss_value,
        "auto_move_disabled_count": auto_move_disabled_count,
    }

    try:
        setattr(module, "_thunderboost_state", boost_state)
        setattr(module, "_thunderboost_stats", boost_state)  # old compatibility
    except Exception:
        pass

    # Warmup may need train mode for backward, but thunderBoost must not change
    # the caller's persistent train/eval state.
    module.train(was_training)

    if state:
        return module, boost_state

    return module