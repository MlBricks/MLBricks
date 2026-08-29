# Copyright 2026 Zameer Hussain and Akhtar Hussain
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# See LICENSE and LICENSING_NOTICE.md; commercial use requires a separate written license.

"""ElasticBit portable weight quantization for MLBricks.

ElasticBit stores weights in a packed 2–8 bit representation. The cached
portable runtime lazily materializes one dequantized execution weight, while
the native CUDA ``runtime="packed"`` linear consumes the bitstream directly and
avoids full-weight materialization. Both runtimes use the same checkpoint
representation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import torch
from torch import nn
import torch.nn.functional as F

from ..runtime import normalize_backend
from .native_api import RuntimeMatrix, NativeFP16Matrix, bitsAnaliser, available as native_runtime_available
from ..planner import EXECUTION_PLANNER


_DTYPE_NAMES: dict[torch.dtype, str] = {
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.float32: "float32",
    torch.float64: "float64",
}
_NAME_DTYPES = {name: dtype for dtype, name in _DTYPE_NAMES.items()}


def _dtype_name(dtype: torch.dtype | None) -> str | None:
    if dtype is None:
        return None
    if dtype not in _DTYPE_NAMES:
        raise ValueError(f"Unsupported ElasticBit dtype: {dtype}")
    return _DTYPE_NAMES[dtype]


def _dtype_from_name(name: str | None) -> torch.dtype | None:
    if name is None:
        return None
    try:
        return _NAME_DTYPES[str(name)]
    except KeyError as exc:
        raise ValueError(f"Unknown ElasticBit dtype name: {name!r}") from exc


@dataclass(frozen=True)
class ElasticBitConfig:
    """Configuration for symmetric group-wise weight quantization."""

    bits: int = 4
    group_size: int = 128
    scale_dtype: torch.dtype = torch.float16
    compute_dtype: torch.dtype | None = None
    cache_dequantized: bool = True
    runtime: str = "auto"
    backend: str = "auto"

    def __post_init__(self) -> None:
        if not 2 <= self.bits <= 8:
            raise ValueError("bits must be between 2 and 8")
        if self.group_size <= 0:
            raise ValueError("group_size must be positive")
        if self.scale_dtype not in {torch.float16, torch.float32, torch.bfloat16}:
            raise ValueError("scale_dtype must be float16, bfloat16, or float32")
        if self.compute_dtype is not None and self.compute_dtype not in {
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
        }:
            raise ValueError("compute_dtype must be a floating-point dtype or None")
        runtime = str(self.runtime).strip().lower()
        if runtime not in {"auto", "cached", "packed"}:
            raise ValueError("runtime must be one of: auto, cached, packed")
        backend = normalize_backend(self.backend, warn_legacy=True)
        # Legacy runtime names remain loadable, but backend is the canonical API.
        if backend == "auto" and runtime == "packed":
            backend = "native"
        elif backend == "auto" and runtime == "cached":
            backend = "pytorch"
        object.__setattr__(self, "runtime", runtime)
        object.__setattr__(self, "backend", backend)

    def to_manifest(self) -> dict[str, Any]:
        return {
            "bits": int(self.bits),
            "group_size": int(self.group_size),
            "scale_dtype": _dtype_name(self.scale_dtype),
            "compute_dtype": _dtype_name(self.compute_dtype),
            "cache_dequantized": bool(self.cache_dequantized),
            "runtime": self.runtime,
            "backend": self.backend,
        }

    @classmethod
    def from_manifest(cls, data: dict[str, Any]) -> "ElasticBitConfig":
        return cls(
            bits=int(data.get("bits", 4)),
            group_size=int(data.get("group_size", 128)),
            scale_dtype=_dtype_from_name(data.get("scale_dtype")) or torch.float16,
            compute_dtype=_dtype_from_name(data.get("compute_dtype")),
            cache_dequantized=bool(data.get("cache_dequantized", True)),
            runtime=str(data.get("runtime", "auto")),
            backend=str(data.get("backend", "auto")),
        )


@dataclass
class PackedElasticBit:
    packed: torch.Tensor
    scales: torch.Tensor
    shape: tuple[int, ...]
    bits: int
    group_size: int
    original_numel: int

    @property
    def storage_bytes(self) -> int:
        return (
            self.packed.numel() * self.packed.element_size()
            + self.scales.numel() * self.scales.element_size()
        )


def _pack_unsigned(values: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack unsigned values into a compact uint8 bitstream without Python loops."""
    values = values.detach().to(device="cpu", dtype=torch.int64).flatten()
    if values.numel() == 0:
        return torch.empty(0, dtype=torch.uint8)
    mask = (1 << bits) - 1
    values = values & mask
    bit_pos = torch.arange(values.numel(), dtype=torch.int64) * int(bits)
    byte_index = bit_pos >> 3
    offset = bit_pos & 7
    output_size = (values.numel() * int(bits) + 7) // 8
    accum = torch.zeros(output_size, dtype=torch.int64)

    low = (values << offset) & 0xFF
    accum.index_add_(0, byte_index, low)

    spill_mask = offset + int(bits) > 8
    if bool(spill_mask.any()):
        spill_index = byte_index[spill_mask] + 1
        high = values[spill_mask] >> (8 - offset[spill_mask])
        accum.index_add_(0, spill_index, high)

    return (accum & 0xFF).to(torch.uint8)


def _unpack_unsigned(packed: torch.Tensor, count: int, bits: int) -> torch.Tensor:
    """Vectorized unpack that runs on the same device as ``packed``."""
    if count <= 0:
        return torch.empty(0, dtype=torch.int16, device=packed.device)
    data = packed.detach().to(dtype=torch.int64).flatten()
    bit_pos = torch.arange(count, device=data.device, dtype=torch.int64) * int(bits)
    byte_index = bit_pos >> 3
    offset = bit_pos & 7
    value = data[byte_index] >> offset
    spill_mask = offset + int(bits) > 8
    if bool(spill_mask.any()):
        high = data[byte_index[spill_mask] + 1] << (8 - offset[spill_mask])
        value = value.clone()
        value[spill_mask] |= high
    return (value & ((1 << bits) - 1)).to(torch.int16)


def quantize_tensor(
    tensor: torch.Tensor,
    config: ElasticBitConfig | None = None,
) -> PackedElasticBit:
    """Quantize a floating tensor using symmetric per-group scaling."""
    config = config or ElasticBitConfig()
    if not tensor.is_floating_point():
        raise TypeError("ElasticBit only quantizes floating-point tensors")

    source = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous().flatten()
    numel = source.numel()
    groups = (numel + config.group_size - 1) // config.group_size
    padded_numel = groups * config.group_size
    if padded_numel != numel:
        source = F.pad(source, (0, padded_numel - numel))
    grouped = source.view(groups, config.group_size)

    qmax = (1 << (config.bits - 1)) - 1
    qmin = -(1 << (config.bits - 1))
    max_abs = grouped.abs().amax(dim=1)
    scales = torch.where(max_abs > 0, max_abs / qmax, torch.ones_like(max_abs))
    quantized = torch.round(grouped / scales[:, None]).clamp(qmin, qmax).to(torch.int16)
    unsigned = quantized - qmin

    return PackedElasticBit(
        packed=_pack_unsigned(unsigned, config.bits),
        scales=scales.to(config.scale_dtype),
        shape=tuple(tensor.shape),
        bits=config.bits,
        group_size=config.group_size,
        original_numel=numel,
    )


def dequantize_tensor(
    value: PackedElasticBit,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Restore a packed tensor, unpacking on the destination device when possible."""
    target = value.packed.device if device is None else torch.device(device)
    packed = value.packed.to(target, non_blocking=True)
    scales = value.scales.to(target, non_blocking=True)
    groups = scales.numel()
    count = groups * value.group_size
    unsigned = _unpack_unsigned(packed, count, value.bits).to(torch.int32)
    qmin = -(1 << (value.bits - 1))
    signed = unsigned + qmin
    grouped = signed.view(groups, value.group_size).to(torch.float32)
    restored = grouped * scales.to(torch.float32)[:, None]
    restored = restored.flatten()[: value.original_numel].view(value.shape)
    return restored.to(dtype=dtype)


class _ElasticWeightMixin:
    _dequant_cache: dict[tuple[str, int | None, torch.dtype], torch.Tensor]

    def _init_cache(self) -> None:
        self._dequant_cache = {}

    def clear_cache(self) -> None:
        self._dequant_cache.clear()

    def _apply(self, fn):
        self.clear_cache()
        EXECUTION_PLANNER.clear_owner_routes(self)
        return super()._apply(fn)

    def _load_from_state_dict(self, *args, **kwargs):
        self.clear_cache()
        EXECUTION_PLANNER.clear_owner_routes(self)
        return super()._load_from_state_dict(*args, **kwargs)

    @property
    def backend(self) -> str:
        return self.config.backend

    def set_backend(self, backend: str, *, recursive: bool = True):
        del recursive
        value = normalize_backend(backend, warn_legacy=True)
        data = self.config.to_manifest()
        data["backend"] = value
        data["runtime"] = "auto"
        self.config = ElasticBitConfig.from_manifest(data)
        self.clear_cache()
        EXECUTION_PLANNER.clear_owner_routes(self)
        return self

    def resolved_backend(self) -> str:
        if self.backend == "pytorch":
            return "pytorch"
        if self.backend == "native":
            return "native-required"
        routes = EXECUTION_PLANNER.owner_routes(self)
        if routes:
            return "+".join(sorted(set(routes.values())))
        return "planner(auto; qualify-once)"

    def _materialized_weight(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (device.type, device.index, dtype)
        if self.config.cache_dequantized:
            cached = self._dequant_cache.get(key)
            if cached is not None:
                return cached
        weight = dequantize_tensor(self.packed(), device=device, dtype=dtype)
        if self.config.cache_dequantized:
            self._dequant_cache[key] = weight
        return weight


class ElasticLinear(_ElasticWeightMixin, nn.Module):
    """Inference-only packed replacement for ``nn.Linear``.

    The speed-first cached runtime lazily dequantizes the weight once. The
    CUDA ``runtime="packed"`` path consumes the bitstream directly for true
    low-memory inference without constructing a full dequantized weight.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        config: ElasticBitConfig | None = None,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.config = config or ElasticBitConfig()
        self.original_numel = out_features * in_features
        groups = (self.original_numel + self.config.group_size - 1) // self.config.group_size
        packed_bytes = (groups * self.config.group_size * self.config.bits + 7) // 8
        self.register_buffer("packed_weight", torch.zeros(packed_bytes, dtype=torch.uint8))
        self.register_buffer("scales", torch.zeros(groups, dtype=self.config.scale_dtype))
        self.register_buffer(
            "weight_shape",
            torch.tensor([out_features, in_features], dtype=torch.int64),
        )
        self.bias = (
            nn.Parameter(torch.zeros(out_features), requires_grad=False)
            if bias
            else None
        )
        self._init_cache()

    def _assign_packed(self, packed: PackedElasticBit) -> None:
        self.packed_weight = packed.packed
        self.scales = packed.scales
        self.weight_shape = torch.tensor(packed.shape, dtype=torch.int64)
        self.original_numel = packed.original_numel
        self.clear_cache()

    @classmethod
    def from_linear(
        cls,
        layer: nn.Linear,
        config: ElasticBitConfig | None = None,
    ) -> "ElasticLinear":
        result = cls(
            layer.in_features,
            layer.out_features,
            bias=layer.bias is not None,
            config=config,
        )
        result._assign_packed(quantize_tensor(layer.weight, result.config))
        if layer.bias is not None:
            result.bias.data.copy_(layer.bias.detach().to(result.bias))
        return result.to(device=layer.weight.device)

    def packed(self) -> PackedElasticBit:
        return PackedElasticBit(
            packed=self.packed_weight,
            scales=self.scales,
            shape=tuple(int(v) for v in self.weight_shape.detach().cpu().tolist()),
            bits=self.config.bits,
            group_size=self.config.group_size,
            original_numel=self.original_numel,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = self.config.compute_dtype or x.dtype
        x_compute = x.to(dtype)
        policy = self.config.backend
        extra = (
            int(self.in_features), int(self.out_features),
            int(self.config.bits), int(self.config.group_size),
        )

        def native_impl_available():
            packed_linear = None
            available = False
            if x_compute.is_cuda:
                try:
                    from ..esa.native import elastic_linear_packed as packed_linear
                    from ..esa.native import custom_ops_registered as custom_ops_registered
                    from ..esa.native import cuda_available as native_cuda_available
                    available = bool(custom_ops_registered() and native_cuda_available())
                except (ImportError, AttributeError, RuntimeError):
                    available = False
            return available, packed_linear

        frozen = EXECUTION_PLANNER.owner_routes(self).get(("elastic_linear", False))
        packed_linear = None
        native_available = False

        if policy == "pytorch":
            route = "pytorch"
        elif policy == "native":
            native_available, packed_linear = native_impl_available()
            if not native_available or packed_linear is None:
                raise RuntimeError(
                    "ElasticBit backend='native' requires the packed MLBricks CUDA extension"
                )
            route = "native"
        elif frozen in {"native", "pytorch"}:
            # Element-local auto decision was already qualified. Do not probe or
            # benchmark again on subsequent forwards.
            route = frozen
            if route == "native":
                try:
                    from ..esa.native import elastic_linear_packed as packed_linear
                    native_available = True
                except (ImportError, AttributeError, RuntimeError):
                    native_available = False
        elif torch.is_grad_enabled():
            # Packed native ElasticLinear is inference-only. Keep training on
            # the transparent PyTorch graph and do not freeze an inference route.
            route = "pytorch"
        else:
            native_available, packed_linear = native_impl_available()
            if native_available and packed_linear is not None:
                # Materialize/copy steady-state operands before qualification so
                # the one-time benchmark compares execution, not lazy setup.
                weight = self._materialized_weight(x.device, dtype)
                bias = None if self.bias is None else self.bias.to(device=x.device, dtype=dtype)
                native_bias = (
                    x_compute.new_empty(0) if bias is None else bias
                )
                route = EXECUTION_PLANNER.qualify_operator_once(
                    self,
                    "elastic_linear",
                    x_compute,
                    {
                        "native": lambda: packed_linear(
                            x_compute, self.packed_weight, self.scales, native_bias,
                            self.config.bits, self.config.group_size,
                            self.out_features, self.in_features,
                        ),
                        "pytorch": lambda: F.linear(x_compute, weight, bias),
                    },
                    requested_backend="auto",
                    native_available=True,
                    native_supports_training=False,
                    training=False,
                    extra=extra,
                    default_auto="pytorch",
                )
            else:
                route = EXECUTION_PLANNER.select_operator_once(
                    self,
                    "elastic_linear",
                    x_compute,
                    requested_backend="auto",
                    native_available=False,
                    native_supports_training=False,
                    training=False,
                    extra=extra,
                    default_auto="pytorch",
                )

        if route == "native":
            if packed_linear is None:
                native_available, packed_linear = native_impl_available()
            if native_available and packed_linear is not None:
                bias = (
                    x_compute.new_empty(0)
                    if self.bias is None
                    else self.bias.to(device=x_compute.device, dtype=x_compute.dtype)
                )
                output = packed_linear(
                    x_compute, self.packed_weight, self.scales, bias,
                    self.config.bits, self.config.group_size,
                    self.out_features, self.in_features,
                )
                return output.to(x.dtype) if output.dtype != x.dtype else output

            if policy == "native":
                raise RuntimeError(
                    "ElasticBit backend='native' requires the packed MLBricks CUDA extension"
                )

            # A frozen native route can only become invalid after an explicit
            # environment/device change. Clear this element and safely demote.
            EXECUTION_PLANNER.clear_owner_routes(self)
            route = EXECUTION_PLANNER.select_operator_once(
                self, "elastic_linear", x_compute, requested_backend="auto",
                native_available=False, native_supports_training=False, training=False,
                extra=extra, default_auto="pytorch",
            )

        weight = self._materialized_weight(x.device, dtype)
        bias = None if self.bias is None else self.bias.to(device=x.device, dtype=dtype)
        output = F.linear(x_compute, weight, bias)
        return output.to(x.dtype) if output.dtype != x.dtype else output

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bits={self.config.bits}, group_size={self.config.group_size}, "
            f"bias={self.bias is not None}"
        )


class ElasticEmbedding(_ElasticWeightMixin, nn.Module):
    """Inference-only packed replacement for ``nn.Embedding``."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        config: ElasticBitConfig | None = None,
    ) -> None:
        super().__init__()
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.config = config or ElasticBitConfig()
        self.original_numel = num_embeddings * embedding_dim
        groups = (self.original_numel + self.config.group_size - 1) // self.config.group_size
        packed_bytes = (groups * self.config.group_size * self.config.bits + 7) // 8
        self.register_buffer("packed_weight", torch.zeros(packed_bytes, dtype=torch.uint8))
        self.register_buffer("scales", torch.zeros(groups, dtype=self.config.scale_dtype))
        self.register_buffer(
            "weight_shape",
            torch.tensor([num_embeddings, embedding_dim], dtype=torch.int64),
        )
        self._init_cache()

    def _assign_packed(self, packed: PackedElasticBit) -> None:
        self.packed_weight = packed.packed
        self.scales = packed.scales
        self.weight_shape = torch.tensor(packed.shape, dtype=torch.int64)
        self.original_numel = packed.original_numel
        self.clear_cache()

    @classmethod
    def from_embedding(
        cls,
        layer: nn.Embedding,
        config: ElasticBitConfig | None = None,
    ) -> "ElasticEmbedding":
        result = cls(layer.num_embeddings, layer.embedding_dim, config=config)
        result._assign_packed(quantize_tensor(layer.weight, result.config))
        return result.to(device=layer.weight.device)

    def packed(self) -> PackedElasticBit:
        return PackedElasticBit(
            self.packed_weight,
            self.scales,
            tuple(int(v) for v in self.weight_shape.detach().cpu().tolist()),
            self.config.bits,
            self.config.group_size,
            self.original_numel,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.config.backend == "native":
            raise RuntimeError(
                "ElasticEmbedding does not yet have a packed native kernel; use auto/pytorch"
            )
        dtype = self.config.compute_dtype or torch.get_default_dtype()
        weight = self._materialized_weight(input_ids.device, dtype)
        return F.embedding(input_ids, weight)


class ElasticBit:
    """ElasticBit 4-32 bit runtime plus MLBricks compatibility helpers.

    The standalone ElasticBit 0.2 API is available as class attributes:
    ``ElasticBit.RuntimeMatrix``, ``ElasticBit.NativeFP16Matrix`` and
    ``ElasticBit.bitsAnaliser``. Existing MLBricks tensor/module quantization
    helpers remain available for compatibility and PyTorch fallback execution.
    """

    RuntimeMatrix = RuntimeMatrix
    NativeFP16Matrix = NativeFP16Matrix
    bitsAnaliser = staticmethod(bitsAnaliser)
    native_runtime_available = staticmethod(native_runtime_available)

    def __init__(
        self,
        bits: int = 4,
        group_size: int = 128,
        *,
        scale_dtype: torch.dtype = torch.float16,
        compute_dtype: torch.dtype | None = None,
        cache_dequantized: bool = True,
        runtime: str = "auto",
        backend: str = "auto",
    ) -> None:
        self.config = ElasticBitConfig(
            bits=bits,
            group_size=group_size,
            scale_dtype=scale_dtype,
            compute_dtype=compute_dtype,
            cache_dequantized=cache_dequantized,
            runtime=runtime,
            backend=backend,
        )

    @property
    def backend(self) -> str:
        return self.config.backend

    def set_backend(self, backend: str):
        value = normalize_backend(backend, warn_legacy=True)
        data = self.config.to_manifest()
        data["backend"] = value
        data["runtime"] = "auto"
        self.config = ElasticBitConfig.from_manifest(data)
        return self

    def resolved_backend(self) -> str:
        if self.backend == "pytorch":
            return "pytorch"
        if self.backend == "native":
            return "native-required"
        routes = EXECUTION_PLANNER.owner_routes(self)
        if routes:
            return "+".join(sorted(set(routes.values())))
        return "planner(auto; qualify-once)"

    @property
    def bits(self) -> int:
        return self.config.bits

    @property
    def group_size(self) -> int:
        return self.config.group_size

    def quantize(self, tensor: torch.Tensor) -> PackedElasticBit:
        return quantize_tensor(tensor, self.config)

    def dequantize(
        self,
        value: PackedElasticBit,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        return dequantize_tensor(value, device=device, dtype=dtype)

    def linear(self, layer: nn.Linear) -> ElasticLinear:
        return ElasticLinear.from_linear(layer, self.config)

    def embedding(self, layer: nn.Embedding) -> ElasticEmbedding:
        return ElasticEmbedding.from_embedding(layer, self.config)

    def quantize_module(
        self,
        module: nn.Module,
        *,
        include_embeddings: bool = False,
        skip_names: Iterable[str] = (),
    ) -> nn.Module:
        return quantize_module(
            module,
            self.config,
            include_embeddings=include_embeddings,
            skip_names=skip_names,
        )

    apply = quantize_module

    def __repr__(self) -> str:
        return (
            f"ElasticBit(bits={self.config.bits}, "
            f"group_size={self.config.group_size}, "
            f"scale_dtype={self.config.scale_dtype}, "
            f"compute_dtype={self.config.compute_dtype}, "
            f"cache_dequantized={self.config.cache_dequantized}, "
            f"runtime={self.config.runtime!r}, backend={self.config.backend!r})"
        )


def _weight_identity(layer: nn.Module) -> int | None:
    weight = getattr(layer, "weight", None)
    return id(weight) if isinstance(weight, nn.Parameter) else None


def _share_packed(target: ElasticLinear | ElasticEmbedding, source: ElasticLinear | ElasticEmbedding) -> None:
    target.packed_weight = source.packed_weight
    target.scales = source.scales
    target.weight_shape = source.weight_shape
    target.original_numel = source.original_numel
    # Tied modules share both packed storage and the lazy materialized weight.
    target._dequant_cache = source._dequant_cache


def quantize_module(
    module: nn.Module,
    config: ElasticBitConfig | None = None,
    *,
    include_embeddings: bool = False,
    skip_names: Iterable[str] = (),
) -> nn.Module:
    """Recursively replace supported layers while preserving shared weights.

    If embeddings are excluded, a Linear whose weight is tied to an embedding
    is left untouched instead of creating a duplicate packed LM-head copy.
    When embeddings are included, tied embedding/LM-head modules share the same
    packed buffers.
    """
    config = config or ElasticBitConfig()
    skipped = set(skip_names)
    embedding_weight_ids = {
        ident
        for child in module.modules()
        if isinstance(child, nn.Embedding)
        for ident in [_weight_identity(child)]
        if ident is not None
    }
    shared: dict[int, ElasticLinear | ElasticEmbedding] = {}

    def recurse(parent: nn.Module) -> None:
        for name, child in list(parent.named_children()):
            if name in skipped:
                continue
            ident = _weight_identity(child)
            replacement: ElasticLinear | ElasticEmbedding | None = None

            if isinstance(child, nn.Linear):
                if not include_embeddings and ident in embedding_weight_ids:
                    continue
                replacement = ElasticLinear(
                    child.in_features,
                    child.out_features,
                    bias=child.bias is not None,
                    config=config,
                )
                if ident is not None and ident in shared:
                    _share_packed(replacement, shared[ident])
                else:
                    replacement._assign_packed(quantize_tensor(child.weight, config))
                    if ident is not None:
                        shared[ident] = replacement
                if child.bias is not None:
                    replacement.bias.data.copy_(child.bias.detach().to(replacement.bias))

            elif include_embeddings and isinstance(child, nn.Embedding):
                replacement = ElasticEmbedding(
                    child.num_embeddings,
                    child.embedding_dim,
                    config=config,
                )
                if ident is not None and ident in shared:
                    _share_packed(replacement, shared[ident])
                else:
                    replacement._assign_packed(quantize_tensor(child.weight, config))
                    if ident is not None:
                        shared[ident] = replacement

            if replacement is not None:
                source_weight = getattr(child, "weight", None)
                if isinstance(source_weight, nn.Parameter):
                    replacement.to(device=source_weight.device)
                setattr(parent, name, replacement)
            else:
                recurse(child)

    recurse(module)
    return module


def elasticbit_manifest(module: nn.Module) -> dict[str, Any] | None:
    """Return serialization metadata for all ElasticBit modules in ``module``."""
    entries: list[dict[str, Any]] = []
    shared_ids: dict[tuple[int, int, int], str] = {}
    next_group = 0
    for name, child in module.named_modules():
        if not isinstance(child, (ElasticLinear, ElasticEmbedding)):
            continue
        storage = child.packed_weight.untyped_storage()
        key = (int(storage.data_ptr()), int(storage.nbytes()), int(child.packed_weight.storage_offset()))
        group = shared_ids.get(key)
        if group is None:
            group = f"w{next_group}"
            next_group += 1
            shared_ids[key] = group
        entry: dict[str, Any] = {
            "name": name,
            "kind": "linear" if isinstance(child, ElasticLinear) else "embedding",
            "config": child.config.to_manifest(),
            "shared_group": group,
        }
        if isinstance(child, ElasticLinear):
            entry.update(
                in_features=int(child.in_features),
                out_features=int(child.out_features),
                bias=child.bias is not None,
            )
        else:
            entry.update(
                num_embeddings=int(child.num_embeddings),
                embedding_dim=int(child.embedding_dim),
            )
        entries.append(entry)
    if not entries:
        return None
    return {"type": "elasticbit", "format_version": 1, "modules": entries}


def _replace_named_module(root: nn.Module, name: str, replacement: nn.Module) -> None:
    parts = name.split(".") if name else []
    if not parts:
        raise ValueError("Cannot replace the root module from an ElasticBit manifest")
    parent = root
    for part in parts[:-1]:
        parent = parent._modules[part]
    parent._modules[parts[-1]] = replacement


def restore_elasticbit_modules(module: nn.Module, manifest: dict[str, Any] | None) -> nn.Module:
    """Recreate ElasticBit wrappers before loading a packed state dict."""
    if not manifest:
        return module
    if manifest.get("type") != "elasticbit":
        raise ValueError(f"Unsupported quantization manifest: {manifest.get('type')!r}")

    shared: dict[str, ElasticLinear | ElasticEmbedding] = {}
    for entry in manifest.get("modules", []):
        config = ElasticBitConfig.from_manifest(entry.get("config", {}))
        kind = entry.get("kind")
        if kind == "linear":
            replacement: ElasticLinear | ElasticEmbedding = ElasticLinear(
                int(entry["in_features"]),
                int(entry["out_features"]),
                bias=bool(entry.get("bias", True)),
                config=config,
            )
        elif kind == "embedding":
            replacement = ElasticEmbedding(
                int(entry["num_embeddings"]),
                int(entry["embedding_dim"]),
                config=config,
            )
        else:
            raise ValueError(f"Unknown ElasticBit module kind: {kind!r}")

        group = str(entry.get("shared_group", ""))
        if group and group in shared:
            _share_packed(replacement, shared[group])
        elif group:
            shared[group] = replacement
        _replace_named_module(module, str(entry["name"]), replacement)
    return module


__all__ = [
    "ElasticBit",
    "ElasticBitConfig",
    "PackedElasticBit",
    "ElasticLinear",
    "ElasticEmbedding",
    "quantize_tensor",
    "dequantize_tensor",
    "quantize_module",
    "elasticbit_manifest",
    "restore_elasticbit_modules",
]
