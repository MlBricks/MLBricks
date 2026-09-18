# Copyright 2026 Zameer Hussain and Akhtar Hussain
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# See LICENSE and LICENSING_NOTICE.md; commercial use requires a separate written license.

"""ElasticBit adaptive threshold-driven weight compression.

Public contract
---------------
ElasticBit has no fixed-bit quantization API.  The user supplies representative
calibration data and an allowed error threshold.  ElasticBit selects the
smallest safe storage width automatically.

Storage and execution are intentionally separate:

* storage search: 3..15 bit, with FP16 as the threshold-preserving fallback;
* ``hardwareNative`` decode: T4 uses the validated W4A16/W8A16/W16A16 map
  with one-warp-per-output-row FastWarp GEMV; other CUDA GPUs auto-tune the
  legal built-in execution widths per matrix shape;
* ``fullPrecision`` decode: every stored width executes as W16A16 FastWarp GEMV;
* prefill materializes FP16 weights on-device and uses framework/vendor GEMM;
* activations stay FP16 in every execution path.

The analyzer validates both low-bit and full-precision arithmetic before
accepting a storage width, so policy switching or hardware widening never
requires recalibration or recompression.
"""
from __future__ import annotations

from dataclasses import dataclass
import gc
import io
import os
import json
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping, Sequence
import zipfile

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .native_api import available as _nativeAvailable
from .native_api import importError as _nativeImportError
from .native_api import requireNative as _requireNative


_VALID_POLICIES = {"hardwareNative", "fullPrecision"}
_MODEL_FORMAT = "ElasticBitModel"
_MODEL_FORMAT_VERSION = 1


def _checkPolicy(value: str) -> str:
    value = str(value)
    if value not in _VALID_POLICIES:
        raise ValueError("decodePolicy must be 'hardwareNative' or 'fullPrecision'")
    return value


def _toNumpy2D(value: Any, *, name: str) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().float().cpu().numpy()
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D matrix")
    return np.ascontiguousarray(array, dtype=np.float32)


def _toCalibrationBatches(calibrationData: Any) -> list[Any]:
    if torch.is_tensor(calibrationData) or isinstance(calibrationData, Mapping):
        return [calibrationData]
    if isinstance(calibrationData, (str, bytes)):
        raise TypeError("calibrationData must contain model inputs, not text")
    try:
        batches = list(calibrationData)
    except TypeError as exc:
        raise TypeError("calibrationData must be a model input or an iterable of model inputs") from exc
    if not batches:
        raise ValueError("calibrationData cannot be empty")
    return batches


def _callModel(model: nn.Module, batch: Any) -> Any:
    if isinstance(batch, Mapping):
        return model(**batch)
    if isinstance(batch, tuple):
        return model(*batch)
    return model(batch)


def _moveBatch(batch: Any, device: torch.device) -> Any:
    if torch.is_tensor(batch):
        return batch.to(device)
    if isinstance(batch, Mapping):
        return {key: _moveBatch(value, device) for key, value in batch.items()}
    if isinstance(batch, tuple):
        return tuple(_moveBatch(value, device) for value in batch)
    if isinstance(batch, list):
        return [_moveBatch(value, device) for value in batch]
    return batch


def _relativeL2Torch(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    reference32 = reference.float()
    candidate32 = candidate.float()
    denominator = torch.linalg.vector_norm(reference32).clamp_min(1.0e-30)
    return float((torch.linalg.vector_norm(candidate32 - reference32) / denominator).item())


@torch.inference_mode()
def _selectBitsTorch(
    weight: torch.Tensor,
    calibration: np.ndarray,
    threshold: float,
) -> tuple[int, float, np.ndarray | None, np.ndarray | None]:
    """Select ElasticBit storage width on the model GPU.

    This is the model-compression fast path.  It preserves the public
    threshold-driven contract but avoids the old scalar CPU analyzer, whose
    cost was O(bits * calibration_rows * weight_elements).  The reference is
    the real FP16 framework GEMM used by native model execution.  For a
    compressed selection, the exact selected integer codes and row scales are
    returned so model compression can pack them directly instead of quantizing
    the original FP16 weight a second time.
    """
    if weight.device.type != "cuda":
        raise RuntimeError("ElasticBit GPU analyzer requires CUDA weights")
    if weight.dtype != torch.float16:
        raise RuntimeError("ElasticBit GPU analyzer requires FP16 weights")
    if not np.isfinite(threshold) or float(threshold) < 0.0:
        raise ValueError("threshold must be a finite non-negative value")

    calibrationTensor = torch.from_numpy(
        np.ascontiguousarray(calibration, dtype=np.float32)
    ).to(device=weight.device, dtype=torch.float16)

    # Native reference: FP16 activations + FP16 weights + vendor GEMM.
    reference = F.linear(calibrationTensor, weight).detach()

    weight32 = weight.detach().float()
    maxAbs = weight32.abs().amax(dim=1)

    for bits in range(3, 16):
        qmax = float((1 << (bits - 1)) - 1)
        scale = torch.where(
            maxAbs > 0.0,
            maxAbs / qmax,
            torch.ones_like(maxAbs),
        )

        quantized = torch.round(weight32 / scale[:, None]).clamp(
            min=-qmax, max=qmax
        )
        dequantized16 = (quantized * scale[:, None]).to(torch.float16)

        # fullPrecision / W16A16 execution.
        fullOutput = F.linear(calibrationTensor, dequantized16)
        fullError = _relativeL2Torch(reference, fullOutput)

        if bits <= 8:
            # hardwareNative low-bit execution keeps activations FP16 and
            # accumulates the exact integer codes against FP16 activations,
            # followed by the per-row scale and FP16 output rounding.
            integerDot = torch.matmul(
                calibrationTensor.float(),
                quantized.transpose(0, 1),
            )
            hardwareOutput = (integerDot * scale[None, :]).to(torch.float16)
            hardwareError = _relativeL2Torch(reference, hardwareOutput)
        else:
            # 9..15-bit storage executes through W16A16.
            hardwareError = fullError

        error = max(hardwareError, fullError)
        if error <= float(threshold):
            # Preserve the exact analyzer result. int16 covers the full signed
            # range needed by ElasticBit's 3..15-bit storage search. Moving the
            # chosen codes/scales to CPU is substantially cheaper than copying
            # the original FP16 weight to CPU and quantizing it again there.
            selectedCodes = np.ascontiguousarray(
                quantized.to(torch.int16).cpu().numpy(), dtype=np.int16
            )
            selectedScales = np.ascontiguousarray(
                scale.float().cpu().numpy(), dtype=np.float32
            )
            return bits, float(error), selectedCodes, selectedScales

        del quantized, dequantized16, fullOutput
        if bits <= 8:
            del integerDot, hardwareOutput

    # FP16 fallback is exact relative to the native FP16 reference.
    return 16, 0.0, None, None


def _moduleDevice(module: nn.Module) -> torch.device:
    for parameter in module.parameters():
        return parameter.device
    for buffer in module.buffers():
        return buffer.device
    return torch.device("cpu")


def _replaceNamedModule(root: nn.Module, name: str, replacement: nn.Module) -> None:
    parts = name.split(".") if name else []
    if not parts:
        raise ValueError("ElasticBit cannot replace the root module")
    parent = root
    for part in parts[:-1]:
        parent = parent._modules[part]
    parent._modules[parts[-1]] = replacement


@dataclass(frozen=True)
class BitCandidate:
    bits: int
    hardwareNativeError: float
    fullPrecisionError: float
    error: float
    storageBytes: int
    memoryReduction: float
    passes: bool


@dataclass(frozen=True)
class BitAnalysis:
    threshold: float
    selectedBits: int
    selectedError: float
    candidates: tuple[BitCandidate, ...]


@dataclass(frozen=True)
class BackendInfo:
    available: bool
    device: str | None
    deviceIndex: int | None
    architecture: str | None
    weightWidths: tuple[int, ...]
    activateDType: str | None
    prefill: str | None
    decodePlanner: str | None
    validated: bool
    error: str | None = None

    def __str__(self) -> str:
        if not self.available:
            return f"ElasticBitBackend(unavailable, error={self.error!r})"
        return (
            "ElasticBitBackend("
            f"device={self.device!r}, architecture={self.architecture!r}, "
            f"weightWidths={self.weightWidths}, activateDType={self.activateDType!r}, "
            f"prefill={self.prefill!r}, decodePlanner={self.decodePlanner!r}, "
            f"validated={self.validated})"
        )


class RuntimeMatrix:
    """Advanced single-matrix ElasticBit runtime.

    Construction is intentionally restricted to :meth:`compress` and
    :meth:`load`; there is no constructor that accepts a user-selected bit width.
    """

    def __init__(self, native: Any) -> None:
        self._native = native

    @classmethod
    def compress(
        cls,
        weights: Any,
        calibrationData: Any,
        threshold: float,
        decodePolicy: str = "hardwareNative",
    ) -> "RuntimeMatrix":
        native = _requireNative()
        policy = _checkPolicy(decodePolicy)
        w = _toNumpy2D(weights, name="weights")
        c = _toNumpy2D(calibrationData, name="calibrationData")
        return cls(native.RuntimeMatrix.compress(w, c, float(threshold), policy))

    @classmethod
    def _compressSelected(
        cls,
        weights: Any,
        selectedBits: int,
        threshold: float,
        selectedError: float,
        decodePolicy: str = "hardwareNative",
    ) -> "RuntimeMatrix":
        native = _requireNative()
        policy = _checkPolicy(decodePolicy)
        w = _toNumpy2D(weights, name="weights")
        return cls(native.RuntimeMatrix._compressSelected(
            w, int(selectedBits), float(threshold), float(selectedError), policy
        ))

    @classmethod
    def _compressSelectedCodes(
        cls,
        codes: np.ndarray,
        scales: np.ndarray,
        selectedBits: int,
        threshold: float,
        selectedError: float,
        decodePolicy: str = "hardwareNative",
    ) -> "RuntimeMatrix":
        """Pack an already-selected analyzer result without requantizing.

        Internal model-compression fast path. The public API remains
        threshold-driven and never exposes a user-selected bit width.
        """
        native = _requireNative()
        policy = _checkPolicy(decodePolicy)
        q = np.ascontiguousarray(codes, dtype=np.int16)
        s = np.ascontiguousarray(scales, dtype=np.float32)
        if q.ndim != 2:
            raise ValueError("codes must be a 2D matrix")
        if s.ndim != 1 or s.shape[0] != q.shape[0]:
            raise ValueError("scales must contain one value per weight row")
        return cls(native.RuntimeMatrix._compressSelectedCodes(
            q, s, int(selectedBits), float(threshold), float(selectedError), policy
        ))

    @classmethod
    def load(
        cls,
        path: str | Path,
        decodePolicy: str = "hardwareNative",
    ) -> "RuntimeMatrix":
        native = _requireNative()
        policy = _checkPolicy(decodePolicy)
        return cls(native.RuntimeMatrix.load(str(path), policy))

    def forward(self, x: Any) -> Any:
        if torch.is_tensor(x):
            if x.is_cuda and x.dtype == torch.float16 and x.numel() == self.cols:
                return self._native.forwardTorch(x.contiguous())
            values = np.asarray(x.detach().float().cpu().numpy(), dtype=np.float32).reshape(-1)
            output = self._native.forward(np.ascontiguousarray(values))
            return torch.from_numpy(np.asarray(output)).to(device=x.device, dtype=x.dtype)
        values = np.asarray(x, dtype=np.float32).reshape(-1)
        return np.asarray(self._native.forward(np.ascontiguousarray(values)))

    def benchmark(self, x: Any, iterations: int = 500) -> float:
        if torch.is_tensor(x):
            x = x.detach().float().cpu().numpy()
        values = np.ascontiguousarray(np.asarray(x, dtype=np.float32).reshape(-1))
        return float(self._native.benchmark(values, int(iterations)))

    def dequantize(self) -> np.ndarray:
        return np.asarray(self._native.dequantize(), dtype=np.float32)

    def _materializeTorch(self) -> torch.Tensor:
        return self._native._materializeTorch()

    def save(self, path: str | Path) -> None:
        self._native.save(str(path))

    def setDecodePolicy(self, decodePolicy: str) -> "RuntimeMatrix":
        self._native.setDecodePolicy(_checkPolicy(decodePolicy))
        return self

    @property
    def rows(self) -> int:
        return int(self._native.rows)

    @property
    def cols(self) -> int:
        return int(self._native.cols)

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(int(v) for v in self._native.shape)

    @property
    def storageBits(self) -> int:
        return int(self._native.storageBits)

    @property
    def executionWidth(self) -> int:
        return int(self._native.executionWidth)

    @property
    def executionPlanner(self) -> str:
        return str(self._native.executionPlanner)

    @property
    def decodePolicy(self) -> str:
        return str(self._native.decodePolicy)

    @property
    def activateDType(self) -> str:
        return str(self._native.activateDType)

    @property
    def storageBytes(self) -> int:
        return int(self._native.storageBytes)

    @property
    def executionBytes(self) -> int:
        return int(self._native.executionBytes)

    @property
    def originalBytes(self) -> int:
        return int(self._native.originalBytes)

    @property
    def memoryReduction(self) -> float:
        return float(self._native.memoryReduction)

    @property
    def threshold(self) -> float:
        return float(self._native.threshold)

    @property
    def error(self) -> float:
        return float(self._native.error)


class _ElasticLinear(nn.Module):
    """Internal model wrapper. Not part of the public import surface."""

    def __init__(
        self,
        matrix: RuntimeMatrix,
        inFeatures: int,
        outFeatures: int,
        bias: torch.Tensor | None,
    ) -> None:
        super().__init__()
        self.matrix = matrix
        self.inFeatures = int(inFeatures)
        self.outFeatures = int(outFeatures)
        if bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(bias.detach().clone(), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.inFeatures:
            raise ValueError(
                f"ElasticBit Linear expected last dimension {self.inFeatures}, got {x.shape[-1]}"
            )
        rows = x.numel() // self.inFeatures

        # M=1 decode enters RuntimeMatrix directly. W4A16/W8A16/W16A16 all
        # use the validated one-warp-per-output-row FastWarp CUDA topology.
        if rows == 1 and x.is_cuda and x.dtype == torch.float16:
            y = self.matrix.forward(x)
            if self.bias is not None:
                y = y + self.bias.to(device=y.device, dtype=y.dtype)
            return y

        # Prefill stays on the framework/vendor GEMM path. Low-bit execution
        # weights expand transiently on-device; W16A16 returns a zero-copy view
        # of its existing persistent FP16 execution allocation.
        weight = self.matrix._materializeTorch()
        if weight.dtype != x.dtype:
            weight = weight.to(dtype=x.dtype)
        bias = self.bias
        if bias is not None and (bias.device != x.device or bias.dtype != x.dtype):
            bias = bias.to(device=x.device, dtype=x.dtype)
        return F.linear(x, weight, bias)


class _ModelElasticBit:
    def __init__(
        self,
        model: nn.Module,
        threshold: float,
        decodePolicy: str,
        compressedNames: Sequence[str],
        skippedNames: Sequence[str],
    ) -> None:
        self._model = model
        self.threshold = float(threshold)
        self.decodePolicy = _checkPolicy(decodePolicy)
        self.compressedNames = tuple(compressedNames)
        self.skippedNames = tuple(skippedNames)

    def _layers(self) -> list[tuple[str, _ElasticLinear]]:
        modules = dict(self._model.named_modules())
        return [
            (name, modules[name])
            for name in self.compressedNames
            if isinstance(modules.get(name), _ElasticLinear)
        ]

    def setDecodePolicy(self, decodePolicy: str) -> "_ModelElasticBit":
        policy = _checkPolicy(decodePolicy)
        seen: set[int] = set()
        for _, layer in self._layers():
            ident = id(layer.matrix)
            if ident in seen:
                continue
            layer.matrix.setDecodePolicy(policy)
            seen.add(ident)
        self.decodePolicy = policy
        return self

    def summary(self) -> str:
        backend = ElasticBit.backend()
        lines = [
            "ElasticBit Summary",
            "────────────────────────────────────────────────────────────",
            f"Threshold       : {self.threshold}",
            f"Decode Policy   : {self.decodePolicy}",
            f"Backend         : {backend.architecture or 'unavailable'}",
            f"Prefill         : {backend.prefill or 'unknown'}",
            f"Decode Planner  : {backend.decodePlanner or 'unknown'}",
            f"Activate DType  : {backend.activateDType or 'unknown'}",
            "",
            f"{'Layer':36s} {'Stored':>8s} {'Execute':>10s} {'Planner':>14s} {'Error':>10s}",
        ]
        for name, layer in self._layers():
            matrix = layer.matrix
            lines.append(
                f"{name[:36]:36s} {str(matrix.storageBits) + '-bit':>8s} "
                f"{'W' + str(matrix.executionWidth) + 'A16':>10s} "
                f"{matrix.executionPlanner:>14s} {matrix.error:10.6f}"
            )
        if self.skippedNames:
            lines.extend(["", "Uncompressed (no safe calibration path):"])
            lines.extend(f"  {name}" for name in self.skippedNames)
        text = "\n".join(lines)
        print(text)
        return text


class ElasticBit:
    """Clean ElasticBit public namespace."""

    RuntimeMatrix = RuntimeMatrix

    @staticmethod
    def analyze(weights: Any, calibrationData: Any, threshold: float) -> BitAnalysis:
        native = _requireNative()
        w = _toNumpy2D(weights, name="weights")
        c = _toNumpy2D(calibrationData, name="calibrationData")
        raw = native.analyze(w, c, float(threshold))
        candidates = tuple(
            BitCandidate(
                bits=int(item["bits"]),
                hardwareNativeError=float(item["hardwareNativeError"]),
                fullPrecisionError=float(item["fullPrecisionError"]),
                error=float(item["error"]),
                storageBytes=int(item["storageBytes"]),
                memoryReduction=float(item["memoryReduction"]),
                passes=bool(item["passes"]),
            )
            for item in raw["candidates"]
        )
        return BitAnalysis(
            threshold=float(raw["threshold"]),
            selectedBits=int(raw["selectedBits"]),
            selectedError=float(raw["selectedError"]),
            candidates=candidates,
        )

    @staticmethod
    def compressMatrix(
        weights: Any,
        calibrationData: Any,
        threshold: float,
        decodePolicy: str = "hardwareNative",
    ) -> RuntimeMatrix:
        return RuntimeMatrix.compress(
            weights,
            calibrationData,
            threshold,
            decodePolicy=decodePolicy,
        )

    @staticmethod
    def loadMatrix(
        path: str | Path,
        decodePolicy: str = "hardwareNative",
    ) -> RuntimeMatrix:
        return RuntimeMatrix.load(path, decodePolicy=decodePolicy)

    @staticmethod
    def backend() -> BackendInfo:
        if not _nativeAvailable():
            error = _nativeImportError()
            return BackendInfo(
                available=False,
                device=None,
                deviceIndex=None,
                architecture=None,
                weightWidths=(),
                activateDType=None,
                prefill=None,
                decodePlanner=None,
                validated=False,
                error=None if error is None else str(error),
            )
        raw = _requireNative().backendInfo()
        return BackendInfo(
            available=True,
            device=str(raw["device"]),
            deviceIndex=int(raw["deviceIndex"]),
            architecture=str(raw["architecture"]),
            weightWidths=tuple(int(v) for v in raw["weightWidths"]),
            activateDType=str(raw["activateDType"]),
            prefill=str(raw["prefill"]),
            decodePlanner=str(raw["decodePlanner"]),
            validated=bool(raw["validated"]),
        )

    @staticmethod
    def compress(
        model: nn.Module,
        calibrationData: Any,
        threshold: float,
        decodePolicy: str = "hardwareNative",
    ) -> nn.Module:
        if not isinstance(model, nn.Module):
            raise TypeError("ElasticBit.compress() expects a torch.nn.Module")
        _requireNative()
        policy = _checkPolicy(decodePolicy)
        batches = _toCalibrationBatches(calibrationData)
        device = _moduleDevice(model)
        if device.type != "cuda":
            raise RuntimeError("ElasticBit model compression currently requires a CUDA model")

        # The validated runtime is weight-only FP16 activation execution.
        for module in model.modules():
            if isinstance(module, nn.Linear) and module.weight.dtype != torch.float16:
                raise RuntimeError(
                    "ElasticBit.compress() currently requires FP16 Linear weights. "
                    "Convert the model to float16 before compression."
                )

        embeddingWeightIds = {
            id(module.weight)
            for module in model.modules()
            if isinstance(module, nn.Embedding)
        }

        collected: dict[str, list[np.ndarray]] = {}
        handles = []
        maxRows = 32

        def makeHook(name: str):
            def hook(_module: nn.Module, inputs: tuple[Any, ...]) -> None:
                if not inputs or not torch.is_tensor(inputs[0]):
                    return
                tensor = inputs[0]
                rows = tensor.detach().float().reshape(-1, tensor.shape[-1])
                have = sum(item.shape[0] for item in collected.get(name, []))
                remaining = maxRows - have
                if remaining <= 0:
                    return
                sample = rows[:remaining].cpu().numpy().astype(np.float32, copy=True)
                collected.setdefault(name, []).append(sample)
            return hook

        originalLinears: list[tuple[str, nn.Linear]] = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                originalLinears.append((name, module))
                if id(module.weight) not in embeddingWeightIds:
                    handles.append(module.register_forward_pre_hook(makeHook(name)))

        model.eval()
        try:
            with torch.inference_mode():
                for rawBatch in batches:
                    _callModel(model, _moveBatch(rawBatch, device))
        finally:
            for handle in handles:
                handle.remove()

        # Combine calibration from every use of a shared Linear weight before
        # selecting its precision, so one shared RuntimeMatrix satisfies all uses.
        # Keep only names/ids here: retaining every original Linear object would
        # keep its FP16 weight alive after replacement and defeat streaming compression.
        namesByWeight: dict[int, list[str]] = {}
        for name, module in originalLinears:
            namesByWeight.setdefault(id(module.weight), []).append(name)
        del originalLinears
        handles.clear()

        compressedNames: list[str] = []
        skippedNames: list[str] = []
        progress = os.environ.get("MLBRICKS_ELASTICBIT_PROGRESS", "").strip().lower() in {
            "1", "true", "yes", "on"
        }
        compressible = [
            (weightId, names)
            for weightId, names in namesByWeight.items()
            if weightId not in embeddingWeightIds
        ]

        for matrixIndex, (weightId, names) in enumerate(compressible, start=1):
            module = model.get_submodule(names[0])
            if not isinstance(module, nn.Linear):
                raise RuntimeError(f"ElasticBit expected Linear module at {names[0]}")

            samples = [
                item
                for name in names
                for item in collected.get(name, [])
            ]
            if not samples:
                skippedNames.extend(names)
                continue
            calibration = np.ascontiguousarray(np.concatenate(samples, axis=0), dtype=np.float32)

            if progress:
                freeBytes, totalBytes = torch.cuda.mem_get_info(device)
                print(
                    f"[ElasticBit] {matrixIndex}/{len(compressible)} analyzing {names[0]} "
                    f"shape={tuple(module.weight.shape)} rows={calibration.shape[0]} "
                    f"free={freeBytes / (1024**3):.2f}/{totalBytes / (1024**3):.2f} GiB",
                    flush=True,
                )

            selectedBits, selectedError, selectedCodes, selectedScales = _selectBitsTorch(
                module.weight, calibration, float(threshold)
            )

            # _selectBitsTorch uses PyTorch CUDA tensors while the native RuntimeMatrix
            # owns its execution buffers via raw cudaMalloc. Return cached analysis
            # blocks to CUDA before the native allocation so the two allocators do not
            # artificially compete for the same free VRAM.
            torch.cuda.empty_cache()

            if progress:
                print(
                    f"[ElasticBit] {matrixIndex}/{len(compressible)} selected "
                    f"{selectedBits}-bit error={selectedError:.6g}",
                    flush=True,
                )

            if selectedBits <= 15:
                # Reuse the exact analyzer-selected integer codes/scales. This avoids
                # the old GPU analyze -> discard -> FP16 GPU->CPU -> requantize path.
                assert selectedCodes is not None and selectedScales is not None
                matrix = RuntimeMatrix._compressSelectedCodes(
                    selectedCodes,
                    selectedScales,
                    selectedBits,
                    float(threshold),
                    selectedError,
                    decodePolicy=policy,
                )
                weights = None
            else:
                # FP16 fallback has no quantized analyzer payload. Preserve the
                # original FP16 values exactly in MLB4.
                weights = np.ascontiguousarray(
                    module.weight.detach().cpu().float().numpy(), dtype=np.float32
                )
                matrix = RuntimeMatrix._compressSelected(
                    weights,
                    selectedBits,
                    float(threshold),
                    selectedError,
                    decodePolicy=policy,
                )

            # Replace every Linear sharing this weight immediately.  This releases
            # the original FP16 parameter group before the next compressed matrix is
            # created, keeping peak VRAM close to the final compressed model instead
            # of full-model + compressed-model memory.
            for name in names:
                current = model.get_submodule(name)
                if not isinstance(current, nn.Linear):
                    raise RuntimeError(f"ElasticBit expected Linear module at {name}")
                replacement = _ElasticLinear(
                    matrix=matrix,
                    inFeatures=current.in_features,
                    outFeatures=current.out_features,
                    bias=current.bias,
                ).to(device=device)
                _replaceNamedModule(model, name, replacement)
                compressedNames.append(name)
                collected.pop(name, None)
                del current, replacement

            del module, weights, selectedCodes, selectedScales, calibration, samples, matrix
            gc.collect()
            torch.cuda.empty_cache()

            if progress:
                freeBytes, totalBytes = torch.cuda.mem_get_info(device)
                print(
                    f"[ElasticBit] {matrixIndex}/{len(compressible)} packed/replaced "
                    f"free={freeBytes / (1024**3):.2f}/{totalBytes / (1024**3):.2f} GiB",
                    flush=True,
                )

        model.elasticbit = _ModelElasticBit(
            model,
            threshold=float(threshold),
            decodePolicy=policy,
            compressedNames=compressedNames,
            skippedNames=skippedNames,
        )
        return model

    @staticmethod
    def save(model: nn.Module, path: str | Path) -> Path:
        controller = getattr(model, "elasticbit", None)
        if not isinstance(controller, _ModelElasticBit):
            raise ValueError("ElasticBit.save() expects a model produced by ElasticBit.compress()")

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        modules = dict(model.named_modules())

        with tempfile.TemporaryDirectory(prefix="elasticbit-save-") as tempDir:
            root = Path(tempDir)
            matricesDir = root / "matrices"
            matricesDir.mkdir(parents=True, exist_ok=True)
            layerEntries = []
            savedMatrices: dict[int, str] = {}

            for index, name in enumerate(controller.compressedNames):
                layer = modules.get(name)
                if not isinstance(layer, _ElasticLinear):
                    raise RuntimeError(f"ElasticBit layer disappeared before save: {name}")
                matrixId = id(layer.matrix)
                matrixFile = savedMatrices.get(matrixId)
                if matrixFile is None:
                    matrixFile = f"matrix_{len(savedMatrices):05d}.mlb"
                    layer.matrix.save(matricesDir / matrixFile)
                    savedMatrices[matrixId] = matrixFile
                layerEntries.append({
                    "name": name,
                    "matrix": matrixFile,
                    "inFeatures": layer.inFeatures,
                    "outFeatures": layer.outFeatures,
                    "hasBias": layer.bias is not None,
                })

            manifest = {
                "format": _MODEL_FORMAT,
                "version": _MODEL_FORMAT_VERSION,
                "threshold": controller.threshold,
                "decodePolicy": controller.decodePolicy,
                "compressedNames": list(controller.compressedNames),
                "skippedNames": list(controller.skippedNames),
                "layers": layerEntries,
            }
            (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            torch.save(model.state_dict(), root / "state.pt")

            with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for file in root.rglob("*"):
                    if file.is_file():
                        archive.write(file, file.relative_to(root).as_posix())

        return destination

    @staticmethod
    def load(
        model: nn.Module,
        path: str | Path,
        decodePolicy: str = "hardwareNative",
    ) -> nn.Module:
        if not isinstance(model, nn.Module):
            raise TypeError("ElasticBit.load() requires the model architecture as its first argument")
        _requireNative()
        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(source)

        with tempfile.TemporaryDirectory(prefix="elasticbit-load-") as tempDir:
            root = Path(tempDir)
            with zipfile.ZipFile(source, "r") as archive:
                archive.extractall(root)
            manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
            if manifest.get("format") != _MODEL_FORMAT or manifest.get("version") != _MODEL_FORMAT_VERSION:
                raise ValueError("Unsupported ElasticBit model artifact")

            policy = _checkPolicy(decodePolicy)
            modules = dict(model.named_modules())
            matrixCache: dict[str, RuntimeMatrix] = {}
            compressedNames: list[str] = []

            for entry in manifest["layers"]:
                name = str(entry["name"])
                target = modules.get(name)
                if not isinstance(target, nn.Linear):
                    raise ValueError(f"Model architecture does not contain expected Linear layer: {name}")
                device = target.weight.device
                if device.type != "cuda":
                    raise RuntimeError("ElasticBit.load() requires the target model on CUDA")
                matrixName = str(entry["matrix"])
                matrix = matrixCache.get(matrixName)
                if matrix is None:
                    with torch.cuda.device(device):
                        matrix = RuntimeMatrix.load(root / "matrices" / matrixName, decodePolicy=policy)
                    matrixCache[matrixName] = matrix
                replacement = _ElasticLinear(
                    matrix=matrix,
                    inFeatures=int(entry["inFeatures"]),
                    outFeatures=int(entry["outFeatures"]),
                    bias=target.bias if bool(entry["hasBias"]) else None,
                ).to(device=device)
                _replaceNamedModule(model, name, replacement)
                compressedNames.append(name)
                modules[name] = replacement

            state = torch.load(root / "state.pt", map_location=_moduleDevice(model), weights_only=True)
            model.load_state_dict(state, strict=True)
            model.elasticbit = _ModelElasticBit(
                model,
                threshold=float(manifest["threshold"]),
                decodePolicy=policy,
                compressedNames=compressedNames,
                skippedNames=tuple(manifest.get("skippedNames", ())),
            )
            return model


__all__ = [
    "ElasticBit",
    "RuntimeMatrix",
    "BitAnalysis",
    "BitCandidate",
    "BackendInfo",
]
