# Copyright 2026 Zameer Hussain and Akhtar Hussain
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# See LICENSE and LICENSING_NOTICE.md; commercial use requires a separate written license.

"""Exact Stage-2F T4 latency-aware planner from the Granite research notebook.

This module intentionally does *not* generalize the validated notebook algorithm.
It is a direct library port of notebook09b41bd228(6).ipynb cells 37-41:

- exact candidate set;
- exact M values;
- exact iteration/round counts;
- exact numerical guards;
- exact hierarchical full-layer selection margin;
- exact final no-regression guard.

The first library version is deliberately limited to one Granite 4.2 8B decoder
layer on a validated Tesla T4.  That restriction is intentional: extending the
algorithm to other models, shapes, sequence buckets, or GPUs requires a separate
validated notebook rather than an unverified generalized implementation.
"""
from __future__ import annotations

from dataclasses import dataclass
import gc
import itertools
import json
import math
import random
import statistics
from typing import Any, Mapping, Sequence

import torch
from torch import nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Frozen notebook constants.  Keep these values in parity with the reference
# notebook.  tests/test_elasticbit_stage2f_parity.py locks this surface.
# ---------------------------------------------------------------------------

STAGE2F_PLANNER_NAME = "Stage-2F T4 Latency-Aware Planner v1"
STAGE2F_REFERENCE_NOTEBOOK = "notebook09b41bd228(6).ipynb"

S2F_M_VALUES = (1, 16, 128, 512)
S2F_ITERS = {1: 240, 16: 120, 128: 45, 512: 14}
S2F_ROUNDS = 5
S2F_MAX_GROUP_REL_L2 = 2e-2

S2F_TOP_K = 3
S2F_FULL_LAYER_MARGIN = 0.02
S2F_FULL_ITERS = {1: 60, 16: 40, 128: 14, 512: 4}
S2F_FULL_ROUNDS = 3

S2F_FINAL_ITERS = {1: 100, 16: 70, 128: 20, 512: 6}
S2F_FINAL_ROUNDS = 5
S2F_MAX_FULL_REL_L2 = 2e-2
S2F_FINAL_DEPLOY_MARGIN = 0.01

S2F_QKV_CANDIDATES = (
    "eb_separate",
    "dense_separate",
    "dense_packed",
    "dense_packed_contiguous",
    "q_plus_kv",
    "qk_plus_v",
    "qv_plus_k",
)
S2F_QKV_M1_CANDIDATES = (
    "dense_separate_mv",
    "dense_packed_mv",
    "q_plus_kv_mv",
)
S2F_GATEUP_CANDIDATES = (
    "eb_separate",
    "dense_separate",
    "dense_packed",
    "dense_packed_contiguous",
)
S2F_GATEUP_M1_CANDIDATES = (
    "dense_separate_mv",
    "dense_packed_mv",
)

# The reference notebook is Granite 4.2 8B.  Do not silently widen this scope.
_REFERENCE_HIDDEN_SIZE = 4096
_REFERENCE_INTERMEDIATE_SIZE = 12800
_REFERENCE_NUM_HEADS = 32
_REFERENCE_NUM_KV_HEADS = 8

_SUFFIXES = {
    "q": "self_attn.q_proj",
    "k": "self_attn.k_proj",
    "v": "self_attn.v_proj",
    "o": "self_attn.o_proj",
    "gate": "mlp.gate_proj",
    "up": "mlp.up_proj",
    "down": "mlp.down_proj",
}


def _clearCuda() -> None:
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect()


def _graniteRelL2(a: torch.Tensor, b: torch.Tensor) -> float:
    """Stage-2E-local relative L2 metric copied from the notebook."""
    a32 = a.float()
    b32 = b.float()
    return float(
        torch.linalg.vector_norm(a32 - b32)
        / (torch.linalg.vector_norm(a32) + 1e-12)
    )


def _moduleBySuffix(root: nn.Module, suffix: str) -> tuple[str, nn.Module]:
    matches = [
        (name, module)
        for name, module in root.named_modules()
        if name.endswith(suffix)
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Stage-2F expected exactly one module ending with {suffix!r}; "
            f"found {[x[0] for x in matches]}"
        )
    return matches[0]


def _materializeMatrix(mod: nn.Module) -> torch.Tensor:
    matrix = mod.matrix
    if hasattr(matrix, "_materializeTorch"):
        weight = matrix._materializeTorch()
        # Exact notebook behavior: clone the execution weight so benchmark
        # references are independent of RuntimeMatrix-owned memory.
        return (
            weight.detach()
            .to(device="cuda", dtype=torch.float16)
            .contiguous()
            .clone()
        )

    # Compatibility fallback copied from the reference notebook.
    import numpy as np

    array = matrix.dequantize()
    return torch.from_numpy(np.asarray(array)).to(
        device="cuda",
        dtype=torch.float16,
    ).contiguous()


def _baselineCall(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    # A forced re-tune may encounter proxies installed by a previous optimize()
    # call.  Use their original ElasticBit path rather than the cached Stage-2F
    # route so the notebook baseline remains unchanged.
    baseline = getattr(module, "_stage2fBaselineForward", None)
    if callable(baseline):
        return baseline(x)
    return module(x)


def _split2(
    z: torch.Tensor,
    a: int,
    b: int,
    make_contiguous: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    x, y = torch.split(z, [a, b], dim=-1)
    if make_contiguous:
        x = x.contiguous()
        y = y.contiguous()
    return x, y


@torch.inference_mode()
def _benchOnce(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()

    return float(start.elapsed_time(end) / iters)


def _interleavedBench(
    candidates: Mapping[str, Any],
    warmup: int,
    iters: int,
    rounds: int = 5,
    seed: int = 1234,
) -> list[dict[str, Any]]:
    """Exact Stage-2F interleaved benchmark loop from the notebook."""
    samples: dict[str, list[float]] = {name: [] for name in candidates}
    names = list(candidates)

    for name in names:
        for _ in range(max(2, warmup // 2)):
            candidates[name]()
    torch.cuda.synchronize()

    for roundIndex in range(rounds):
        order = list(names)
        random.Random(seed + roundIndex).shuffle(order)
        for name in order:
            milliseconds = _benchOnce(
                candidates[name],
                warmup=max(2, warmup // 4),
                iters=iters,
            )
            samples[name].append(milliseconds)

    rows: list[dict[str, Any]] = []
    for name, values in samples.items():
        valuesSorted = sorted(values)
        rows.append({
            "candidate": name,
            "median_ms": float(statistics.median(valuesSorted)),
            "min_ms": float(min(valuesSorted)),
            "max_ms": float(max(valuesSorted)),
            "samples_ms": [float(value) for value in valuesSorted],
        })
    return sorted(rows, key=lambda row: row["median_ms"])


def _qkvError(
    reference: Sequence[torch.Tensor],
    candidate: Sequence[torch.Tensor],
) -> float:
    return max(
        _graniteRelL2(a, b)
        for a, b in zip(reference, candidate)
    )


def _projectionCalls(qkvStrategy: str, gateupStrategy: str) -> int:
    qkvCalls = {
        "eb_separate": 3,
        "dense_separate": 3,
        "dense_separate_mv": 3,
        "dense_packed": 1,
        "dense_packed_contiguous": 1,
        "dense_packed_mv": 1,
        "q_plus_kv": 2,
        "q_plus_kv_mv": 2,
        "qk_plus_v": 2,
        "qv_plus_k": 2,
    }.get(qkvStrategy, 3)

    gateCalls = {
        "eb_separate": 2,
        "dense_separate": 2,
        "dense_separate_mv": 2,
        "dense_packed": 1,
        "dense_packed_contiguous": 1,
        "dense_packed_mv": 1,
    }.get(gateupStrategy, 2)

    return qkvCalls + gateCalls + 2  # O + Down


def _findReferenceGraniteBlock(model: nn.Module) -> tuple[str, nn.Module, Any]:
    matches = [
        (name, module)
        for name, module in model.named_modules()
        if module.__class__.__name__ == "GraniteDecoderLayer"
    ]
    if len(matches) != 1:
        raise NotImplementedError(
            "ElasticBit.optimize() Stage-2F v1 intentionally matches the validated "
            "one-layer Granite notebook and currently requires exactly one "
            f"GraniteDecoderLayer; found {len(matches)}.  Whole-model/generalized "
            "planning is intentionally not substituted for the notebook algorithm."
        )

    name, block = matches[0]
    config = getattr(getattr(block, "self_attn", None), "config", None)
    if config is None:
        raise RuntimeError("Stage-2F could not read GraniteAttention.config")

    actual = (
        int(getattr(config, "hidden_size")),
        int(getattr(config, "intermediate_size")),
        int(getattr(config, "num_attention_heads")),
        int(getattr(config, "num_key_value_heads")),
    )
    expected = (
        _REFERENCE_HIDDEN_SIZE,
        _REFERENCE_INTERMEDIATE_SIZE,
        _REFERENCE_NUM_HEADS,
        _REFERENCE_NUM_KV_HEADS,
    )
    if actual != expected:
        raise NotImplementedError(
            "ElasticBit.optimize() Stage-2F v1 is frozen to the validated "
            f"Granite 4.2 8B notebook shape {expected}; received {actual}."
        )

    return name, block, config


def _projectionModules(block: nn.Module) -> dict[str, nn.Module]:
    mods = {
        "q": block.self_attn.q_proj,
        "k": block.self_attn.k_proj,
        "v": block.self_attn.v_proj,
        "o": block.self_attn.o_proj,
        "gate": block.mlp.gate_proj,
        "up": block.mlp.up_proj,
        "down": block.mlp.down_proj,
    }
    for key, module in mods.items():
        if not hasattr(module, "matrix"):
            raise RuntimeError(
                f"Stage-2F requires the real ElasticBit matrix-backed {key} projection"
            )

    # The notebook reference has bias=False for Q/K/V/Gate/Up.  Do not silently
    # generalize the exact candidate math to biased projections.
    for key in ("q", "k", "v", "gate", "up"):
        if getattr(mods[key], "bias", None) is not None:
            raise NotImplementedError(
                f"Stage-2F notebook parity requires bias-free {key} projection"
            )
    return mods


def _topologySignature(mods: Mapping[str, nn.Module]) -> dict[str, Any]:
    signature: dict[str, Any] = {}
    for key in ("q", "k", "v", "o", "gate", "up", "down"):
        matrix = mods[key].matrix
        signature[key] = {
            "shape": [int(matrix.rows), int(matrix.cols)],
            "storageBits": int(matrix.storageBits),
            "threshold": float(matrix.threshold),
            "error": float(matrix.error),
        }
    return signature


def _hardwareRecord(backend: Any) -> dict[str, Any]:
    return {
        "device": str(backend.device),
        "architecture": str(backend.architecture),
        "torch": str(torch.__version__),
        "cuda": None if torch.version.cuda is None else str(torch.version.cuda),
        "planner": STAGE2F_PLANNER_NAME,
    }


def _hardwareKey(record: Mapping[str, Any]) -> str:
    return json.dumps(dict(record), sort_keys=True, separators=(",", ":"))


@dataclass
class _CandidateBundle:
    mods: dict[str, nn.Module]
    weights: dict[str, torch.Tensor]
    W_QKV: torch.Tensor
    W_GATEUP: torch.Tensor
    W_KV: torch.Tensor
    W_QK: torch.Tensor
    W_QV: torch.Tensor
    qSize: int
    kSize: int
    vSize: int
    gSize: int
    uSize: int


def _buildBundle(mods: dict[str, nn.Module]) -> _CandidateBundle:
    # Exact notebook behavior: materialize all seven real execution weights,
    # even though Stage-2F candidate selection only packs Q/K/V/Gate/Up.
    weights = {key: _materializeMatrix(mods[key]) for key in _SUFFIXES}

    W_QKV = torch.cat(
        [weights["q"], weights["k"], weights["v"]],
        dim=0,
    ).contiguous()
    W_GATEUP = torch.cat(
        [weights["gate"], weights["up"]],
        dim=0,
    ).contiguous()
    W_KV = torch.cat([weights["k"], weights["v"]], dim=0).contiguous()
    W_QK = torch.cat([weights["q"], weights["k"]], dim=0).contiguous()
    W_QV = torch.cat([weights["q"], weights["v"]], dim=0).contiguous()

    separateBytes = sum(
        weights[key].numel() * weights[key].element_size()
        for key in ("q", "k", "v", "gate", "up")
    )
    fusedBytes = W_QKV.numel() * 2 + W_GATEUP.numel() * 2
    if fusedBytes != separateBytes:
        raise AssertionError("Stage-2F concatenation changed coefficient count")

    return _CandidateBundle(
        mods=mods,
        weights=weights,
        W_QKV=W_QKV,
        W_GATEUP=W_GATEUP,
        W_KV=W_KV,
        W_QK=W_QK,
        W_QV=W_QV,
        qSize=int(weights["q"].shape[0]),
        kSize=int(weights["k"].shape[0]),
        vSize=int(weights["v"].shape[0]),
        gSize=int(weights["gate"].shape[0]),
        uSize=int(weights["up"].shape[0]),
    )


def _candidateFunctions(bundle: _CandidateBundle):
    mods = bundle.mods
    weights = bundle.weights

    @torch.inference_mode()
    def qkvEbSeparate(x):
        return (
            _baselineCall(mods["q"], x),
            _baselineCall(mods["k"], x),
            _baselineCall(mods["v"], x),
        )

    @torch.inference_mode()
    def qkvDenseSeparate(x):
        return (
            F.linear(x, weights["q"]),
            F.linear(x, weights["k"]),
            F.linear(x, weights["v"]),
        )

    @torch.inference_mode()
    def qkvDensePacked(x):
        z = F.linear(x, bundle.W_QKV)
        return torch.split(
            z,
            [bundle.qSize, bundle.kSize, bundle.vSize],
            dim=-1,
        )

    @torch.inference_mode()
    def qkvDensePackedContiguous(x):
        q, k, v = torch.split(
            F.linear(x, bundle.W_QKV),
            [bundle.qSize, bundle.kSize, bundle.vSize],
            dim=-1,
        )
        return q.contiguous(), k.contiguous(), v.contiguous()

    @torch.inference_mode()
    def qkvQPlusKv(x):
        q = F.linear(x, weights["q"])
        k, v = _split2(
            F.linear(x, bundle.W_KV),
            bundle.kSize,
            bundle.vSize,
        )
        return q, k, v

    @torch.inference_mode()
    def qkvQkPlusV(x):
        q, k = _split2(
            F.linear(x, bundle.W_QK),
            bundle.qSize,
            bundle.kSize,
        )
        v = F.linear(x, weights["v"])
        return q, k, v

    @torch.inference_mode()
    def qkvQvPlusK(x):
        q, v = _split2(
            F.linear(x, bundle.W_QV),
            bundle.qSize,
            bundle.vSize,
        )
        k = F.linear(x, weights["k"])
        return q, k, v

    @torch.inference_mode()
    def qkvDenseSeparateMv(x):
        assert x.shape[0] == 1
        v = x[0]
        return (
            torch.mv(weights["q"], v).unsqueeze(0),
            torch.mv(weights["k"], v).unsqueeze(0),
            torch.mv(weights["v"], v).unsqueeze(0),
        )

    @torch.inference_mode()
    def qkvDensePackedMv(x):
        assert x.shape[0] == 1
        z = torch.mv(bundle.W_QKV, x[0]).unsqueeze(0)
        return torch.split(
            z,
            [bundle.qSize, bundle.kSize, bundle.vSize],
            dim=-1,
        )

    @torch.inference_mode()
    def qkvQPlusKvMv(x):
        assert x.shape[0] == 1
        v = x[0]
        q = torch.mv(weights["q"], v).unsqueeze(0)
        kv = torch.mv(bundle.W_KV, v).unsqueeze(0)
        k, vv = torch.split(
            kv,
            [bundle.kSize, bundle.vSize],
            dim=-1,
        )
        return q, k, vv

    @torch.inference_mode()
    def gateupEbSeparate(x):
        gate = _baselineCall(mods["gate"], x)
        up = _baselineCall(mods["up"], x)
        return F.silu(gate) * up

    @torch.inference_mode()
    def gateupDenseSeparate(x):
        gate = F.linear(x, weights["gate"])
        up = F.linear(x, weights["up"])
        return F.silu(gate) * up

    @torch.inference_mode()
    def gateupDensePacked(x):
        gate, up = torch.split(
            F.linear(x, bundle.W_GATEUP),
            [bundle.gSize, bundle.uSize],
            dim=-1,
        )
        return F.silu(gate) * up

    @torch.inference_mode()
    def gateupDensePackedContiguous(x):
        gate, up = torch.split(
            F.linear(x, bundle.W_GATEUP),
            [bundle.gSize, bundle.uSize],
            dim=-1,
        )
        gate = gate.contiguous()
        up = up.contiguous()
        return F.silu(gate) * up

    @torch.inference_mode()
    def gateupDenseSeparateMv(x):
        assert x.shape[0] == 1
        v = x[0]
        gate = torch.mv(weights["gate"], v).unsqueeze(0)
        up = torch.mv(weights["up"], v).unsqueeze(0)
        return F.silu(gate) * up

    @torch.inference_mode()
    def gateupDensePackedMv(x):
        assert x.shape[0] == 1
        z = torch.mv(bundle.W_GATEUP, x[0]).unsqueeze(0)
        gate, up = torch.split(z, [bundle.gSize, bundle.uSize], dim=-1)
        return F.silu(gate) * up

    qkv = {
        "eb_separate": qkvEbSeparate,
        "dense_separate": qkvDenseSeparate,
        "dense_packed": qkvDensePacked,
        "dense_packed_contiguous": qkvDensePackedContiguous,
        "q_plus_kv": qkvQPlusKv,
        "qk_plus_v": qkvQkPlusV,
        "qv_plus_k": qkvQvPlusK,
    }
    gateup = {
        "eb_separate": gateupEbSeparate,
        "dense_separate": gateupDenseSeparate,
        "dense_packed": gateupDensePacked,
        "dense_packed_contiguous": gateupDensePackedContiguous,
    }
    qkvM1 = {
        "dense_separate_mv": qkvDenseSeparateMv,
        "dense_packed_mv": qkvDensePackedMv,
        "q_plus_kv_mv": qkvQPlusKvMv,
    }
    gateupM1 = {
        "dense_separate_mv": gateupDenseSeparateMv,
        "dense_packed_mv": gateupDensePackedMv,
    }
    return qkv, gateup, qkvM1, gateupM1


def _graniteInputs(config: Any, seqLen: int, device: torch.device):
    try:
        from transformers.models.granite.modeling_granite import GraniteRotaryEmbedding
    except Exception as exc:  # pragma: no cover - depends on optional research dependency
        raise RuntimeError(
            "ElasticBit.optimize() Stage-2F notebook parity requires "
            "transformers with GraniteRotaryEmbedding available"
        ) from exc

    hiddenSize = int(config.hidden_size)
    x = torch.randn(
        1,
        seqLen,
        hiddenSize,
        device=device,
        dtype=torch.float16,
    )

    positionIds = torch.arange(
        seqLen,
        device=device,
        dtype=torch.long,
    ).unsqueeze(0)

    rope = GraniteRotaryEmbedding(config=config).to(device).eval()
    with torch.no_grad():
        cos, sin = rope(x, positionIds)

    neg = torch.finfo(torch.float16).min
    causal = torch.zeros(
        1,
        1,
        seqLen,
        seqLen,
        device=device,
        dtype=torch.float16,
    )
    upper = torch.triu(
        torch.ones(
            seqLen,
            seqLen,
            device=device,
            dtype=torch.bool,
        ),
        diagonal=1,
    )
    causal = causal.masked_fill(upper[None, None, :, :], neg)
    return x, causal, cos, sin


def _runNotebookPlanner(
    block: nn.Module,
    config: Any,
    bundle: _CandidateBundle,
) -> dict[str, Any]:
    qkvFns, gateupFns, qkvM1Fns, gateupM1Fns = _candidateFunctions(bundle)
    hiddenSize = int(config.hidden_size)
    device = next(block.parameters()).device

    # ------------------------------------------------------------------
    # Notebook cell 38 — isolated T4 autotune.
    # ------------------------------------------------------------------
    groupRows: list[dict[str, Any]] = []
    groupRankings: dict[tuple[int, str], list[dict[str, Any]]] = {}

    torch.manual_seed(20260921)

    for M in S2F_M_VALUES:
        x = torch.randn(
            M,
            hiddenSize,
            device=device,
            dtype=torch.float16,
        )

        qFns = dict(qkvFns)
        gFns = dict(gateupFns)
        if M == 1:
            qFns.update(qkvM1Fns)
            gFns.update(gateupM1Fns)

        qkvRef = qkvFns["dense_separate"](x)
        gateupRef = gateupFns["dense_separate"](x)

        validQkv: dict[str, Any] = {}
        for name, fn in qFns.items():
            try:
                out = fn(x)
                error = _qkvError(qkvRef, out)
                if math.isfinite(error) and error <= S2F_MAX_GROUP_REL_L2:
                    validQkv[name] = (lambda fn=fn, x=x: fn(x))
                else:
                    print(f"Reject QKV M={M} {name}: rel_l2={error:.6g}")
            except Exception as exc:
                print(f"Reject QKV M={M} {name}: {type(exc).__name__}: {exc}")

        validGateup: dict[str, Any] = {}
        for name, fn in gFns.items():
            try:
                out = fn(x)
                error = _graniteRelL2(gateupRef, out)
                if math.isfinite(error) and error <= S2F_MAX_GROUP_REL_L2:
                    validGateup[name] = (lambda fn=fn, x=x: fn(x))
                else:
                    print(f"Reject GateUp M={M} {name}: rel_l2={error:.6g}")
            except Exception as exc:
                print(f"Reject GateUp M={M} {name}: {type(exc).__name__}: {exc}")

        qRows = _interleavedBench(
            validQkv,
            warmup=12,
            iters=S2F_ITERS[M],
            rounds=S2F_ROUNDS,
            seed=1000 + M,
        )
        gRows = _interleavedBench(
            validGateup,
            warmup=12,
            iters=S2F_ITERS[M],
            rounds=S2F_ROUNDS,
            seed=2000 + M,
        )

        groupRankings[(M, "QKV")] = qRows
        groupRankings[(M, "GateUp")] = gRows

        for rank, row in enumerate(qRows, start=1):
            groupRows.append({"M": M, "group": "QKV", "rank": rank, **row})
        for rank, row in enumerate(gRows, start=1):
            groupRows.append({"M": M, "group": "GateUp", "rank": rank, **row})

        print()
        print("=" * 92)
        print(f"M={M} isolated ranking")
        print("QKV   :", [(row["candidate"], round(row["median_ms"], 5)) for row in qRows[:5]])
        print("GateUp:", [(row["candidate"], round(row["median_ms"], 5)) for row in gRows[:5]])

        del x, qkvRef, gateupRef
        _clearCuda()

    # ------------------------------------------------------------------
    # Notebook cell 39 — actual Granite layer math harness.
    # ------------------------------------------------------------------
    def applyEb(name: str, x: torch.Tensor) -> torch.Tensor:
        originalShape = x.shape[:-1]
        flat = x.reshape(-1, x.shape[-1]).contiguous()
        y = _baselineCall(bundle.mods[name], flat)
        return y.reshape(*originalShape, y.shape[-1])

    def applyQkv(strategy: str, h: torch.Tensor):
        flat = h.reshape(-1, h.shape[-1]).contiguous()
        fn = qkvFns.get(strategy, qkvM1Fns.get(strategy))
        if fn is None:
            raise KeyError(f"Unknown QKV strategy: {strategy}")
        q, k, v = fn(flat)
        prefix = h.shape[:-1]
        return (
            q.reshape(*prefix, q.shape[-1]),
            k.reshape(*prefix, k.shape[-1]),
            v.reshape(*prefix, v.shape[-1]),
        )

    def applyGateup(strategy: str, h: torch.Tensor) -> torch.Tensor:
        flat = h.reshape(-1, h.shape[-1]).contiguous()
        fn = gateupFns.get(strategy, gateupM1Fns.get(strategy))
        if fn is None:
            raise KeyError(f"Unknown GateUp strategy: {strategy}")
        y = fn(flat)
        return y.reshape(*h.shape[:-1], y.shape[-1])

    @torch.inference_mode()
    def graniteForward(
        x: torch.Tensor,
        causalMask: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        qkvStrategy: str = "eb_separate",
        gateupStrategy: str = "eb_separate",
    ) -> torch.Tensor:
        residual = x
        h = block.input_layernorm(x)
        batch, seq = h.shape[:2]

        q, k, v = applyQkv(qkvStrategy, h)

        q = q.view(
            batch,
            seq,
            int(config.num_attention_heads),
            hiddenSize // int(config.num_attention_heads),
        ).transpose(1, 2)
        k = k.view(
            batch,
            seq,
            int(config.num_key_value_heads),
            hiddenSize // int(config.num_attention_heads),
        ).transpose(1, 2)
        v = v.view(
            batch,
            seq,
            int(config.num_key_value_heads),
            hiddenSize // int(config.num_attention_heads),
        ).transpose(1, 2)

        cosU = cos.unsqueeze(1)
        sinU = sin.unsqueeze(1)
        qHalf = torch.cat(
            (-q[..., q.shape[-1] // 2:], q[..., : q.shape[-1] // 2]),
            dim=-1,
        )
        kHalf = torch.cat(
            (-k[..., k.shape[-1] // 2:], k[..., : k.shape[-1] // 2]),
            dim=-1,
        )
        q = q * cosU + qHalf * sinU
        k = k * cosU + kHalf * sinU

        numHeads = int(config.num_attention_heads)
        numKvHeads = int(config.num_key_value_heads)
        numKvGroups = numHeads // numKvHeads
        kRep = k.repeat_interleave(numKvGroups, dim=1)
        vRep = v.repeat_interleave(numKvGroups, dim=1)

        scaling = float(config.attention_multiplier)
        attnScores = torch.matmul(q, kRep.transpose(-2, -1)) * scaling
        attnScores = attnScores + causalMask
        attnProbs = F.softmax(attnScores.float(), dim=-1).to(torch.float16)

        attn = torch.matmul(attnProbs, vRep)
        attn = attn.transpose(1, 2).contiguous().view(batch, seq, hiddenSize)
        attn = applyEb("o", attn)

        residualMultiplier = float(config.residual_multiplier)
        h = residual + attn * residualMultiplier

        residual = h
        h = block.post_attention_layernorm(h)
        h = applyGateup(gateupStrategy, h)
        h = applyEb("down", h)
        return residual + h * residualMultiplier

    # ------------------------------------------------------------------
    # Notebook cell 40 — hierarchical full-layer autotune.
    # ------------------------------------------------------------------
    fullSearchRows: list[dict[str, Any]] = []
    selectedPlans: dict[int, dict[str, Any]] = {}

    for sequenceLength in S2F_M_VALUES:
        args = _graniteInputs(config, sequenceLength, device)
        qRank = groupRankings[(sequenceLength, "QKV")]
        gRank = groupRankings[(sequenceLength, "GateUp")]

        qNames = [row["candidate"] for row in qRank[:S2F_TOP_K]]
        gNames = [row["candidate"] for row in gRank[:S2F_TOP_K]]
        if "eb_separate" not in qNames:
            qNames.append("eb_separate")
        if "eb_separate" not in gNames:
            gNames.append("eb_separate")

        pairCandidates: dict[str, Any] = {}
        for qName, gName in itertools.product(qNames, gNames):
            key = f"{qName} | {gName}"
            pairCandidates[key] = (
                lambda qName=qName, gName=gName, args=args:
                    graniteForward(
                        *args,
                        qkvStrategy=qName,
                        gateupStrategy=gName,
                    )
            )

        rows = _interleavedBench(
            pairCandidates,
            warmup=6,
            iters=S2F_FULL_ITERS[sequenceLength],
            rounds=S2F_FULL_ROUNDS,
            seed=3000 + sequenceLength,
        )

        baselineKey = "eb_separate | eb_separate"
        baselineRow = next(row for row in rows if row["candidate"] == baselineKey)
        bestRow = rows[0]
        measuredGain = baselineRow["median_ms"] / bestRow["median_ms"] - 1.0

        if (
            bestRow["candidate"] != baselineKey
            and measuredGain >= S2F_FULL_LAYER_MARGIN
        ):
            selected = bestRow["candidate"]
            reason = "accepted_end_to_end_speedup"
        else:
            selected = baselineKey
            reason = "rollback_no_safe_end_to_end_gain"

        qSelected, gSelected = [value.strip() for value in selected.split("|")]
        selectedPlans[sequenceLength] = {
            "qkv": qSelected,
            "gateup": gSelected,
            "search_best": bestRow["candidate"],
            "search_best_ms": float(bestRow["median_ms"]),
            "baseline_ms": float(baselineRow["median_ms"]),
            "measured_gain": float(measuredGain),
            "decision": reason,
            "projection_calls": int(_projectionCalls(qSelected, gSelected)),
        }

        for rank, row in enumerate(rows, start=1):
            fullSearchRows.append({
                "sequence_length": sequenceLength,
                "rank": rank,
                **row,
            })

        print()
        print("=" * 100)
        print(f"S={sequenceLength}")
        print("Search best :", bestRow["candidate"], f"{bestRow['median_ms']:.5f} ms")
        print("Baseline    :", f"{baselineRow['median_ms']:.5f} ms")
        print("Gain        :", f"{100 * measuredGain:.2f}%")
        print("Final plan  :", selected)
        print("Decision    :", reason)
        print("Proj calls  :", selectedPlans[sequenceLength]["projection_calls"])

        del args
        _clearCuda()

    # ------------------------------------------------------------------
    # Notebook cell 41 — final quality + robust A/B.
    # ------------------------------------------------------------------
    qualityRows: list[dict[str, Any]] = []
    finalBenchRows: list[dict[str, Any]] = []

    for sequenceLength in S2F_M_VALUES:
        args = _graniteInputs(config, sequenceLength, device)
        plan = selectedPlans[sequenceLength]

        with torch.inference_mode():
            yBase = graniteForward(
                *args,
                qkvStrategy="eb_separate",
                gateupStrategy="eb_separate",
            )
            yOptimized = graniteForward(
                *args,
                qkvStrategy=plan["qkv"],
                gateupStrategy=plan["gateup"],
            )

        relativeL2 = _graniteRelL2(yBase, yOptimized)
        maxAbs = float((yBase.float() - yOptimized.float()).abs().max())
        verified = bool(
            math.isfinite(relativeL2)
            and relativeL2 <= S2F_MAX_FULL_REL_L2
        )
        qualityRows.append({
            "sequence_length": sequenceLength,
            "qkv": plan["qkv"],
            "gateup": plan["gateup"],
            "relative_l2": relativeL2,
            "max_abs": maxAbs,
            "verified": verified,
        })

        if not verified:
            plan["qkv"] = "eb_separate"
            plan["gateup"] = "eb_separate"
            plan["decision"] = "rollback_quality_guard"
            plan["projection_calls"] = 7

        finalCandidates = {
            "ElasticBit_baseline": (
                lambda args=args:
                    graniteForward(
                        *args,
                        qkvStrategy="eb_separate",
                        gateupStrategy="eb_separate",
                    )
            ),
            "Stage2F_adaptive": (
                lambda args=args, plan=plan:
                    graniteForward(
                        *args,
                        qkvStrategy=plan["qkv"],
                        gateupStrategy=plan["gateup"],
                    )
            ),
        }

        finalRows = _interleavedBench(
            finalCandidates,
            warmup=8,
            iters=S2F_FINAL_ITERS[sequenceLength],
            rounds=S2F_FINAL_ROUNDS,
            seed=4000 + sequenceLength,
        )
        byName = {row["candidate"]: row for row in finalRows}
        baselineMs = byName["ElasticBit_baseline"]["median_ms"]
        selectedMs = byName["Stage2F_adaptive"]["median_ms"]
        speedup = baselineMs / selectedMs

        isBaselinePlan = (
            plan["qkv"] == "eb_separate"
            and plan["gateup"] == "eb_separate"
        )
        if isBaselinePlan:
            finalDecision = "BASELINE_SELECTED"
            deployedMs = baselineMs
            deployedSpeedup = 1.0
            deployedCalls = 7
        elif speedup < 1.01:
            finalDecision = "FINAL_ROLLBACK_TO_BASELINE"
            deployedMs = baselineMs
            deployedSpeedup = 1.0
            deployedCalls = 7
        else:
            finalDecision = "USE_SELECTED_PLAN"
            deployedMs = selectedMs
            deployedSpeedup = speedup
            deployedCalls = int(plan["projection_calls"])

        finalBenchRows.append({
            "sequence_length": sequenceLength,
            "qkv": plan["qkv"],
            "gateup": plan["gateup"],
            "baseline_ms": baselineMs,
            "selected_ms": selectedMs,
            "raw_speedup_x": speedup,
            "deployment_decision": finalDecision,
            "deployed_ms": deployedMs,
            "deployed_speedup_x": deployedSpeedup,
            "deployed_projection_calls": deployedCalls,
        })

        print(
            f"S={sequenceLength:4d} | baseline={baselineMs:.5f} ms | "
            f"selected={selectedMs:.5f} ms | raw={speedup:.3f}x | "
            f"deploy={deployedSpeedup:.3f}x | calls={deployedCalls} | "
            f"{finalDecision}"
        )

        del args, yBase, yOptimized
        _clearCuda()

    if not all(row["verified"] for row in qualityRows):
        raise AssertionError("A selected Stage-2F path failed the full-layer quality guard.")
    if not all(row["deployed_speedup_x"] >= 0.999 for row in finalBenchRows):
        raise AssertionError("No-regression deployment guard failed.")

    deployedPlans: dict[str, dict[str, str]] = {}
    for row in finalBenchRows:
        sequenceLength = int(row["sequence_length"])
        if row["deployment_decision"] == "USE_SELECTED_PLAN":
            deployedPlans[str(sequenceLength)] = {
                "qkv": str(row["qkv"]),
                "gateup": str(row["gateup"]),
            }
        else:
            deployedPlans[str(sequenceLength)] = {
                "qkv": "eb_separate",
                "gateup": "eb_separate",
            }

    return {
        "planner": STAGE2F_PLANNER_NAME,
        "referenceNotebook": STAGE2F_REFERENCE_NOTEBOOK,
        "MValues": list(S2F_M_VALUES),
        "selectedPlans": {str(key): value for key, value in selectedPlans.items()},
        "deployedPlans": deployedPlans,
        "groupAutotune": groupRows,
        "fullLayerSearch": fullSearchRows,
        "quality": qualityRows,
        "finalBenchmarks": finalBenchRows,
    }


class _Stage2FRuntime:
    """Apply the final notebook-selected plan inside the Granite layer math.

    This deliberately mirrors ``granite_s2f_forward`` instead of introducing a
    generalized graph rewriter.  Cached decoding and unsupported call shapes
    delegate to the untouched Hugging Face GraniteDecoderLayer forward.
    """

    def __init__(
        self,
        block: nn.Module,
        config: Any,
        bundle: _CandidateBundle,
        deployedPlans: Mapping[str, Mapping[str, str]],
        originalForward: Any,
    ) -> None:
        self.block = block
        self.config = config
        self.bundle = bundle
        self.originalForward = originalForward
        self.deployedPlans = {
            int(key): {
                "qkv": str(value["qkv"]),
                "gateup": str(value["gateup"]),
            }
            for key, value in deployedPlans.items()
        }
        self.qkvFns, self.gateupFns, self.qkvM1Fns, self.gateupM1Fns = _candidateFunctions(bundle)

    def _fallback(
        self,
        hiddenStates: torch.Tensor,
        attentionMask: torch.Tensor | None,
        positionIds: torch.Tensor | None,
        pastKeyValues: Any,
        useCache: bool | None,
        positionEmbeddings: tuple[torch.Tensor, torch.Tensor] | None,
        kwargs: Mapping[str, Any],
    ) -> torch.Tensor:
        return self.originalForward(
            hidden_states=hiddenStates,
            attention_mask=attentionMask,
            position_ids=positionIds,
            past_key_values=pastKeyValues,
            use_cache=useCache,
            position_embeddings=positionEmbeddings,
            **dict(kwargs),
        )

    def _applyEb(self, name: str, x: torch.Tensor) -> torch.Tensor:
        originalShape = x.shape[:-1]
        flat = x.reshape(-1, x.shape[-1]).contiguous()
        y = _baselineCall(self.bundle.mods[name], flat)
        return y.reshape(*originalShape, y.shape[-1])

    def _applyQkv(self, strategy: str, h: torch.Tensor):
        flat = h.reshape(-1, h.shape[-1]).contiguous()
        fn = self.qkvFns.get(strategy, self.qkvM1Fns.get(strategy))
        if fn is None:
            raise KeyError(f"Unknown QKV strategy: {strategy}")
        q, k, v = fn(flat)
        prefix = h.shape[:-1]
        return (
            q.reshape(*prefix, q.shape[-1]),
            k.reshape(*prefix, k.shape[-1]),
            v.reshape(*prefix, v.shape[-1]),
        )

    def _applyGateup(self, strategy: str, h: torch.Tensor) -> torch.Tensor:
        flat = h.reshape(-1, h.shape[-1]).contiguous()
        fn = self.gateupFns.get(strategy, self.gateupM1Fns.get(strategy))
        if fn is None:
            raise KeyError(f"Unknown GateUp strategy: {strategy}")
        y = fn(flat)
        return y.reshape(*h.shape[:-1], y.shape[-1])

    @torch.inference_mode()
    def forward(
        self,
        hiddenStates: torch.Tensor,
        attentionMask: torch.Tensor | None = None,
        positionIds: torch.Tensor | None = None,
        pastKeyValues: Any = None,
        useCache: bool | None = False,
        positionEmbeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        hiddenSize = int(self.config.hidden_size)
        rows = hiddenStates.numel() // hiddenSize
        plan = self.deployedPlans.get(rows)

        # The reference Stage-2F notebook did not implement KV-cache mutation.
        # Do not invent one here. Cached generation stays on the untouched
        # ElasticBit/Hugging Face path until a separately validated kernel/path
        # exists.
        if (
            plan is None
            or (plan["qkv"] == "eb_separate" and plan["gateup"] == "eb_separate")
            or pastKeyValues is not None
            or bool(useCache)
            or positionEmbeddings is None
            or attentionMask is None
        ):
            return self._fallback(
                hiddenStates,
                attentionMask,
                positionIds,
                pastKeyValues,
                useCache,
                positionEmbeddings,
                kwargs,
            )

        # Exact notebook cell 39 math, with cos/sin and causal mask supplied by
        # the real Granite model rather than synthesized again at runtime.
        cos, sin = positionEmbeddings
        residual = hiddenStates
        h = self.block.input_layernorm(hiddenStates)
        batch, seq = h.shape[:2]

        q, k, v = self._applyQkv(plan["qkv"], h)

        q = q.view(
            batch,
            seq,
            int(self.config.num_attention_heads),
            hiddenSize // int(self.config.num_attention_heads),
        ).transpose(1, 2)
        k = k.view(
            batch,
            seq,
            int(self.config.num_key_value_heads),
            hiddenSize // int(self.config.num_attention_heads),
        ).transpose(1, 2)
        v = v.view(
            batch,
            seq,
            int(self.config.num_key_value_heads),
            hiddenSize // int(self.config.num_attention_heads),
        ).transpose(1, 2)

        cosU = cos.unsqueeze(1)
        sinU = sin.unsqueeze(1)
        qHalf = torch.cat(
            (-q[..., q.shape[-1] // 2:], q[..., : q.shape[-1] // 2]),
            dim=-1,
        )
        kHalf = torch.cat(
            (-k[..., k.shape[-1] // 2:], k[..., : k.shape[-1] // 2]),
            dim=-1,
        )
        q = q * cosU + qHalf * sinU
        k = k * cosU + kHalf * sinU

        numHeads = int(self.config.num_attention_heads)
        numKvHeads = int(self.config.num_key_value_heads)
        numKvGroups = numHeads // numKvHeads
        kRep = k.repeat_interleave(numKvGroups, dim=1)
        vRep = v.repeat_interleave(numKvGroups, dim=1)

        scaling = float(self.config.attention_multiplier)
        attnScores = torch.matmul(q, kRep.transpose(-2, -1)) * scaling
        attnScores = attnScores + attentionMask
        attnProbs = F.softmax(attnScores.float(), dim=-1).to(torch.float16)

        attn = torch.matmul(attnProbs, vRep)
        attn = attn.transpose(1, 2).contiguous().view(batch, seq, hiddenSize)
        attn = self._applyEb("o", attn)

        residualMultiplier = float(self.config.residual_multiplier)
        h = residual + attn * residualMultiplier

        residual = h
        h = self.block.post_attention_layernorm(h)
        h = self._applyGateup(plan["gateup"], h)
        h = self._applyEb("down", h)
        return residual + h * residualMultiplier


def _uninstallRuntime(block: nn.Module, controller: Any) -> None:
    original = getattr(block, "_elasticbitStage2FOriginalForward", None)
    if original is not None:
        block.forward = original
        try:
            delattr(block, "_elasticbitStage2FOriginalForward")
        except AttributeError:
            pass
        try:
            delattr(block, "_elasticbitStage2FRuntime")
        except AttributeError:
            pass
    controller._stage2fRuntime = None
    controller.activeOptimization = None
    _clearCuda()


def _installRuntime(
    block: nn.Module,
    config: Any,
    bundle: _CandidateBundle,
    deployedPlans: Mapping[str, Mapping[str, str]],
    controller: Any,
) -> _Stage2FRuntime:
    import types

    # Preserve the original Hugging Face forward exactly once.  A forced
    # re-tune restores it before installing a fresh Stage-2F runtime.
    originalForward = getattr(block, "_elasticbitStage2FOriginalForward", None)
    if originalForward is None:
        originalForward = block.forward

    runtime = _Stage2FRuntime(
        block=block,
        config=config,
        bundle=bundle,
        deployedPlans=deployedPlans,
        originalForward=originalForward,
    )

    def stage2fForward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: Any = None,
        use_cache: bool | None = False,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        return self._elasticbitStage2FRuntime.forward(
            hiddenStates=hidden_states,
            attentionMask=attention_mask,
            positionIds=position_ids,
            pastKeyValues=past_key_values,
            useCache=use_cache,
            positionEmbeddings=position_embeddings,
            **kwargs,
        )

    block._elasticbitStage2FOriginalForward = originalForward
    block._elasticbitStage2FRuntime = runtime
    block.forward = types.MethodType(stage2fForward, block)
    controller._stage2fRuntime = runtime
    return runtime


def optimizeModelStage2F(
    model: nn.Module,
    controller: Any,
    backend: Any,
    *,
    force: bool = False,
) -> nn.Module:
    """Run or reuse the exact one-layer Stage-2F T4 notebook planner."""
    if not torch.cuda.is_available():
        raise RuntimeError("ElasticBit.optimize() Stage-2F requires CUDA")
    if not backend.available:
        raise RuntimeError(backend.error or "ElasticBit CUDA runtime is unavailable")
    if not (
        bool(backend.validated)
        and str(backend.architecture) == "sm75"
        and "T4" in str(backend.device).upper()
    ):
        raise NotImplementedError(
            "ElasticBit.optimize() Stage-2F v1 is intentionally limited to the "
            "validated Tesla T4 notebook.  A different GPU must be validated by "
            "its own notebook before adding a planner implementation."
        )

    _, block, config = _findReferenceGraniteBlock(model)
    if model.training or block.training:
        raise RuntimeError("ElasticBit.optimize() Stage-2F requires model.eval()")
    if str(getattr(config, "hidden_act", "silu")) != "silu":
        raise NotImplementedError("Stage-2F notebook parity requires Granite SwiGLU/SILU")
    attentionImpl = str(getattr(config, "_attn_implementation", "eager"))
    if attentionImpl != "eager":
        raise NotImplementedError(
            "Stage-2F notebook parity requires Granite eager attention; "
            f"received {attentionImpl!r}"
        )

    # If this exact runtime is already active on the current model, a normal
    # optimize() call is a true no-op: no re-materialization and no benchmark.
    hardware = _hardwareRecord(backend)
    cacheKey = _hardwareKey(hardware)
    active = getattr(controller, "activeOptimization", None)
    if (
        not force
        and isinstance(active, dict)
        and active.get("hardwareKey") == cacheKey
        and getattr(controller, "_stage2fRuntime", None) is not None
    ):
        print("ElasticBit.optimize(): Stage-2F profile already active; no work needed")
        return model

    if force or getattr(controller, "_stage2fRuntime", None) is not None:
        _uninstallRuntime(block, controller)

    mods = _projectionModules(block)
    topology = _topologySignature(mods)

    profiles = getattr(controller, "optimizationProfiles", None)
    if not isinstance(profiles, dict):
        profiles = {}
        controller.optimizationProfiles = profiles

    profile = None if force else profiles.get(cacheKey)
    if profile is not None:
        if profile.get("planner") != STAGE2F_PLANNER_NAME:
            profile = None
        elif profile.get("topology") != topology:
            profile = None

    # Keep a cache hit benchmark-free. Materializing the exact execution
    # tensors is still necessary to instantiate the notebook-selected runtime.
    if profile is not None:
        print("ElasticBit.optimize(): Stage-2F profile cache HIT")
        print("GPU                :", backend.device)
        print("Architecture       :", backend.architecture)
        print("Benchmarking       : skipped")
        bundle = _buildBundle(mods)
        _installRuntime(
            block,
            config,
            bundle,
            profile["deployedPlans"],
            controller,
        )
        controller.activeOptimization = {
            "planner": STAGE2F_PLANNER_NAME,
            "hardwareKey": cacheKey,
            "cache": "hit",
        }
        return model

    print("ElasticBit.optimize(): Stage-2F profile cache MISS")
    print("GPU                :", backend.device)
    print("Architecture       :", backend.architecture)
    print("Reference notebook :", STAGE2F_REFERENCE_NOTEBOOK)
    print("Benchmarking exact Stage-2F candidates...")

    bundle = _buildBundle(mods)
    report = _runNotebookPlanner(block, config, bundle)
    report["hardware"] = hardware
    report["topology"] = topology
    profiles[cacheKey] = report

    _installRuntime(
        block,
        config,
        bundle,
        report["deployedPlans"],
        controller,
    )
    controller.activeOptimization = {
        "planner": STAGE2F_PLANNER_NAME,
        "hardwareKey": cacheKey,
        "cache": "miss",
    }

    print("ElasticBit.optimize(): Stage-2F profile cached on the model controller")
    print("Save the model with ElasticBit.save() to persist this hardware profile.")
    return model


__all__ = [
    "STAGE2F_PLANNER_NAME",
    "STAGE2F_REFERENCE_NOTEBOOK",
    "S2F_M_VALUES",
    "S2F_ITERS",
    "S2F_ROUNDS",
    "S2F_MAX_GROUP_REL_L2",
    "S2F_TOP_K",
    "S2F_FULL_LAYER_MARGIN",
    "S2F_FULL_ITERS",
    "S2F_FULL_ROUNDS",
    "S2F_FINAL_ITERS",
    "S2F_FINAL_ROUNDS",
    "S2F_MAX_FULL_REL_L2",
    "S2F_FINAL_DEPLOY_MARGIN",
    "S2F_QKV_CANDIDATES",
    "S2F_QKV_M1_CANDIDATES",
    "S2F_GATEUP_CANDIDATES",
    "S2F_GATEUP_M1_CANDIDATES",
    "optimizeModelStage2F",
]
