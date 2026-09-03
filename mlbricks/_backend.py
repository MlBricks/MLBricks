from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import torch

_EXTENSIONS: Dict[Tuple[int, int, str, str], object] = {}
_BUILD_LOCK = threading.Lock()


def _safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", text).strip("_")


def cuda_available() -> bool:
    return bool(torch.cuda.is_available())


def _load_prebuilt_extension():
    try:
        return importlib.import_module("mlbricks._gauss_cuda")
    except Exception:
        return None


def load_cuda_extension(verbose: bool = False):
    """Load the prebuilt native extension; JIT is opt-in for development."""
    if not cuda_available():
        return None

    device = torch.cuda.current_device()
    major, minor = torch.cuda.get_device_capability(device)
    torch_ver = torch.__version__.split("+")[0]
    cuda_ver = str(torch.version.cuda)
    key = (major, minor, torch_ver, cuda_ver)

    if key in _EXTENSIONS:
        return _EXTENSIONS[key]

    with _BUILD_LOCK:
        if key in _EXTENSIONS:
            return _EXTENSIONS[key]

        ext = _load_prebuilt_extension()
        if ext is not None:
            _EXTENSIONS[key] = ext
            return ext

        # Production installs should build mlbricks._gauss_cuda once at
        # installation/build time. Runtime JIT remains available only as an
        # explicit development escape hatch.
        allow_jit = os.environ.get("MLBRICKS_GAUSS_ALLOW_JIT", "0").lower() in {
            "1", "true", "yes", "on"
        }
        if not allow_jit:
            return None

        try:
            from torch.utils.cpp_extension import load
        except Exception:
            return None

        here = Path(__file__).resolve().parent
        sources = [
            str(here / "bolt" / "bolt_attention_bindings.cpp"),
            str(here / "bolt" / "bolt_attention_cuda.cu"),
        ]
        gpu_name = torch.cuda.get_device_name(device)
        token = hashlib.sha1(
            f"{gpu_name}|{major}.{minor}|{torch_ver}|{cuda_ver}".encode()
        ).hexdigest()[:10]
        module_name = f"mlbricks_gauss_{major}{minor}_{token}"

        old_arch = os.environ.get("TORCH_CUDA_ARCH_LIST")
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
        os.environ.setdefault("MAX_JOBS", "2")
        try:
            ext = load(
                name=module_name,
                sources=sources,
                extra_cflags=["-O3"],
                extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
                with_cuda=True,
                verbose=verbose,
            )
        except Exception:
            ext = None
        finally:
            if old_arch is None:
                os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
            else:
                os.environ["TORCH_CUDA_ARCH_LIST"] = old_arch

        if ext is not None:
            _EXTENSIONS[key] = ext
        return ext


@dataclass(frozen=True)
class KernelConfig:
    mode: int
    splits: int

    @property
    def mode_name(self) -> str:
        if self.mode == 0:
            return "stream"
        if self.mode == 1:
            return "tiled8"
        if self.mode == 2:
            return "standalone_twopass"
        if self.mode == 3:
            return "r16_subwarp"
        return f"mode{self.mode}"


class TuneStore:
    def __init__(self):
        root = Path.home() / ".cache" / "mlbricks" / "gauss"
        root.mkdir(parents=True, exist_ok=True)
        self.root = root
        self._memory: Dict[str, dict] = {}
        self._loaded_paths = set()
        self._lock = threading.Lock()

    def _path(self) -> Path:
        if not cuda_available():
            return self.root / "cpu.json"
        d = torch.cuda.current_device()
        major, minor = torch.cuda.get_device_capability(d)
        gpu = _safe_name(torch.cuda.get_device_name(d))
        tv = torch.__version__.split("+")[0]
        cv = str(torch.version.cuda)
        return self.root / f"{gpu}_sm{major}{minor}_torch{tv}_cuda{cv}.json"

    def load(self) -> dict:
        p = self._path()
        key = str(p)
        with self._lock:
            if key in self._loaded_paths:
                return self._memory.setdefault(key, {})
            data = {}
            if p.exists():
                try:
                    data = json.loads(p.read_text())
                except Exception:
                    data = {}
            self._memory[key] = data
            self._loaded_paths.add(key)
            return data

    def save(self, data: dict) -> None:
        p = self._path()
        key = str(p)
        with self._lock:
            self._memory[key] = dict(data)
            self._loaded_paths.add(key)
            tmp = p.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=2, sort_keys=True))
            tmp.replace(p)

    def clear_memory(self) -> None:
        with self._lock:
            self._memory.clear()
            self._loaded_paths.clear()


TUNE_STORE = TuneStore()


def _pow2_bucket(x: int) -> int:
    if x <= 1:
        return 1
    return 1 << int(math.ceil(math.log2(x)))


def shape_key(kind: str, B: int, H: int, T: int, W: int, dtype: str = "fp16") -> str:
    return (
        f"{kind}|B{_pow2_bucket(B)}|H{_pow2_bucket(H)}|"
        f"T{_pow2_bucket(T)}|W{W}|{dtype}"
    )


def candidate_splits(B: int, H: int, T: int, mode: int):
    if not cuda_available():
        return [1]
    sm = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    bh = B * H
    mults = [1, 2, 4, 8] if mode == 0 else [1, 2, 3]
    vals = {1}
    for mult in mults:
        vals.add(max(1, math.ceil((sm * mult) / bh)))
    for s in list(vals):
        vals.add(max(1, s // 2))
        vals.add(min(128, s * 2))
    min_tokens = 64 if mode == 0 else 192
    max_by_tokens = max(1, math.ceil(T / min_tokens))
    return sorted(s for s in vals if 1 <= s <= 128 and s <= max_by_tokens)


def heuristic_config(*, kind: str, B: int, H: int, T: int, W: int) -> KernelConfig:
    # Deterministic no-benchmark fallback for autotune_kernels=False.
    mode = 0 if T < 768 else 1
    vals = candidate_splits(B, H, T, mode)
    if not vals:
        return KernelConfig(mode, 1)
    # Favor enough independent work to fill the GPU without over-splitting.
    return KernelConfig(mode, vals[len(vals) // 2])


class AttentionWorkspace:
    def __init__(self, B: int, H: int, W: int, splits: int, device):
        bh = B * H
        self.pm = torch.empty(bh, splits, device=device, dtype=torch.float32)
        self.pl = torch.empty(bh, splits, device=device, dtype=torch.float32)
        self.po = torch.empty(bh, splits, W, device=device, dtype=torch.float32)
        self.out = torch.empty(B, H, W, device=device, dtype=torch.float16)


class WorkspacePool:
    def __init__(self):
        self._items = {}

    def get(self, kind: str, B: int, H: int, W: int, splits: int, device):
        # T is intentionally NOT part of this key: no workspace buffer depends
        # on sequence length. This prevents one retained GPU allocation set per
        # decode position during autoregressive generation.
        key = (kind, B, H, W, splits, str(device))
        ws = self._items.get(key)
        if ws is None:
            ws = AttentionWorkspace(B, H, W, splits, device)
            self._items[key] = ws
        return ws

    def clear(self):
        self._items.clear()

    def __len__(self):
        return len(self._items)


WORKSPACES = WorkspacePool()


@torch.no_grad()
def _median_us(fn, trials: int = 5, warmup: int = 2) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    vals = []
    for _ in range(trials):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); torch.cuda.synchronize()
        vals.append(a.elapsed_time(b) * 1000.0)
    vals.sort()
    return float(vals[len(vals) // 2])


@torch.no_grad()
def autotune(
    *, kind: str, q: torch.Tensor, a: torch.Tensor, b: torch.Tensor,
    head_dim: int, ext, force: bool = False, used_length: int | None = None,
) -> KernelConfig:
    B, H, W = q.shape
    T = a.shape[2] if used_length is None else int(used_length)
    if T < 1 or T > a.shape[2]:
        raise ValueError("used_length must be in [1, cache_capacity]")
    key = shape_key(kind, B, H, T, W, str(q.dtype))
    cache = TUNE_STORE.load()
    if (not force) and key in cache:
        x = cache[key]
        return KernelConfig(int(x["mode"]), int(x["splits"]))

    best = None
    best_us = float("inf")
    scale = 1.0 / math.sqrt(float(head_dim))
    for mode in (0, 1):
        for splits in candidate_splits(B, H, T, mode):
            ws = WORKSPACES.get(kind, B, H, W, splits, q.device)
            if kind == "baseline":
                if used_length is not None and hasattr(ext, "baseline_decode_out_used"):
                    fn = lambda: ext.baseline_decode_out_used(
                        q, a, b, ws.pm, ws.pl, ws.po, ws.out,
                        scale, mode, splits, T,
                    )
                else:
                    fn = lambda: ext.baseline_decode_out(
                        q, a, b, ws.pm, ws.pl, ws.po, ws.out,
                        scale, mode, splits,
                    )
            else:
                if used_length is not None and hasattr(ext, "gauss_decode_out_used"):
                    fn = lambda: ext.gauss_decode_out_used(
                        q, a, b, ws.pm, ws.pl, ws.po, ws.out,
                        scale, mode, splits, T,
                    )
                else:
                    fn = lambda: ext.gauss_decode_out(
                        q, a, b, ws.pm, ws.pl, ws.po, ws.out,
                        scale, mode, splits,
                    )
            us = _median_us(fn)
            if us < best_us:
                best_us = us
                best = KernelConfig(mode, splits)

    if best is None:
        best = heuristic_config(kind=kind, B=B, H=H, T=T, W=W)
    cache[key] = {"mode": best.mode, "splits": best.splits, "latency_us": best_us}
    TUNE_STORE.save(cache)
    return best


@torch.no_grad()
def autotune_gauss_no_o(
    *, q: torch.Tensor, c: torch.Tensor, rho: torch.Tensor,
    weight: torch.Tensor, head_dim: int, ext, force: bool = False,
    used_length: int | None = None,
    c_now: torch.Tensor | None = None,
    rho_now: torch.Tensor | None = None,
    position: int | None = None,
) -> KernelConfig:
    """Autotune the actual no-O Bolt decode + output projection path.

    For the production B=1/H=4/R=16/D=128 shape, preserve the existing exact
    standalone two-pass mode (2) and add the R16 subwarp mode (3) as another
    candidate.  The winner is cached per GPU/context bucket, so the new kernel
    is used only where it is actually faster.
    """
    B, H, R = q.shape
    T = c.shape[2] if used_length is None else int(used_length)
    if T < 1 or T > c.shape[2]:
        raise ValueError("used_length must be in [1, cache_capacity]")

    is_append = c_now is not None or rho_now is not None or position is not None
    if is_append:
        if c_now is None or rho_now is None or position is None:
            raise ValueError("c_now, rho_now and position must be provided together")
        if int(position) + 1 != T:
            raise ValueError("append position must equal used_length - 1")

    D = int(weight.size(0))
    supports_r16 = bool(
        B == 1 and H == 4 and R == 16 and D == 128
        and hasattr(ext, "gauss_r16_scan_supported")
        and bool(ext.gauss_r16_scan_supported())
    )
    kind = "gauss_no_o_r16_v1_append" if is_append else "gauss_no_o_r16_v1"
    key = shape_key(kind, B, H, T, R, str(q.dtype))
    cache = TUNE_STORE.load()
    if (not force) and key in cache:
        x = cache[key]
        mode = int(x["mode"])
        if mode != 3 or supports_r16:
            return KernelConfig(mode, int(x["splits"]))

    # Keep all existing exact implementations available.  Mode 3 is added only
    # for the validated R16 production geometry.
    modes = (0, 1, 2, 3) if supports_r16 else (0, 1, 2)
    scale = 1.0 / math.sqrt(float(head_dim))
    best = None
    best_us = float("inf")

    for mode in modes:
        # stream/two-pass use one warp; tiled8/R16-subwarp use 8 warps.
        split_mode = 0 if mode in (0, 2) else 1
        split_candidates = set(candidate_splits(B, H, T, split_mode))
        if mode in (0, 2):
            max_by_tokens = max(1, min(128, math.ceil(T / 64)))
            for value in (1, 2, 4, 8, 16, 32, 64, 128):
                if value <= max_by_tokens:
                    split_candidates.add(value)
            # Preserve the historical standalone split schedule as a candidate.
            split_candidates.add(max(1, min(32, math.ceil(T / 256))))

        for splits in sorted(split_candidates):
            ws = WORKSPACES.get("gauss", B, H, R, splits, q.device)
            if is_append:
                fn_call = lambda mode=mode, splits=splits, ws=ws: ext.gauss_decode_append_project_out(
                    q, c_now, rho_now, c, rho,
                    ws.pm, ws.pl, ws.po, weight,
                    scale, mode, splits, int(position),
                )
            else:
                fn_call = lambda mode=mode, splits=splits, ws=ws: ext.gauss_decode_project_out_used(
                    q, c, rho, ws.pm, ws.pl, ws.po, weight,
                    scale, mode, splits, T,
                )
            us = _median_us(fn_call)
            if us < best_us:
                best_us = us
                best = KernelConfig(mode, splits)

    if best is None:
        # This helper is called only for the standalone no-O geometry, so retain
        # its proven exact two-pass schedule as the deterministic fallback.
        best = KernelConfig(2, max(1, min(32, math.ceil(T / 256))))
    cache[key] = {"mode": best.mode, "splits": best.splits, "latency_us": best_us}
    TUNE_STORE.save(cache)
    return best


@torch.no_grad()
def autotune_gauss_rope(
    *, q: torch.Tensor, c: torch.Tensor, rho: torch.Tensor, head_dim: int,
    ext, rope_base: float, rope_dim: int, force: bool = False,
    used_length: int | None = None,
) -> KernelConfig:
    """Autotune native Gauss RoPE decode without materializing rotated C."""
    B, H, W = q.shape
    T = c.shape[2] if used_length is None else int(used_length)
    if T < 1 or T > c.shape[2]:
        raise ValueError("used_length must be in [1, cache_capacity]")
    kind = "gauss_rope"
    key = shape_key(kind, B, H, T, W, str(q.dtype)) + f"|base{float(rope_base):g}|rd{int(rope_dim)}"
    cache = TUNE_STORE.load()
    if (not force) and key in cache:
        x = cache[key]
        return KernelConfig(int(x["mode"]), int(x["splits"]))

    best = None
    best_us = float("inf")
    scale = 1.0 / math.sqrt(float(head_dim))
    for mode in (0, 1):
        for splits in candidate_splits(B, H, T, mode):
            ws = WORKSPACES.get(kind, B, H, W, splits, q.device)
            fn = lambda mode=mode, splits=splits, ws=ws: ext.gauss_rope_decode_out_used(
                q, c, rho, ws.pm, ws.pl, ws.po, ws.out,
                scale, mode, splits, T, float(rope_base), int(rope_dim),
            )
            us = _median_us(fn)
            if us < best_us:
                best_us = us
                best = KernelConfig(mode, splits)

    if best is None:
        best = heuristic_config(kind=kind, B=B, H=H, T=T, W=W)
    cache[key] = {"mode": best.mode, "splits": best.splits, "latency_us": best_us}
    TUNE_STORE.save(cache)
    return best
