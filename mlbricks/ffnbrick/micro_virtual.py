from __future__ import annotations

import math
import torch
import torch.nn.functional as F
from torch import nn

from . import native as _native
from ..runtime import normalize_backend


class MicroVirtualFFN(nn.Module):
    """Pass-specific gated Micro-FFN for lightweight virtual refinement.

    Each pass computes a SwiGLU-like hidden update::

        silu(W_gate x) * (W_up x) -> W_down

    ``W_down`` is zero-initialized, so the module begins as an exact zero
    update and can safely be inserted into a residual path.

    When the optional native extension is built, ``forward`` runs through the
    C++ backend. CUDA tensors keep GEMMs in cuBLAS/ATen, consume packed
    gate/value activations without intermediate copies, and fuse residual
    accumulation into the down-projection GEMM during multi-pass refinement.
    """

    def __init__(
        self,
        d_model: int,
        hidden_dim: int = 64,
        refinements: int = 1,
        use_native: bool | None = None,
        fused_cuda: bool = True,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        if d_model <= 0 or hidden_dim <= 0 or refinements <= 0:
            raise ValueError("d_model, hidden_dim, and refinements must be positive")
        self.d_model = int(d_model)
        self.hidden_dim = int(hidden_dim)
        self.refinements = int(refinements)
        self.backend = normalize_backend(backend, warn_legacy=True)
        if use_native is not None and self.backend == "auto":
            self.backend = "auto" if bool(use_native) else "pytorch"
        self.use_native = self.backend != "pytorch"
        self.fused_cuda = bool(fused_cuda)
        self._ffnbrick_native_cache = None

        self.gate = nn.Parameter(torch.empty(refinements, hidden_dim, d_model))
        self.up = nn.Parameter(torch.empty(refinements, hidden_dim, d_model))
        self.down = nn.Parameter(torch.zeros(refinements, d_model, hidden_dim))
        nn.init.kaiming_uniform_(self.gate, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.up, a=math.sqrt(5))
        nn.init.zeros_(self.down)

    def set_backend(self, backend: str, *, recursive: bool = True):
        del recursive
        self.backend = normalize_backend(backend, warn_legacy=True)
        self.use_native = self.backend != "pytorch"
        return self

    def resolved_backend(self) -> str:
        if self.backend == "pytorch":
            return "pytorch"
        if self.backend == "native":
            return "native-required"
        return "planner(auto)" if _native.is_available() else "pytorch"

    def _require_native_for(self, x: torch.Tensor) -> None:
        if self.backend == "native" and not _native.inference_native_allowed(self, x):
            raise RuntimeError(
                "MicroVirtualFFN backend='native' requested but native eager inference "
                "is unavailable for this call"
            )

    def reset_identity(self) -> None:
        """Restore exact zero-update initialization."""
        with torch.no_grad():
            nn.init.zeros_(self.down)

    def _validate_forward(self, x: torch.Tensor, refinement_index: int) -> None:
        if x.shape[-1] != self.d_model:
            raise ValueError(f"expected last dimension {self.d_model}, got {x.shape[-1]}")
        if not 0 <= refinement_index < self.refinements:
            raise IndexError("refinement_index is out of range")

    def forward_python(self, x: torch.Tensor, refinement_index: int = 0) -> torch.Tensor:
        """Reference PyTorch implementation, useful for parity/benchmarking."""
        self._validate_forward(x, refinement_index)
        gate = F.linear(x, self.gate[refinement_index])
        value = F.linear(x, self.up[refinement_index])
        hidden = F.silu(gate) * value
        return F.linear(hidden, self.down[refinement_index])

    def forward(self, x: torch.Tensor, refinement_index: int = 0) -> torch.Tensor:
        self._validate_forward(x, refinement_index)
        self._require_native_for(x)
        if _native.inference_native_allowed(self, x):
            return _native.micro_virtual_forward(self, x, refinement_index)
        return self.forward_python(x, refinement_index)

    def refine(self, x: torch.Tensor) -> torch.Tensor:
        """Apply every configured pass as residual refinement.

        Eager no-grad inference runs every pass inside one native call;
        training and torch.compile retain the original PyTorch loop.
        """
        self._require_native_for(x)
        if _native.inference_native_allowed(self, x):
            return _native.micro_virtual_refine(self, x)
        for index in range(self.refinements):
            x = x + self.forward_python(x, index)
        return x
