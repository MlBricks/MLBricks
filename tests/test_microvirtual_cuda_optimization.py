from __future__ import annotations

from pathlib import Path

import torch

from mlbricks import MicroVirtualFFN


def test_micro_refine_equations_remain_exact_python_reference():
    torch.manual_seed(712)
    layer = MicroVirtualFFN(32, hidden_dim=7, refinements=3, use_native=False)
    with torch.no_grad():
        layer.down.normal_(0.0, 0.02)
    x = torch.randn(2, 5, 32)

    expected = x
    for i in range(layer.refinements):
        expected = expected + layer.forward_python(expected, i)

    actual = layer.refine(x)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_micro_packed_cuda_path_avoids_split_contiguous_copies_in_source():
    root = Path(__file__).resolve().parents[1]
    cpp = (root / "mlbricks/ffnbrick/csrc/ffnbrick.cpp").read_text()
    cuda = (root / "mlbricks/ffnbrick/csrc/cuda/fused_ops.cu").read_text()

    assert "silu_mul_packed_cuda(gate_up.contiguous(), hidden_dim)" in cpp
    assert "at::addmm(" in cpp
    assert "silu_mul_packed_kernel" in cuda
    # The optimized branch must not reintroduce the two narrow-view copies.
    optimized = cpp[cpp.index("Tensor micro_hidden_packed_2d"):cpp.index("} // namespace")]
    assert "gate.contiguous()" not in optimized
    assert "value.contiguous()" not in optimized
