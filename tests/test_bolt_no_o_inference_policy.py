from pathlib import Path


def test_bolt_training_keeps_native_stage1_sdpa_and_inference_has_no_o_route():
    root = Path(__file__).resolve().parents[1]
    attention = (root / "mlbricks" / "bolt" / "attention.py").read_text()
    cuda = (root / "mlbricks" / "bolt" / "bolt_attention_cuda.cu").read_text()
    bindings = (root / "mlbricks" / "bolt" / "bolt_attention_bindings.cpp").read_text()

    # Training stays on the proven native Stage-1 + SDPA route.
    assert "q, c, rho = self._stage1_training(x)" in attention
    assert "y = _sdpa(" in attention

    # Recurrent inference can skip the global O activation and project directly.
    assert "gauss_decode_append_project_out" in attention
    assert "gauss_decode_project_out_used" in attention
    assert "merge_project_partial" in cuda
    assert 'm.def("gauss_decode_append_project_out"' in bindings


def test_no_o_route_preserves_bias_fallback():
    root = Path(__file__).resolve().parents[1]
    attention = (root / "mlbricks" / "bolt" / "attention.py").read_text()
    assert "self.out_proj.bias is None" in attention
