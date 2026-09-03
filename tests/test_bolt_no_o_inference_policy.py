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
    assert "gauss_twopass_r16_partial" in cuda
    assert "gauss_twopass_r16_append_partial" in cuda
    assert "standalone two-pass no-O mode requires H=4, R=16, D=128" in cuda
    assert 'm.def("gauss_decode_append_project_out"' in bindings


def test_no_o_route_preserves_bias_fallback():
    root = Path(__file__).resolve().parents[1]
    attention = (root / "mlbricks" / "bolt" / "attention.py").read_text()
    assert "self.out_proj.bias is None" in attention


def test_standalone_no_o_fast_config_is_narrow_and_exact():
    from mlbricks import Bolt

    bolt = Bolt(
        d_model=128,
        num_heads=4,
        latent_dim=16,
        bias=False,
        position=None,
        backend="pytorch",
    )

    expected = {
        1: 1,
        255: 1,
        256: 1,
        257: 2,
        512: 2,
        2048: 8,
        8192: 32,
        16384: 32,
    }
    for used, splits in expected.items():
        cfg = bolt._standalone_no_o_config(batch=1, used=used)
        assert cfg is not None
        assert cfg.mode == 2
        assert cfg.splits == splits

    # Do not force the standalone schedule outside the measured batch/shape.
    assert bolt._standalone_no_o_config(batch=4, used=8192) is None
    assert Bolt(
        d_model=256,
        num_heads=4,
        latent_dim=16,
        bias=False,
        backend="pytorch",
    )._standalone_no_o_config(batch=1, used=8192) is None


def test_fast_no_o_cuda_reducer_is_present_with_generic_fallback():
    root = Path(__file__).resolve().parents[1]
    cuda = (root / "mlbricks" / "bolt" / "bolt_attention_cuda.cu").read_text()

    assert "merge_project_h4_r16_d128" in cuda
    assert "use_fast_no_o_merge" in cuda
    assert "launch_merge_project" in cuda
    assert "merge_project_partial" in cuda
