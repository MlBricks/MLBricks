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

    for used in (1, 255, 256, 257, 512, 2048, 8192, 16384):
        cfg = bolt._standalone_no_o_config(batch=1, used=used)
        assert cfg is not None
        assert cfg.mode == 3
        assert cfg.mode_name == "r16_subwarp"
        assert 1 <= cfg.splits <= 128

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


def test_no_o_second_stage_planner_is_r16_only():
    root = Path(__file__).resolve().parents[1]
    backend = (root / "mlbricks" / "_backend.py").read_text()
    start = backend.index("def autotune_gauss_no_o(")
    end = backend.index("def autotune_gauss_rope(", start)
    section = backend[start:end]

    # Compatibility entry point remains, but it no longer races modes 0/1/2/3.
    assert "for mode in" not in section
    assert "TUNE_STORE" not in section
    assert "_median_us" not in section
    assert "r16_no_o_config" in section
    assert "KernelConfig(3" in backend

def test_r16_gpu_aware_split_selector_scales_with_context_and_sms():
    from mlbricks._backend import r16_gpu_aware_splits

    # Tesla T4-like geometry: 40 SMs, B=1, H=4 => occupancy quantum 10.
    expected_t4 = {
        512: 2,
        2048: 10,
        4096: 20,
        8192: 40,
        16384: 40,
        32768: 40,
        65536: 40,
        131072: 40,
        262144: 40,
    }
    for context, expected in expected_t4.items():
        assert r16_gpu_aware_splits(
            B=1, H=4, T=context, sm_count=40
        ) == expected

    # A larger GPU must adapt from hardware capacity instead of inheriting
    # the T4's 40-split ceiling.
    assert r16_gpu_aware_splits(B=1, H=4, T=32768, sm_count=80) == 80
    assert r16_gpu_aware_splits(B=1, H=4, T=131072, sm_count=80) == 80
    assert r16_gpu_aware_splits(B=1, H=4, T=32768, sm_count=108) == 108

    # The public selector remains bounded by the native workspace contract.
    assert r16_gpu_aware_splits(B=1, H=4, T=1_000_000, sm_count=132) == 128


def test_r16_gpu_aware_split_selector_validates_inputs():
    import pytest
    from mlbricks._backend import r16_gpu_aware_splits

    with pytest.raises(ValueError):
        r16_gpu_aware_splits(B=1, H=4, T=0, sm_count=40)
    with pytest.raises(ValueError):
        r16_gpu_aware_splits(B=1, H=4, T=512, sm_count=0)
