from __future__ import annotations

import inspect
import pytest

from mlbricks import ElasticBit, RuntimeMatrix, BitAnalysis, BackendInfo


def test_clean_public_surface_has_no_fixed_bit_api():
    assert not hasattr(ElasticBit, "bits")
    assert not hasattr(ElasticBit, "quantize")
    assert not hasattr(ElasticBit, "dequantize")
    assert not hasattr(ElasticBit, "quantize_module")
    assert not hasattr(ElasticBit, "apply")
    assert not hasattr(RuntimeMatrix, "from_auto")


def test_public_api_uses_camel_case_names():
    sig = inspect.signature(ElasticBit.compressMatrix)
    assert "calibrationData" in sig.parameters
    assert "decodePolicy" in sig.parameters
    assert "calibration_data" not in sig.parameters
    assert "decode_policy" not in sig.parameters
    assert hasattr(RuntimeMatrix, "setDecodePolicy")


def test_backend_api_is_safe_without_native_extension():
    info = ElasticBit.backend()
    assert isinstance(info, BackendInfo)
    assert isinstance(info.available, bool)
    assert isinstance(info.validated, bool)
    if info.available:
        assert info.prefill == "native"
        assert info.decodePlanner in {"t4Validated", "cudaAutoTune"}


def test_native_only_calls_fail_cleanly_when_extension_missing():
    if ElasticBit.backend().available:
        pytest.skip("native ElasticBit extension is available in this environment")
    with pytest.raises(RuntimeError, match="CUDA runtime"):
        ElasticBit.analyze([[1.0, 2.0]], [[1.0, 1.0]], 0.01)


def test_runtime_matrix_has_only_threshold_driven_factories():
    assert callable(RuntimeMatrix.compress)
    assert callable(RuntimeMatrix.load)
    assert not hasattr(RuntimeMatrix, "auto")
