from __future__ import annotations

import inspect
import pytest

from mlbricks import ElasticBit, RuntimeMatrix, BitAnalysis, BackendInfo
from mlbricks.elasticbit.core import _semanticOutputWeightIds


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
    assert sig.parameters["decodePolicy"].default == "hardwareNative"
    assert inspect.signature(ElasticBit.loadMatrix).parameters["decodePolicy"].default == "hardwareNative"
    assert inspect.signature(RuntimeMatrix.compress).parameters["decodePolicy"].default == "hardwareNative"
    assert inspect.signature(RuntimeMatrix.load).parameters["decodePolicy"].default == "hardwareNative"
    assert "calibration_data" not in sig.parameters
    assert "decode_policy" not in sig.parameters
    assert "preserveOutputHead" in inspect.signature(ElasticBit.compress).parameters
    assert inspect.signature(ElasticBit.compress).parameters["preserveOutputHead"].default is True
    assert hasattr(RuntimeMatrix, "setDecodePolicy")


def test_backend_api_is_safe_without_native_extension():
    info = ElasticBit.backend()
    assert isinstance(info, BackendInfo)
    assert isinstance(info.available, bool)
    assert isinstance(info.validated, bool)
    if info.available:
        assert info.prefill == "native"
        assert info.decodePlanner in {"t4Validated", "cudaAutoTune"}
        assert info.decodeEngine == "FastWarp"
        assert info.decodeStatus in {"production", "experimental"}
        assert isinstance(info.decodeValidation, str)
        if info.validated:
            assert info.decodePlanner == "t4Validated"
            assert info.decodeStatus == "production"
            assert info.decodeValidation == "fastwarp_t4_sm75_m1_w4w8w16_v1"
        else:
            assert info.decodeStatus == "experimental"
            assert info.decodeValidation == "unvalidated"


def test_native_only_calls_fail_cleanly_when_extension_missing():
    if ElasticBit.backend().available:
        pytest.skip("native ElasticBit extension is available in this environment")
    with pytest.raises(RuntimeError, match="CUDA runtime"):
        ElasticBit.analyze([[1.0, 2.0]], [[1.0, 1.0]], 0.01)


def test_runtime_matrix_has_only_threshold_driven_factories():
    assert callable(RuntimeMatrix.compress)
    assert callable(RuntimeMatrix.load)
    assert not hasattr(RuntimeMatrix, "auto")


def test_semantic_output_head_detection_is_model_agnostic():
    import torch.nn as nn

    class ToyDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(31, 8)
            self.block = nn.Linear(8, 8, bias=False)
            self.vocab_projection = nn.Linear(8, 31, bias=False)

        def get_output_embeddings(self):
            return self.vocab_projection

    model = ToyDecoder()
    protected = _semanticOutputWeightIds(model)
    assert protected == {id(model.vocab_projection.weight)}
    assert id(model.block.weight) not in protected


def test_semantic_output_head_detection_never_depends_on_module_name():
    import torch.nn as nn

    class OddNames(nn.Module):
        def __init__(self):
            super().__init__()
            self.not_a_head = nn.Linear(4, 7, bias=False)

        def get_output_embeddings(self):
            return self.not_a_head

    model = OddNames()
    assert _semanticOutputWeightIds(model) == {id(model.not_a_head.weight)}
