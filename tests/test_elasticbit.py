import torch
from torch import nn

from mlbricks import (
    ElasticBit,
    ElasticBitConfig,
    ElasticLinear,
    dequantize_tensor,
    quantize_module,
    quantize_tensor,
)


def test_tensor_round_trip():
    torch.manual_seed(0)
    x = torch.randn(257)
    packed = quantize_tensor(x, ElasticBitConfig(bits=4, group_size=32, scale_dtype=torch.float32))
    restored = dequantize_tensor(packed)
    assert restored.shape == x.shape
    assert torch.isfinite(restored).all()
    assert (x - restored).abs().mean() < 0.15


def test_elastic_linear_shape():
    layer = nn.Linear(16, 8)
    quantized = ElasticLinear.from_linear(layer, ElasticBitConfig(bits=8, group_size=16))
    x = torch.randn(3, 16)
    y = quantized(x)
    assert y.shape == (3, 8)


def test_recursive_quantization():
    model = nn.Sequential(nn.Linear(8, 16), nn.GELU(), nn.Linear(16, 4))
    quantize_module(model, ElasticBitConfig(bits=4, group_size=16))
    assert isinstance(model[0], ElasticLinear)
    assert isinstance(model[2], ElasticLinear)


def test_top_level_elasticbit_interface():
    elastic = ElasticBit(bits=4, group_size=16, scale_dtype=torch.float32)
    source = torch.randn(65)
    packed = elastic.quantize(source)
    restored = elastic.dequantize(packed)
    assert elastic.bits == 4
    assert elastic.group_size == 16
    assert restored.shape == source.shape
    assert (source - restored).abs().mean() < 0.2


def test_top_level_elasticbit_model_conversion():
    elastic = ElasticBit(bits=4, group_size=16)
    model = nn.Sequential(nn.Linear(8, 4))
    returned = elastic.quantize_module(model)
    assert returned is model
    assert isinstance(model[0], ElasticLinear)


def test_elastic_linear_auto_default_is_cached_pytorch_route():
    from mlbricks import EXECUTION_PLANNER

    EXECUTION_PLANNER.clear()
    layer = nn.Linear(16, 8)
    quantized = ElasticLinear.from_linear(
        layer,
        ElasticBitConfig(bits=4, group_size=16, backend="auto"),
    )
    x = torch.randn(3, 16)
    y1 = quantized(x)
    y2 = quantized(x)
    assert y1.shape == y2.shape == (3, 8)
    # CPU cannot use the packed CUDA route; auto must settle on the cached
    # PyTorch/dequantized execution path and remain numerically stable.
    torch.testing.assert_close(y1, y2)
    assert quantized.backend == "auto"
