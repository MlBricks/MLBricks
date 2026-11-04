"""Only the three requested Fast-EsaQ API names change."""
import inspect
import pytest
import torch

from mlbricks import ESA as MLBricksESA
from mlbricks.esa.fast_engine import ESA, CurrentESA, BaseESA, ESAConfig, make_esa, make_model
from mlbricks.esa.fast_engine.core import ESA as CoreESA


def test_direct_names_defaults_and_custom():
    m=MLBricksESA(embd=24,head=6,device='cpu',backend='pytorch',cumpass=8,parallel_chunk=64,cuda_warps=4)
    assert (m.cumpass,m.parallel_chunk,m.cuda_warps)==(8,64,4)
    y,state=m(torch.randn(2,7,24))
    assert y.shape==(2,7,24) and state.shape==(2,6,4)
    d=MLBricksESA(embd=24,head=6,device='cpu',backend='pytorch')
    assert (d.cumpass,d.parallel_chunk,d.cuda_warps)==(16,128,8)


@pytest.mark.parametrize("maker", [
    lambda **kw: ESA(embd=24, head=6, backend="pytorch", **kw),
    lambda **kw: CurrentESA(embd=24, head=6, backend="pytorch", **kw),
    lambda **kw: BaseESA(embd=24, head=6, backend="pytorch", **kw),
    lambda **kw: make_esa(dim=24, heads=6, scan="reference", **kw),
    lambda **kw: make_esa(variant="base", dim=24, heads=6, scan="reference", **kw),
])
def test_engine_public_constructors_accept_new_names(maker):
    m = maker(cumpass=8, parallel_chunk=64, cuda_warps=4)
    assert (m.cumpass, m.parallel_chunk, m.cuda_warps) == (8, 64, 4)


def test_model_factories_and_config():
    model = make_model(vocab_size=32, dim=24, heads=6, layers=1,
                       scan="reference", cumpass=8, parallel_chunk=64, cuda_warps=4)
    assert model.blocks[0].esa is not None
    cfg = ESAConfig(embd=24, head=6, backend="pytorch", cumpass=8, parallel_chunk=64, cuda_warps=4)
    layer = cfg.build(device="cpu")
    assert (layer.cumpass, layer.parallel_chunk, layer.cuda_warps) == (8, 64, 4)


def test_core_signature_only_new_keywords():
    params = inspect.signature(CoreESA).parameters
    for name in ("cumpass", "parallel_chunk", "cuda_warps"):
        assert name in params
    for name in ("thunder_compass", "swqr_chunk", "swqr_warps"):
        assert name not in params
