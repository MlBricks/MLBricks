"""Production BOLT contracts across forward, prefill, and one-token decode."""
import pytest
import torch
from mlbricks import Bolt, BoltConfig, BoltModel


def test_cpu_override_is_supported_for_reference_workloads():
    layer = Bolt(64,4,profile="compact",device="cpu",precision="fp32",backend="pytorch").eval()
    data = torch.randn(2,5,64)
    with torch.no_grad():
        forward = layer(data)
        prefix,cache = layer.prefill(data)
        torch.testing.assert_close(prefix,forward,rtol=2e-5,atol=2e-6)
        assert prefix.device.type=="cpu" and prefix.dtype==torch.float32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("profile",["standard","compact"])
@pytest.mark.parametrize("precision",["fp32","auto"])
def test_cuda_cached_and_forward_use_same_precision(profile,precision):
    torch.manual_seed(919)
    layer=Bolt(384,12,profile=profile,device="auto",precision=precision,backend="pytorch").eval()
    assert next(layer.parameters()).is_cuda
    inputs=torch.randn(1,9,384)  # API transparently moves CPU inputs to CUDA
    with torch.no_grad():
        entire=layer(inputs)
        prefix,cache=layer.prefill(inputs[:,:6])
        assert prefix.dtype==entire.dtype, (profile,precision, prefix.dtype, entire.dtype)
        torch.testing.assert_close(prefix.float(),entire[:,:6].float(),
                                   rtol=.025 if precision=="auto" else 2e-5,
                                   atol=.035 if precision=="auto" else 2e-6)
        output,cache=layer.decode_step(inputs[:,6:7],cache)
        assert output.dtype==entire.dtype
        torch.testing.assert_close(output.float(),entire[:,6:7].float(),
                                   rtol=.035 if precision=="auto" else 3e-5,
                                   atol=.045 if precision=="auto" else 5e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("memory",[True,False])
def test_cuda_model_default_device_and_memory_option(memory):
    cfg=BoltConfig(vocab_size=97,context=32,layers=2,dim=64,heads=4,
                   profile="compact",state_memory=memory,device="auto",precision="fp32",backend="pytorch")
    model=BoltModel(cfg).eval()
    assert model.device.type=="cuda"
    ids=torch.randint(0,97,(1,8),device="cpu")
    with torch.no_grad():
        output=model(ids)
        pref,cs=model.prefill(ids[:,:5])
        assert output.shape==(1,8,97)
        assert pref.shape==(1,5,97)
        torch.testing.assert_close(pref,output[:,:5],rtol=4e-5,atol=1e-5)
        if not memory:
            assert all(c[1] is None for c in cs)
