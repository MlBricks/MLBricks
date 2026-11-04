import inspect
import pytest
import torch
from mlbricks import ESA
from mlbricks.esa.fast_engine.config import ESAConfig


def test_esa_auto_defaults():
    sig = inspect.signature(ESA)
    assert sig.parameters['backend'].default=='auto'
    assert sig.parameters['device'].default is None
    assert ESAConfig(embd=24,head=6).backend=='auto'


def test_direct_esa_backend_selection():
    esa=ESA(embd=24,head=6,device='cpu')
    assert esa.resolved_backend()=='auto' and esa.scan=='auto'
    esa.set_backend('pytorch')
    assert esa.resolved_backend()=='pytorch' and esa.scan=='reference'
    esa.set_backend('auto')
    assert esa.resolved_backend()=='auto' and esa.scan=='auto'
    esa.set_backend('native')
    with pytest.raises(RuntimeError,match='requires a CUDA input'):
        esa(torch.randn(1,2,24))


def test_cpu_native_reference():
    esa=ESA(embd=24,head=6,backend='pytorch',device='cpu')
    output,state=esa(torch.randn(2,5,24))
    assert output.shape==(2,5,24) and state.dtype==torch.float32
