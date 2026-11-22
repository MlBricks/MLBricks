import math
import tempfile
from pathlib import Path

import pytest
import torch

from mlbricks import (ESA, LegacyESA, CurrentESA, BaseESA, DEFAULT_CURRENT_HALF_LIVES,
                      CurrentESAConfig, CurrentESAModel, make_esa,
                      make_current_esa_model, current_esa_forward_tokens,
                      save_current_esa_checkpoint, load_current_esa_checkpoint)

def test_current_half_life_defaults_and_old_esa_still_works():
    m = CurrentESA(embd=24, head=6, backend='pytorch', device='cpu')
    assert tuple(DEFAULT_CURRENT_HALF_LIVES) == (.125,.5,2.,8.,32.,64.)
    torch.testing.assert_close(m.log_half_life.detach().exp(),
                               torch.tensor(DEFAULT_CURRENT_HALF_LIVES), rtol=1e-6, atol=1e-6)
    old = ESA(embd=24, head=4, mode='legacy', backend='pytorch', precision='fp32', device=None)
    assert old(torch.randn(1,4,24)).shape == (1,4,24)
    assert 'log_half_life' not in old.state_dict()

def test_current_forward_prefill_decode_and_gradient():
    torch.manual_seed(3401)
    m = CurrentESA(embd=24, head=6, backend='pytorch', device='cpu')
    x = torch.randn(2,5,24,requires_grad=True)
    y, state = m(x)
    assert y.shape == x.shape and state.shape == (2,6,4)
    (y.square().mean()+state.square().mean()).backward()
    assert m.log_half_life.grad is not None
    assert bool(torch.isfinite(m.log_half_life.grad).all())
    m.eval()
    with torch.no_grad():
        full, end = m(torch.cat((x.detach(), torch.randn(2,1,24)),1))
        a, state = m.prefill(x.detach())
        b, state = m.decode_step(torch.randn(2,24), state)
    assert a.shape == (2,5,24) and b.shape == (2,1,24)
    assert state.shape == (2,6,4)

def test_fastesa_checkpoint_roundtrip_and_legacy_rejection():
    m = CurrentESA(embd=24, head=6, backend='pytorch', device='cpu')
    with tempfile.TemporaryDirectory() as td:
        path = Path(td)/'current.pt'
        save_current_esa_checkpoint(m,path)
        loaded = load_current_esa_checkpoint(path, backend='pytorch', device='cpu')
        assert isinstance(loaded, CurrentESA)
        for key,v in m.state_dict().items():
            torch.testing.assert_close(v,loaded.state_dict()[key],rtol=0,atol=0)
        with pytest.raises(ValueError):
            load_current_esa_checkpoint(path, variant='base',backend='pytorch',device='cpu')

def test_current_model_factory_and_public_dispatch():
    m=make_current_esa_model(vocab_size=64, dim=24, heads=6, layers=1,
                             seed=17, scan='reference')
    for block in m.blocks:
        torch.testing.assert_close(block.esa.log_half_life.detach().exp(),
                   torch.tensor(DEFAULT_CURRENT_HALF_LIVES),rtol=1e-6,atol=1e-6)
    logits,states=current_esa_forward_tokens(m, torch.randint(0,64,(1,5)))
    assert logits.shape==(1,64) and len(states)==1
    assert isinstance(make_esa(architecture='current',embd=24,head=6,device='cpu'),CurrentESA)
    assert isinstance(make_esa(architecture='legacy',embd=24,head=4,precision='fp32',device='cpu'),ESA)
    with pytest.raises(ValueError,match='architecture'):
        make_esa(architecture='unknown')
