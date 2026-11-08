"""CPU-only regression tests. GPU native kernels must still be qualified on T4."""
from __future__ import annotations
from unittest.mock import patch
import torch
from mlbricks.bolt.presets import BoltPreset, BoltCache
from mlbricks.bolt import native_backend


def test_nested_wrapper_preserves_external_autocast_cpu():
    m=BoltPreset(64,4,mode='fast',cache='tight',backend='pytorch').eval()
    x=torch.randn(1,1,64)
    c=BoltCache.allocate(1,4,3,16,0,device='cpu',dtype=torch.bfloat16)
    seen=[]
    original=m._project
    def observing_project(x,*,start_pos=0):
        seen.append(torch.is_autocast_enabled('cpu'))
        return original(x,start_pos=start_pos)
    m._project=observing_project
    with torch.no_grad(),torch.autocast('cpu',dtype=torch.bfloat16):
        y=m.decode_with_cache(x,c,position=0)
    assert seen==[True],f'External autocast was disabled by a nested wrapper: {seen}'
    assert y.isfinite().all() and c.length==1


def test_native_current_matches_half_cache_query_stays_fp32():
    m=BoltPreset(64,4,mode='fast',cache='tight',backend='pytorch').eval()
    x=torch.randn(1,1,64)
    c=BoltCache.allocate(1,4,3,16,0,device='cpu',dtype=torch.float16)
    q=torch.randn(1,4,1,16,dtype=torch.float32)
    current=torch.randn(1,4,1,16,dtype=torch.float32)
    seen=[]
    def fake_append(q,c_now,history,scratch,position,scale):
        seen.append((q.dtype,c_now.dtype,history.dtype))
        history[:,:,position:position+1,:].copy_(c_now)
        return torch.zeros_like(q)
    m._native_enabled=lambda x: True
    m._project=lambda x, start_pos=0:(q,current,current)
    with patch.object(native_backend,'get_extension',return_value=object()),\
         patch.object(native_backend,'decode_append',side_effect=fake_append):
        with torch.no_grad():y=m.decode_with_cache(x,c,position=0)
    assert seen==[(torch.float32,torch.float16,torch.float16)], seen
    assert c.length==1 and y.isfinite().all()
    assert torch.equal(c.c[:,:,:1,:], current.to(torch.float16))


def test_fp16_weights_select_internal_autocast_cpu():
    m=BoltPreset(64,4,mode='fast',cache='tight',backend='pytorch').half().eval()
    x=torch.randn(1,1,64)
    c=BoltCache.allocate(1,4,3,16,0,device='cpu',dtype=torch.float16)
    seen=[]
    original=m._project
    def observing_project(x,*,start_pos=0):
        seen.append(torch.is_autocast_enabled('cpu'))
        return original(x,start_pos=start_pos)
    m._project=observing_project
    with torch.no_grad():
        y=m.decode_with_cache(x,c,position=0)
    assert seen==[True],seen
    assert y.isfinite().all()
