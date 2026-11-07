"""Opt-in multi-CTA native BOLT split-context regression suite."""
import os
import pytest
import torch
from mlbricks.bolt import native_backend


def test_split_opt_in_off_by_default(monkeypatch):
    monkeypatch.delenv("MLBRICKS_BOLT_SPLIT_CONTEXT_READ", raising=False)
    assert not os.getenv("MLBRICKS_BOLT_SPLIT_CONTEXT_READ")
    assert isinstance(native_backend.split_read_calls(), int)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("width", [16, 32])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("length", [512, 4096, 8192, 32768])
def test_split_read_matches_float32_reference(width, dtype, length):
    ext = native_backend.get_extension()
    assert ext is not None and hasattr(ext, "decode_append_split"), "Build modern native CUDA with split"
    torch.manual_seed(2000+width+length)
    device="cuda:0"
    capacity=length+4
    cache=(torch.randn((1,12,capacity,width),device=device,dtype=dtype)*0.02).contiguous()
    q=(torch.randn((1,12,1,width),device=device,dtype=torch.float32)*0.2).contiguous()
    c_new=(torch.randn((1,12,1,width),device=device,dtype=dtype)*0.02).contiguous()
    scratch=torch.empty((1,12,capacity),device=device,dtype=torch.float32)
    scale=width**(-0.5)
    out=ext.decode_append_split(q,c_new,cache,scratch,length,scale,512)
    torch.cuda.synchronize()
    assert out.dtype==torch.float32 and out.shape==q.shape
    torch.testing.assert_close(cache[:,:,length:length+1,:],c_new,rtol=0,atol=0)
    scores=torch.einsum('bhqd,bhkd->bhqk',q.float(),cache[:,:,:length+1,:].float())*scale
    weights=torch.softmax(scores,dim=-1)
    expected=torch.einsum('bhqk,bhkd->bhqd',weights,cache[:,:,:length+1,:].float())
    torch.testing.assert_close(out,expected,rtol=0.003,atol=0.0003)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("width", [16,32])
def test_split_read_matches_existing_native(width):
    ext = native_backend.get_extension()
    assert ext is not None and hasattr(ext, "decode_append_split")
    torch.manual_seed(7500+width)
    context=8192
    q=torch.randn((1,12,1,width),device='cuda',dtype=torch.float32)*.2
    new=torch.randn((1,12,1,width),device='cuda',dtype=torch.float16)*.03
    src=torch.randn((1,12,context+1,width),device='cuda',dtype=torch.float16)*.03
    src0=src.clone();src1=src.clone()
    scratch0=torch.empty((1,12,context+1),device='cuda',dtype=torch.float32)
    scratch1=torch.empty_like(scratch0)
    scale=width**(-0.5)
    y0=ext.decode_append(q.contiguous(),new.contiguous(),src0,scratch0,context,scale)
    y1=ext.decode_append_split(q.contiguous(),new.contiguous(),src1,scratch1,context,scale,512)
    torch.testing.assert_close(y1,y0,rtol=0.003,atol=0.0003)
    torch.testing.assert_close(src1,src0,rtol=0,atol=0)
