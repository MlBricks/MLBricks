"""CUDA native vs PyTorch BOLT + AVQ-198 smoke, timing and parity check.

Install with MLBRICKS_BUILD_BOLT_MODERN_NATIVE=1 and run on NVIDIA T4/Ampere/Hopper.
CUDA Graph benchmarking is a follow-up once strict parity passes on the GPU.
"""
import copy
import json
import time
import torch
from mlbricks import Bolt
from mlbricks.bolt.native_backend import get_extension

assert torch.cuda.is_available(), 'Requires NVIDIA CUDA GPU'
assert get_extension() is not None, 'Install mlbricks._bolt_modern_cuda first'
print('GPU:', torch.cuda.get_device_name(0))
torch.manual_seed(42)
reference=Bolt(embed_dim=384,num_heads=12,mode='quality',avq_dim=198,cache='normal',backend='pytorch').eval().cuda()
native=copy.deepcopy(reference).set_backend('native').eval()
for m in [reference,native]:
    with torch.no_grad():m._modern.avq.gamma.fill_(0.2)
print('resolved native backend:',native.resolved_backend())
for length in (128,512,1024):
    x=torch.randn(1,length,384,device='cuda')
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
        yr=reference(x);yn=native(x)
        error=(yn.float()-yr.float()).abs().max().item()
        print('Prefill parity',length,'max error',error)
        assert error < 0.06, f'Unexpected prefill divergence {error}'
        for _ in range(3):reference(x);native(x)
        torch.cuda.synchronize()
        data={}
        for label,model in [('pytorch',reference),('native',native)]:
            start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(10):model(x)
            end.record();torch.cuda.synchronize()
            data[label]=start.elapsed_time(end)/10
        print(json.dumps({'sequence':length,'ms_per_prefill':data,'relative_speedup':data['pytorch']/data['native']}))
with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
    x=torch.randn(1,288,384,device='cuda')
    _,cache1=reference.prefill(x[:,:256]);_,cache2=native.prefill(x[:,:256])
    data={}
    for label,model,cache in [('pytorch',reference,cache1),('native',native,cache2)]:
        start=time.perf_counter()
        for t in range(256,288):out,cache=model.decode_step(x[:,t:t+1,:],cache)
        torch.cuda.synchronize()
        data[label]={'tok_s':32/(time.perf_counter()-start),'last':out}
    diff=(data['pytorch']['last'].float()-data['native']['last'].float()).abs().max().item()
    print('Progressive decode parity max error',diff)
    print('Progressive decode tok/s',data['pytorch']['tok_s'],data['native']['tok_s'])
    assert diff < .06
