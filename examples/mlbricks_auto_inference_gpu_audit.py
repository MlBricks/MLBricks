"""T4 numerical and latency acceptance audit for automatic BOLT and QKV inference.

Uses independent models with identical weights. Reports logits and cache/state
max-differences rather than relying on generated-token equality alone.
"""
from __future__ import annotations
import argparse
import copy
import json
import statistics
import time
from pathlib import Path

import torch
from mlbricks import BoltModel, AttentionModel
from mlbricks.attention_model import QKVCudaGraphGenerator
from examples.bolt_v05_production_audit import (
    eager_trace as bolt_eager_trace,
    graph_trace as bolt_graph_trace,
    compare as bolt_compare,
)


def qkv_eager_trace(m, prompt, n):
    capacity=prompt.shape[1]+n
    caches=m._allocate_cache(prompt.shape[0],capacity)
    logits=m._prefill_cache(prompt,caches)
    token=logits.argmax(-1)
    tokens=[token.clone()]
    positions=torch.arange(capacity,device=prompt.device)
    position=torch.full((),prompt.size(1),device=prompt.device,dtype=torch.long)
    trace=[]
    for _ in range(n-1):
        logits=m._decode_cache(token,caches,position,positions)
        trace.append(logits.clone())
        token=logits.argmax(-1)
        tokens.append(token.clone())
        position.add_(1)
    return {'tokens':torch.cat([prompt,torch.cat(tokens,dim=1)],dim=1),
            'logits':trace,'cache':[(k.clone(),v.clone()) for k,v in caches]}


def qkv_graph_trace(m,prompt,n):
    session=QKVCudaGraphGenerator(m,prompt,n)
    trace=[]
    for _ in range(n-1):
        session.graph.replay()
        trace.append(session.logits.clone())
    return {'tokens':session.tokens.clone(), 'logits':trace,
            'cache':[(k.clone(),v.clone()) for k,v in session.caches]}


def max_abs(x,y):
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        return float('inf')
    return float((x.float()-y.float()).abs().max().item())


def qkv_compare(a,b):
    logit=max([max_abs(x,y) for x,y in zip(a['logits'],b['logits'])] or [0.])
    kv=max([max(max_abs(ak,bk),max_abs(av,bv)) for (ak,av),(bk,bv)
            in zip(a['cache'],b['cache'])] or [0.])
    mismatches=int((a['tokens']!=b['tokens']).sum().item())
    return {'passed':bool(mismatches==0 and logit==0. and kv==0.),
            'max_logit_delta':logit,'max_kv_delta':kv,'token_mismatches':mismatches}


def timed_request(model,prompt,n, *, trials=3):
    times=[]
    for _ in range(trials):
        torch.cuda.synchronize()
        start=time.perf_counter()
        _=model.generate(prompt,n,temperature=0).clone()
        torch.cuda.synchronize()
        times.append((time.perf_counter()-start)*1000)
    return {'median_request_ms':round(statistics.median(times),4),
            'samples_ms':[round(t,4) for t in times]}


def audit_one(architecture, cache, dtype, *, layers=6, dim=384, heads=12,
              prompt_length=512, generated_tokens=64, repeats=3):
    torch.manual_seed(2026 if cache=='tight' else 5505)
    if architecture=='bolt':
        model=BoltModel(vocab_size=2048,dim=dim,heads=heads,layers=layers,
                        context=prompt_length+generated_tokens+8,
                        bolt_mode='quality',bolt_cache=cache,avq_dim=198,
                        position='none',dropout=0,backend='pytorch')
        with torch.no_grad():
            for block in model.blocks:
                att=block.attn._modern
                att.p.uniform_(.7,1.3)
                att.avq.gamma.fill_(.21)
                att.avq.norm.weight.uniform_(.7,1.3)
    else:
        model=AttentionModel(vocab_size=2048,dim=dim,heads=heads,layers=layers,
                             context=prompt_length+generated_tokens+8)
    model.to(device='cuda',dtype=dtype).eval()
    eager=copy.deepcopy(model).eval()
    graph_model=copy.deepcopy(model).eval()
    prompt=torch.randint(0,2048,(1,prompt_length),device='cuda')
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16,enabled=dtype==torch.float16):
        if architecture=='bolt':
            base=bolt_eager_trace(eager,prompt,generated_tokens)
            if dtype==torch.float16:
                graph_model.enable_bolt_strict_inference(True)
                gr=bolt_graph_trace(graph_model,prompt,generated_tokens)
                r=bolt_compare(base,gr,mode='graph')
                r['full_exact']=bool(r['max_logit_delta']==0 and
                                     r['max_avq_state_delta']==0 and
                                     r['max_kv_delta']==0 and r['final_state_delta']==0 and
                                     r['token_mismatches']==0)
            else:
                r={'full_exact':None,'note':'FP32 graph unvalidated; use eager fallback'}
        else:
            base=qkv_eager_trace(eager,prompt,generated_tokens)
            if dtype==torch.float16:
                gr=qkv_graph_trace(graph_model,prompt,generated_tokens)
                r=qkv_compare(base,gr)
                r['full_exact']=r['passed']
            else:
                r={'full_exact':None,'note':'FP32 graph unvalidated; use eager fallback'}
        # Public path: no optimization flag is set by the caller.
        result=model.generate(prompt,generated_tokens,temperature=0)
        r['automatic_dispatch']=model.inference_info()['last_execution']
        r['public_token_parity']=bool(torch.equal(result,base['tokens']))
        r['dtype_preserved']=next(model.parameters()).dtype==dtype
        if dtype==torch.float16:
            r['passed']=bool(r['full_exact'] and r['public_token_parity']
                             and r['automatic_dispatch']=='strict_cuda_graph'
                             and r['dtype_preserved'])
        else:
            # FP32 graph is not an approved automatic candidate at this stage.
            r['passed']=bool(r['public_token_parity'] and
                             r['automatic_dispatch']=='pytorch_eager' and
                             r['dtype_preserved'])
    if dtype==torch.float16 and r['passed']:
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
            r['latency']=timed_request(model,prompt,generated_tokens,trials=repeats)
    return r


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='/kaggle/working/mlbricks_auto_inference_audit.json')
    parser.add_argument('--quick',action='store_true')
    args=parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit('This audit needs a CUDA GPU')
    print('Torch',torch.__version__,'GPU',torch.cuda.get_device_name(0),flush=True)
    quick=args.quick
    scenarios=[]
    for arch in ('bolt','qkv'):
        caches=('tight','normal') if arch=='bolt' else ('not-applicable',)
        for cache in caches:
            for dtype in (torch.float16,torch.float32):
                scenarios.append((arch,cache,dtype))
    rows=[]
    for arch,cache,dtype in scenarios:
        print('RUNNING',arch,cache,dtype,flush=True)
        try:
            row=audit_one(arch,cache,dtype,layers=2 if quick else 6,
                          dim=96 if quick else 384,heads=6 if quick else 12,
                          prompt_length=24 if quick else 512,
                          generated_tokens=8 if quick else 64,
                          repeats=1 if quick else 3)
        except Exception as e:
            import traceback
            row={'passed':False,'error':str(e),'traceback':traceback.format_exc(limit=8)}
        row.update({'architecture':arch,'cache':cache,'dtype':str(dtype)})
        print('RESULT',json.dumps({k:v for k,v in row.items() if k!='latency'}),flush=True)
        rows.append(row)
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    doc={'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0),
         'rows':rows,'passed':all(r.get('passed',False) for r in rows)}
    path.write_text(json.dumps(doc,indent=2))
    print('REPORT',path,'ALL_PASS',doc['passed'],flush=True)
    if not doc['passed']:
        raise SystemExit(1)

if __name__=='__main__':
    main()
