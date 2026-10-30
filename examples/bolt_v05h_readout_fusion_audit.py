"""MLBricks BOLT v0.5h — reference-rounded CUDA readout validation (experimental).

Built on pinned v0.5c library + v0.5g native-state extension + v0.5h readout.
The original checkpoints and training model remain unchanged. NO release claims.
GPU: strict isolated features, reference teacher-forced full-logit/AVQ/KV,
closed-loop eager+CUDA Graph, randomized end-to-end latency, CUDA event microbench.
"""
from __future__ import annotations
import argparse
import gc
import json
import math
import os
import random
import statistics
import sys
import time
import traceback
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from examples import bolt_v05_production_audit as audit
from examples.bolt_v05e_hybrid_readout_probe import enable_hybrid
from examples.bolt_v05g_state_only import enable_state_only
from examples.bolt_v05h_readout import enable_readout
from mlbricks.bolt import native_v05

CANDIDATES=('readout_pt_gate','readout_raw_gate')
VARIANTS=('pytorch','fully_native','hybrid_avq','recurrence_only')+CANDIDATES

def make_variant(ref,name):
    if name=='pytorch': return audit.configured(ref,False,False)
    if name=='fully_native': return audit.configured(ref,True,True)
    if name=='hybrid_avq': return enable_hybrid(audit.configured(ref,True,True))
    if name=='recurrence_only': return enable_state_only(audit.configured(ref,True,True))
    if name=='readout_pt_gate': return enable_readout(audit.configured(ref,True,True),mode='pt_gate')
    if name=='readout_raw_gate': return enable_readout(audit.configured(ref,True,True),mode='raw_gate')
    raise ValueError(name)

def diff(a,b):
    if a.shape!=b.shape or not bool(torch.isfinite(a).all()) or not bool(torch.isfinite(b).all()):return float('inf')
    return float((a.float()-b.float()).abs().max().item())

@torch.inference_mode()
def isolated(rounds=10):
    """Check FP32 feature bitwise equality using identical states/inputs, no Transformer."""
    rows=[]
    for batch,width,dtype in [(1,198,torch.float16),(4,198,torch.float16),(2,384,torch.float16)]:
        for k in range(rounds):
            torch.manual_seed(2026+batch*31+width+k)
            raw=(torch.randn(batch,1,3*width,device='cuda')* (0.4 if k%2 else 3.0)).to(dtype).contiguous()
            state=(torch.randn(batch,width,device='cuda')* (0.1 if k%3 else 20)).float().contiguous()
            inv=torch.rsqrt(state[:,None,:].square().mean(-1,keepdim=True)+1e-6).contiguous()
            r=torch.tanh(raw[...,2*width:]).float().contiguous()
            ref=(state[:,None,:]*inv)*(1.+r)
            safe=torch.empty_like(ref)
            fast=torch.empty_like(ref)
            native_v05.avq_readout_pt(state,r,inv,safe)
            native_v05.avq_readout_raw(state,raw,inv,fast)
            for name,candidate in [('readout_pt_gate',safe),('readout_raw_gate',fast)]:
                finite=bool(torch.isfinite(candidate).all())
                rows.append(dict(variant=name,batch=batch,width=width,case=k,
                                 passed=finite and torch.equal(ref,candidate),
                                 max_feature_delta=diff(ref,candidate),
                                 mismatched_features=int((ref != candidate).sum().item()) if finite else -1))
        print('ISOLATED',batch,width, 'complete',flush=True)
    return rows

@torch.inference_mode()
def validation(cases,only=VARIANTS):
    rows=[]
    for idx,cfg in enumerate(cases,1):
        torch.manual_seed(cfg['seed']+31)
        prompt=torch.randint(0,2048,(cfg['batch'],cfg['prompt']),device='cuda')
        ref=audit.make_model(cfg)
        baseline=audit.eager_trace(ref,prompt,cfg['tokens'])
        print('\nNUMERICAL CASE',idx,cfg,flush=True)
        for name in only:
            for mode in ('eager','graph'):
                row=dict(case=idx,config=cfg,variant=name,mode=mode,passed=False)
                m=None
                try:
                    m=make_variant(ref,name)
                    trial=audit.graph_trace(m,prompt,cfg['tokens']) if mode=='graph' else audit.eager_trace(m,prompt,cfg['tokens'])
                    row.update(audit.compare(baseline,trial,mode=mode))
                    if any(not torch.isfinite(t).all().item() for t in trial['logits']):
                        row['passed']=False;row['error']='nonfinite logits'
                    if trial.get('_session') is not None:
                        del trial['_session'],trial['_captured_logits']
                    del trial
                except Exception as e:
                    row['error']=repr(e);row['traceback']=traceback.format_exc(limit=6)
                print(name,mode,'PASS' if row['passed'] else 'FAIL',
                      'logits',row.get('max_logit_delta'),'states',row.get('max_avq_state_delta'),
                      'cache',row.get('max_kv_delta'),row.get('error',''),flush=True)
                rows.append(row)
                del m
        del baseline,ref,prompt
        gc.collect();torch.cuda.empty_cache()
    return rows

@torch.inference_mode()
def closed_loop(cases):
    result=[]
    for cfg in cases:
        ref=audit.make_model(cfg)
        for v in ('hybrid_avq',)+CANDIDATES:
            m=make_variant(ref,v)
            for iteration in range(3):
                torch.manual_seed(cfg['seed']+1200+iteration)
                prompt=torch.randint(0,2048,(cfg['batch'],cfg['prompt']),device='cuda')
                row=dict(config=cfg,variant=v,iteration=iteration,passed=False)
                try:
                    base=ref.generate(prompt,cfg['tokens'],temperature=0)
                    eager=m.generate(prompt,cfg['tokens'],temperature=0)
                    session=m.prepare_cuda_graph_generation(prompt,cfg['tokens'])
                    graph=session.generate().clone()
                    row.update(passed=bool(torch.equal(base,eager) and torch.equal(base,graph) and audit.finite_caches(session.caches)),
                               eager_mismatch=int((base!=eager).sum().item()),
                               graph_mismatch=int((base!=graph).sum().item()))
                    del session
                except Exception as e:row['error']=repr(e)
                print('CLOSED LOOP',v,iteration,row['passed'],row.get('error',''),flush=True)
                result.append(row)
            del m
        del ref
        gc.collect();torch.cuda.empty_cache()
    return result

@torch.inference_mode()
def microbench(iterations=200,rep=7):
    """Kernel-level CUDA-event end-to-end readout; no recurrence or linear projection.
    Input allocations performed outside timing. Measures launch overhead + PyTorch
    tanh/RMS reduction + candidate kernel; avoids 'CUDA kernel only' false comparability.
    """
    output=[]
    for batch in (1,4):
        width=198
        torch.manual_seed(101+batch)
        raw=torch.randn(batch,1,3*width,device='cuda',dtype=torch.float16).contiguous()
        st=torch.randn(batch,width,device='cuda').contiguous()
        out=torch.empty(batch,1,width,device='cuda',dtype=torch.float32)
        def inv():return torch.rsqrt(st[:,None,:].square().mean(-1,keepdim=True)+1e-6)
        def reference():
            return (st[:,None,:]*inv())*(1.+torch.tanh(raw[...,2*width:]).float())
        def safe():
            native_v05.avq_readout_pt(st,torch.tanh(raw[...,2*width:]).float().contiguous(),inv().contiguous(),out)
        def aggressive():
            native_v05.avq_readout_raw(st,raw,inv().contiguous(),out)
        acts={'pytorch_readout':reference,'readout_pt_gate':safe,'readout_raw_gate':aggressive}
        for label,fn in acts.items():
            for _ in range(20):fn()
            torch.cuda.synchronize()
            samples=[]
            for _ in range(rep):
                a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
                a.record()
                for _ in range(iterations):fn()
                b.record();b.synchronize()
                samples.append(1000*a.elapsed_time(b)/iterations)
            output.append(dict(batch=batch,width=width,variant=label,
                               median_us=float(statistics.median(samples)),samples_us=samples))
            print('MICRO',batch,label,round(statistics.median(samples),2),'us',flush=True)
    return output

def e2e_bench(case,eligible,repeats=9):
    """Randomize variant measurement order to reduce drift bias. Each request
    includes prefill+graph capture+decode; excludes model load/JIT compilation.
    """
    torch.manual_seed(case['seed']+31)
    prompt=torch.randint(0,2048,(case['batch'],case['prompt']),device='cuda')
    ref=audit.make_model(case)
    models={v:make_variant(ref,v) for v in eligible}
    def measure(v):
        m=models[v]
        with torch.inference_mode():
            session=m.prepare_cuda_graph_generation(prompt,case['tokens'])
            result=session.generate()
        return result
    samples={v:[] for v in eligible}
    for v in eligible:
        measure(v);torch.cuda.synchronize()
    rng=random.Random(2026)
    for i in range(repeats):
        order=list(eligible);rng.shuffle(order)
        for v in order:
            torch.cuda.synchronize()
            start=time.perf_counter()
            measure(v)
            torch.cuda.synchronize()
            samples[v].append(1000*(time.perf_counter()-start))
        print('E2E repeat',i+1,'/',repeats,flush=True)
    rows=[]
    for v in eligible:
        s=sorted(samples[v]);p95=s[max(0,math.ceil(0.95*len(s))-1)]
        rows.append(dict(variant=v,mode='graph',repeats=repeats,
                         p50_ms=float(statistics.median(s)),p95_ms=float(p95),samples_ms=s))
        print('E2E',v,'p50',round(statistics.median(s),3),'p95',round(p95,3),flush=True)
    return rows

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--level',choices=['smoke','full'],default='full')
    p.add_argument('--repeats',type=int,default=9)
    p.add_argument('--output',default='/kaggle/working/bolt_v05h_readout_results.json')
    a=p.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    if os.getenv('MLBRICKS_BOLT_V05_ALLOW_JIT')!='1':raise RuntimeError('Set MLBRICKS_BOLT_V05_ALLOW_JIT=1')
    torch.backends.cuda.matmul.allow_tf32=False
    print('BOLT v0.5h',torch.cuda.get_device_name(0),torch.__version__,flush=True)
    report={'version':'BOLT v0.5h fused reference-rounded readout research',
            'device':torch.cuda.get_device_name(0),'torch':torch.__version__,
            'config':vars(a),'isolated':[],'numerical':[],'closed_loop':[],
            'microbench':[],'e2e':[],'verdict':'NOT_EVALUATED',
            'pending':['trained-checkpoint quality','RoPE','BF16/FP32 full model',
                       'other CUDA architectures','long-duration concurrency/stress']}
    report['isolated']=isolated(rounds=5 if a.level=='smoke' else 10)
    cases=audit.scenarios(a.level)
    report['numerical']=validation(cases)
    report['closed_loop']=closed_loop([cases[0],cases[-1]])
    report['microbench']=microbench(iterations=150 if a.level=='smoke' else 250,rep=5 if a.level=='smoke' else 9)
    gates={}
    for v in CANDIDATES:
        micro=[x for x in report['isolated'] if x['variant']==v]
        num=[x for x in report['numerical'] if x['variant']==v]
        loop=[x for x in report['closed_loop'] if x['variant']==v]
        gates[v]=dict(isolated_bitwise=bool(micro) and all(x['passed'] for x in micro),
                      numerical=bool(num) and all(x['passed'] for x in num),
                      closed_loop=bool(loop) and all(x['passed'] for x in loop))
        gates[v]['eligible_for_benchmark']=all(gates[v].values())
    eligible=['pytorch','hybrid_avq']+[v for v in CANDIDATES if gates[v]['eligible_for_benchmark']]
    if a.repeats>=3:
        report['e2e']=e2e_bench(dict(cache='tight',batch=1,prompt=512,tokens=64,seed=2026),eligible,repeats=a.repeats)
    baseline=next((x['p50_ms'] for x in report['e2e'] if x['variant']=='hybrid_avq'),None)
    for v in CANDIDATES:
        candidate=next((x['p50_ms'] for x in report['e2e'] if x['variant']==v),None)
        gates[v]['p50_ms']=candidate
        gates[v]['relative_speedup_vs_hybrid']=(baseline/candidate if baseline and candidate else None)
        gates[v]['faster_than_hybrid_in_sample']=bool(baseline and candidate and candidate<baseline)
    report['gates']=gates
    passed=[v for v in CANDIDATES if gates[v]['eligible_for_benchmark'] and gates[v]['faster_than_hybrid_in_sample']]
    report['verdict']=('EXPERIMENTAL_SPEEDUP_WITH_PARITY__MORE_RELEASE_GATES_REQUIRED' if passed else
                      'NO_PROVEN_SPEEDUP__RETAIN_HYBRID_EXPERIMENTAL_BACKEND')
    path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2,default=str))
    print('\n=== BOLT v0.5h FINAL ===')
    print('Readout gates:',gates)
    print('Verdict:',report['verdict'])
    print('REPORT',path)
    return 0 if any(g['eligible_for_benchmark'] for g in gates.values()) else 2

if __name__=='__main__':raise SystemExit(main())
