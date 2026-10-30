"""MLBricks BOLT v0.5j: stock-exact arithmetic + graph-compatible routing.

Run from extracted pinned MLBricks-main repository, no Triton/custom CUDA.
The test is diagnostic and does NOT modify the production MLBricks backend.
"""
from __future__ import annotations
import argparse, copy, csv, json, math, os, random, statistics, sys, time, traceback, types
from contextlib import nullcontext
from pathlib import Path
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from mlbricks import BoltModel
from examples import bolt_v05_production_audit as audit


def sync():
    if torch.cuda.is_available(): torch.cuda.synchronize()


def amp(device):
    return torch.autocast('cuda', dtype=torch.float16) if device.type=='cuda' else nullcontext()


def make_stock(cache, seed, prompt, tokens, batch, device):
    torch.manual_seed(seed)
    cuda = device.type=='cuda'
    m = BoltModel(vocab_size=2048, layers=6 if cuda else 2,
                  dim=384 if cuda else 96, heads=12 if cuda else 6,
                  context=prompt+tokens+8, bolt_mode='quality',bolt_cache=cache,
                  avq_dim=198,position='none',dropout=0.,backend='pytorch')
    m.eval().to(device=device,dtype=torch.float16 if cuda else torch.float32)
    with torch.no_grad():
        for block in m.blocks:
            a=block.attn._modern
            a.p.uniform_(.7,1.3)
            a.avq.gamma.fill_(.21)
            a.avq.norm.weight.uniform_(.7,1.3)
    return m


def strict_stock_arithmetic(model):
    """Enable graph-compatible routing, but preserve stock floating-point operation order.

    Specifically disable all three transformations found responsible by v0.5i:
    (1) packed Q/U/G GEMM, (2) weight-folded Q/P, (3) folded AVQ output.
    The graph generator requires optimized_inference=True, hence we install a
    correct _inference_projection shim that returns the already P-scaled Q.
    Non-production, experimental diagnostic: do not save this monkeypatched
    model for later use as a serialized object.
    """
    model.enable_bolt_optimized_inference(True)
    for block in model.blocks:
        a=block.attn._modern
        a.avq.fold_output_inference=False
        def projection(self,x):
            q = F.linear(x,self.q_proj.weight,self.q_proj.bias) * self.p.reshape(-1)
            return q, self.c_proj(x), self.g_proj(x), None
        a._inference_projection=types.MethodType(projection,a)
        a.v05_native_gate=False
        a.avq.v05_native_avq=False
    return model


def finite_report(tr):
    return bool(audit.finite_caches(tr['caches']))


def run_case(cache,seed,prompt_len,tokens,batch,device,repeats):
    stock=make_stock(cache,seed,prompt_len,tokens,batch,device)
    strict_eager=strict_stock_arithmetic(copy.deepcopy(stock))
    strict_graph=strict_stock_arithmetic(copy.deepcopy(stock)) if device.type=='cuda' else None
    packed=copy.deepcopy(stock).enable_bolt_optimized_inference(True)
    models={'stock_eager':stock,'strict_eager':strict_eager,'builtin_eager':packed}
    if strict_graph is not None:
        models['strict_graph']=strict_graph
        models['builtin_graph']=copy.deepcopy(stock).enable_bolt_optimized_inference(True)
    # Identical weights across all variants and independent storage.
    for name,m in models.items():
        for a,b in zip(stock.parameters(),m.parameters()):
            if not torch.equal(a,b): raise RuntimeError(f'{name} parameter mismatch')
            if name!='stock_eager' and a.data_ptr()==b.data_ptr():
                raise RuntimeError(f'{name} shares parameter storage')
    torch.manual_seed(seed+193)
    ids=torch.randint(0,2048,(batch,prompt_len),device=device)
    reference=audit.eager_trace(stock,ids,tokens)
    results={}
    for label,m in models.items():
        tr=None
        graph=label.endswith('graph')
        try:
            tr=audit.graph_trace(m,ids,tokens) if graph else audit.eager_trace(m,ids,tokens)
            row=audit.compare(reference,tr,mode='graph' if graph else 'eager')
            row['finite_cache']=finite_report(tr)
            row['passed']=bool(row['passed'] and row['finite_cache'])
            row['full_logits_exact']=bool(row['max_logit_delta']==0)
            row['states_exact']=bool(row['max_avq_state_delta']==0 and row['final_state_delta']==0)
            row['kv_exact']=bool(row['max_kv_delta']==0)
            results[label]=row
        except Exception as e:
            results[label]={'passed':False,'error':repr(e),'traceback':traceback.format_exc(limit=5)}
        finally:
            if tr is not None:
                tr.pop('_session',None)
                tr.pop('_captured_logits',None)
                del tr
        print(cache,seed,label,results[label].get('passed'),
              'logit',results[label].get('max_logit_delta'),
              'state',results[label].get('max_avq_state_delta'),
              'KV',results[label].get('max_kv_delta'),
              'first',results[label].get('first_failure'),flush=True)
    # Check full public generation API; independent of the trace machinery.
    outputs={}
    if device.type=='cuda':
        for label,m in models.items():
            try:
                if label.endswith('graph'):
                    sess=m.prepare_cuda_graph_generation(ids,tokens)
                    generated=sess.generate().clone()
                    del sess
                else:
                    generated=m.generate(ids,tokens,temperature=0).clone()
                outputs[label]=generated
            except Exception as e:
                results[label]['public_generation_error']=repr(e)
        for label,output in outputs.items():
            results[label]['public_token_matches_stock']=(
                bool(torch.equal(output,outputs['stock_eager'])) if 'stock_eager' in outputs else False)
    # Randomized end-to-end request timing; includes prefill and capture.
    def request(name,m):
        if name.endswith('graph'):
            session=m.prepare_cuda_graph_generation(ids,tokens)
            out=session.generate()
            del session
            return out
        return m.generate(ids,tokens,temperature=0)
    timings={name:[] for name in models}
    if device.type=='cuda':
        for name,m in models.items():
            try:request(name,m)
            except Exception as e:results[name]['timing_error']=repr(e)
        sync()
        rng=random.Random(seed+77)
        order=list(models.items())
        for round_idx in range(repeats):
            rng.shuffle(order)
            for name,m in order:
                if 'timing_error' in results[name]:continue
                try:
                    sync()
                    start=time.perf_counter()
                    result=request(name,m)
                    sync()
                    timings[name].append((time.perf_counter()-start)*1000)
                    del result
                except Exception as e:results[name]['timing_error']=repr(e)
            print('TIMING ROUND',round_idx+1,'/',repeats,flush=True)
    else:
        for name,m in models.items():
            sync()
            start=time.perf_counter()
            request(name,m)
            timings[name].append((time.perf_counter()-start)*1000)
    for name,ms in timings.items():
        if ms:
            results[name]['latency']={
                'p50_ms':round(statistics.median(ms),3),
                'samples_ms':[round(x,3) for x in ms],
                'samples':len(ms)
            }
    return {'cache':cache,'seed':seed,'batch':batch,'prompt':prompt_len,
            'tokens':tokens,'results':results}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',default='/kaggle/working/bolt_v05j_strict_stock_parity.json')
    p.add_argument('--cache',choices=['tight','normal','both'],default='both')
    p.add_argument('--prompt',type=int,default=512)
    p.add_argument('--tokens',type=int,default=64)
    p.add_argument('--batch',type=int,default=1)
    p.add_argument('--repeats',type=int,default=7)
    p.add_argument('--seed',type=int,default=2026)
    args=p.parse_args()
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type=='cpu':
        args.prompt=min(args.prompt,24)
        args.tokens=min(args.tokens,5)
        args.repeats=1
        torch.set_num_threads(4)
    print('TORCH',torch.__version__,'CUDA',torch.version.cuda,'GPU',torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU',flush=True)
    caches=['tight','normal'] if args.cache=='both' else [args.cache]
    rows=[]
    with amp(device),torch.inference_mode():
        for cache in caches:
            rows.append(run_case(cache,args.seed if cache=='tight' else 5505,
                                 args.prompt,args.tokens,args.batch,device,args.repeats))
    path=Path(args.output)
    path.parent.mkdir(parents=True,exist_ok=True)
    report={'torch':torch.__version__,'device':str(device),'config':vars(args),
            'cases':rows,'note':'Stock-faithful arithmetic path; no custom CUDA kernel.'}
    path.write_text(json.dumps(report,indent=2))
    print('\n=== STRICT STOCK MATH SUMMARY ===')
    for case in rows:
        r=case['results']
        for variant in r:
            q=r[variant]
            print(case['cache'],variant,'PASS',q.get('passed'),
                  'exact',q.get('full_logits_exact'),
                  'p50',q.get('latency',{}).get('p50_ms'),
                  'ERROR',q.get('error'),flush=True)
    print('REPORT',path,flush=True)
    if not all(c['results'].get('strict_eager',{}).get('passed') for c in rows):
        raise SystemExit('Strict eager arithmetic unexpectedly failed parity')
    if device.type=='cuda' and not all(c['results'].get('strict_graph',{}).get('passed') for c in rows):
        raise SystemExit('Strict graph arithmetic failed parity; investigate GPU trace')

if __name__=='__main__':
    main()
