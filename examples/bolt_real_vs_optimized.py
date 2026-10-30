"""Real MLBricks BOLT vs v0.5h benchmark (native CUDA; no Triton).

Comparison uses *identical model parameters* on independent model instances:
1. MLBricks stock PyTorch BOLT, optimization disabled.
2. MLBricks PyTorch inference optimized path (Q/P folding, packed Q/U/G).
3. MLBricks v0.5h aggressive readout (opt-in native U/G, AVQ recurrence,
   PyTorch RMS inverse + native reference-rounded readout).
For CUDA Graph, compare (2) and (3), because (1) does not support graphs.

Latency profiles: whole request (prefill+capture+decode), decode-only, and prefill-only.
Numerical checks include generated tokens, full decode logits, AVQ state and active KV.
Randomly initialized model; no trained-checkpoint quality conclusions.
"""
from __future__ import annotations

import argparse
import copy
import csv
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from mlbricks import BoltModel
from examples import bolt_v05_production_audit as audit
from examples.bolt_v05h_readout import enable_readout

VARIANTS = (
    'stock_eager', 'builtin_opt_eager', 'v05h_eager',
    'builtin_opt_graph', 'v05h_graph',
)

PROFILES = {
    'quick': [
        dict(cache='tight', batch=1, prompt=512, tokens=64, seed=2026),
        dict(cache='normal', batch=1, prompt=512, tokens=64, seed=5505),
    ],
    'full': [
        dict(cache='tight', batch=1, prompt=512, tokens=64, seed=2026),
        dict(cache='normal', batch=1, prompt=512, tokens=64, seed=5505),
        dict(cache='tight', batch=4, prompt=128, tokens=32, seed=2026),
        dict(cache='normal', batch=1, prompt=2048, tokens=32, seed=5505),
    ],
}


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def make_models(cfg, device):
    """Create independent storage, same trained *or* random weights; no parameter sharing."""
    torch.manual_seed(cfg['seed'])
    cuda = device.type == 'cuda'
    dtype = torch.float16 if cuda else torch.float32
    common = dict(vocab_size=2048, layers=6 if cuda else 2,
                  dim=384 if cuda else 96, heads=12 if cuda else 6,
                  context=cfg['prompt']+cfg['tokens']+8,
                  bolt_mode='quality', bolt_cache=cfg['cache'], avq_dim=198,
                  position='none', dropout=0, backend='pytorch')
    stock = BoltModel(**common).eval().to(device=device, dtype=dtype)
    with torch.no_grad():
        for block in stock.blocks:
            a = block.attn._modern
            a.p.uniform_(0.7, 1.3)
            a.avq.gamma.fill_(0.21)
            a.avq.norm.weight.uniform_(0.7, 1.3)
    builtin = copy.deepcopy(stock).eval()
    builtin.enable_bolt_optimized_inference(True)
    models = {'stock': stock, 'builtin': builtin}
    if cuda:
        builtin_graph = copy.deepcopy(stock).eval()
        builtin_graph.enable_bolt_optimized_inference(True)
        models['builtin_graph'] = builtin_graph
        for key in ('v05h', 'v05h_graph'):
            fast = copy.deepcopy(stock).eval()
            fast.enable_bolt_optimized_inference(True)
            fast.enable_bolt_v05_native(gate=True, avq=True)
            fast.prepare_bolt_v05_native()  # JIT outside capture/timing
            enable_readout(fast, mode='raw_gate')
            models[key] = fast
    assert models['stock'].blocks[0].attn._modern.latent_dim == (16 if cfg['cache']=='tight' else 32)
    # Check independent parameter storage and checkpoint equivalence.
    basenames = {k: tuple(v.shape) for k, v in stock.state_dict().items()}
    for name, m in models.items():
        assert {k: tuple(v.shape) for k, v in m.state_dict().items()} == basenames, f'{name}: changed checkpoint keys'
        for p, q in zip(stock.parameters(), m.parameters()):
            if name != 'stock':
                assert p.data_ptr() != q.data_ptr(), f'{name}: shared parameter storage'
            assert torch.equal(p, q), f'{name}: weights differ'
    return models


@torch.inference_mode()
def closed_loop(models, prompt, tokens, device):
    """Validate all real public generation APIs, not a hand-written surrogate."""
    stock = models['stock'].generate(prompt, tokens, temperature=0).clone()
    out = {}
    for name in ('stock','builtin') + (('v05h',) if device.type == 'cuda' else ()):
        try:
            eager = models[name].generate(prompt, tokens, temperature=0).clone()
            out[name+'_eager'] = {
                'passed': bool(torch.equal(stock, eager)),
                'token_mismatches':int((stock!=eager).sum().item())}
        except Exception as exc:
            out[name+'_eager']={'passed':False,'error':repr(exc)}
    if device.type == 'cuda':
        for name in ('builtin_graph','v05h_graph'):
            try:
                session=models[name].prepare_cuda_graph_generation(prompt,tokens)
                result=session.generate().clone()
                out[name]={'passed':bool(torch.equal(stock,result) and audit.finite_caches(session.caches)),
                           'token_mismatches':int((stock!=result).sum().item()),
                           'finite_graph_cache':bool(audit.finite_caches(session.caches))}
                del session
            except Exception as exc:
                out[name]={'passed':False,'error':repr(exc)}
    return out


def numerics(models, prompt, tokens, device):
    """Full logits and per-layer AVQ/KV parity, **not** tokens alone."""
    base = audit.eager_trace(models['stock'],prompt,tokens)
    checks = {}
    options=[('stock_eager','stock',False),('builtin_opt_eager','builtin',False)]
    if device.type=='cuda':
        options += [('v05h_eager','v05h',False),('builtin_opt_graph','builtin_graph',True),('v05h_graph','v05h_graph',True)]
    for label,key,graph in options:
        trial=None
        try:
            trial = audit.graph_trace(models[key], prompt, tokens) if graph else audit.eager_trace(models[key],prompt,tokens)
            row = audit.compare(base,trial,mode='graph' if graph else 'eager')
            row['finite_state_kv'] = bool(audit.finite_caches(trial['caches']))
            row['passed'] = bool(row['passed'] and row['finite_state_kv'])
            checks[label]=row
        except Exception as exc:
            checks[label]={'passed':False,'error':repr(exc),'traceback':traceback.format_exc(limit=5)}
        finally:
            if trial is not None:
                trial.pop('_session',None)
                trial.pop('_captured_logits',None)
                del trial
    del base
    return checks


def public_call(m,prompt,n,graph):
    if graph:
        session=m.prepare_cuda_graph_generation(prompt,n)
        result=session.generate()
        return result  # session is released when this returns
    return m.generate(prompt,n,temperature=0)


def samples_ms(fn,repeats):
    times=[]; peaks=[]
    for _ in range(repeats):
        sync()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            before=torch.cuda.memory_allocated()
        else:
            before=0
        t0=time.perf_counter()
        result=fn()
        sync()
        elapsed=(time.perf_counter()-t0)*1000
        if torch.cuda.is_available():
            peak=max(0,torch.cuda.max_memory_allocated()-before)/2**20
        else:peak=0
        times.append(float(elapsed))
        peaks.append(float(peak))
        del result
    return times,peaks


def summarize_ms(times,peak=None):
    ordered=sorted(times)
    return dict(n=len(times),p50_ms=round(statistics.median(times),4),
                observed_p95_ms=round(ordered[max(0,math.ceil(.95*len(ordered))-1)],4),
                mean_ms=round(statistics.mean(times),4),
                min_ms=round(min(times),4),max_ms=round(max(times),4),
                extra_peak_mib=round(max(peak) if peak else 0,3),
                samples_ms=[round(x,4) for x in times])


@torch.inference_mode()
def request_latency(models,prompt,n,repeats,device):
    selected=[('stock_eager','stock',False),('builtin_opt_eager','builtin',False)]
    if device.type=='cuda':
        selected += [('v05h_eager','v05h',False),('builtin_opt_graph','builtin_graph',True),('v05h_graph','v05h_graph',True)]
    for label,key,graph in selected:
        public_call(models[key],prompt,n,graph)
    sync()  # compilation, kernel warmup excluded; capture is INCLUDED in each request
    measurements={label:[] for label,_,_ in selected}
    memory={label:[] for label,_,_ in selected}
    rng=random.Random(9001)
    for i in range(repeats):
        random_order=list(selected)
        rng.shuffle(random_order)
        for label,key,graph in random_order:
            elapsed,peaks=samples_ms(lambda m=models[key],g=graph:public_call(m,prompt,n,g),1)
            measurements[label] += elapsed
            memory[label] += peaks
        print(f'  Request timing round {i+1}/{repeats}',flush=True)
    rows={}
    for label in measurements:
        rows[label]=summarize_ms(measurements[label],memory[label])
    return rows


@torch.inference_mode()
def timed_decode(models,prompt,n,repeats,device):
    """Prefill / graph capture deliberately excluded from decoder throughput."""
    entries=[('stock_eager','stock',False),('builtin_opt_eager','builtin',False)]
    if device.type=='cuda':
        entries += [('v05h_eager','v05h',False),('builtin_opt_graph','builtin_graph',True),('v05h_graph','v05h_graph',True)]
    out={}
    for label,key,graph in entries:
        m=models[key]
        times=[]
        for trial in range(repeats):
            if graph:
                session=m.prepare_cuda_graph_generation(prompt,n)
                sync()
                t0=time.perf_counter()
                session.generate()
                sync()
                t=(time.perf_counter()-t0)*1000
                del session
            else:
                cache=m._generation_caches(prompt,prompt.shape[1]+n)
                logits, cache=m.prefill(prompt,cache_buffers=cache)
                cur=logits[:,-1,:].argmax(dim=-1,keepdim=True)
                sync()
                t0=time.perf_counter()
                for step in range(n-1):
                    logits,cache=m.decode_step(cur,cache,position=prompt.shape[1]+step)
                    cur=logits[:,-1,:].argmax(dim=-1,keepdim=True)
                sync()
                t=(time.perf_counter()-t0)*1000
            times.append(t)
        row=summarize_ms(times)
        row['measured_decode_steps']=n-1
        row['generated_tokens_per_second_p50']=round((n-1)*prompt.size(0)*1000/row['p50_ms'],3)
        out[label]=row
    return out


def _run_inner(level,repeats,output,validate_only=False):
    if repeats < 3:
        raise ValueError('use 3+ repetitions')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type=='cpu':
        print('CPU smoke mode: no native CUDA or CUDA Graph results can be produced.',flush=True)
        configs=[dict(cache='tight',batch=1,prompt=24,tokens=8,seed=2026)]
    else:
        configs=PROFILES[level]
        print('GPU:',torch.cuda.get_device_name(0),'PyTorch:',torch.__version__,flush=True)
        print('CUDA:',torch.version.cuda,'Compile flag:',os.getenv('MLBRICKS_BOLT_V05_ALLOW_JIT'),flush=True)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
    report={'title':'MLBricks stock BOLT vs built-in optimized BOLT vs native v0.5h',
            'status':'benchmarked, not production release',
            'hardware':(torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU'),
            'torch':torch.__version__,
            'dtype':('float16' if device.type=='cuda' else 'float32'),
            'models':'real MLBricks BoltModel; 6 layers, d384, 12 heads (GPU); untrained matched weights',
            'limits':'position=none, no RoPE, no trained quality, no model-load or native JIT in timing',
            'cases':[]}
    failed=False
    for ci,cfg in enumerate(configs,1):
        print('\nCASE',ci,'/',len(configs),cfg,flush=True)
        torch.manual_seed(cfg['seed']+193)
        prompt=torch.randint(0,2048,(cfg['batch'],cfg['prompt']),device=device)
        models=make_models(cfg,device)
        row={'config':cfg}
        row['greedy_output_parity']=closed_loop(models,prompt,cfg['tokens'],device)
        print('  Greedy parity:',{k:v.get('passed') for k,v in row['greedy_output_parity'].items()},flush=True)
        row['numeric_parity']=numerics(models,prompt,cfg['tokens'],device)
        print('  Full numerical parity:',{k:v.get('passed') for k,v in row['numeric_parity'].items()},flush=True)
        allpass=(all(x.get('passed',False) for x in row['greedy_output_parity'].values()) and
                 all(x.get('passed',False) for x in row['numeric_parity'].values()))
        row['parity_passed']=bool(allpass)
        failed=failed or not allpass
        if not validate_only:
            if not allpass:
                print('  WARNING: parity failed; speeds are diagnostic only.',flush=True)
            row['end_to_end_request']=request_latency(models,prompt,cfg['tokens'],repeats,device)
            row['decode_only']=timed_decode(models,prompt,cfg['tokens'],repeats,device)
            for variant, timing in row['end_to_end_request'].items():
                print(f'  E2E {variant:20s}: {timing["p50_ms"]:9.3f} ms; ' +
                      f'extra peak {timing["extra_peak_mib"]:9.2f} MiB',flush=True)
            for variant, timing in row['decode_only'].items():
                print(f'  DECODE {variant:20s}: {timing["generated_tokens_per_second_p50"]:9.2f} tok/s',flush=True)
        if not allpass:
            row['performance_validation']='UNVERIFIED: parity failed; do not use these speeds for release decisions.'
        elif not validate_only:
            row['performance_validation']='Parity passed for this scenario.'
        report['cases'].append(row)
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(report,indent=2,default=str))
        del models,prompt
        gc.collect()
        if device.type=='cuda': torch.cuda.empty_cache()
    report['all_parity_passed']=not failed
    report['completed']=True
    output.write_text(json.dumps(report,indent=2,default=str))
    csvpath=output.with_suffix('.csv')
    with csvpath.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['cache','batch','prompt','tokens','seed','variant','request_p50_ms','request_observed_p95_ms','decode_tok_s','extra_peak_mib'])
        writer.writeheader()
        for case in report['cases']:
            for label,ms in case.get('end_to_end_request',{}).items():
                dec=case.get('decode_only',{}).get(label,{})
                writer.writerow(dict(cache=case['config']['cache'],batch=case['config']['batch'],prompt=case['config']['prompt'],tokens=case['config']['tokens'],seed=case['config']['seed'],variant=label,
                                     request_p50_ms=ms['p50_ms'],request_observed_p95_ms=ms['observed_p95_ms'],decode_tok_s=dec.get('generated_tokens_per_second_p50'),extra_peak_mib=ms['extra_peak_mib']))
    print('\nCompleted. All parity passed:',report['all_parity_passed'],flush=True)
    print('JSON report:',output,' CSV:',csvpath,flush=True)
    return report


def run(level, repeats, output, validate_only=False):
    """Keep every stock/optimized forward under the SAME FP16 autocast policy.

    The stock AVQ readout has an FP32 recurrent state and FP16 back-projection
    weights. Without CUDA autocast, F.linear raises a dtype mismatch before
    benchmarking can start. This wrapper changes only the benchmark execution
    context; it does not patch the stock BOLT implementation or model weights.
    """
    from contextlib import nullcontext
    use_cuda = torch.cuda.is_available()
    context = (torch.autocast(device_type='cuda', dtype=torch.float16)
               if use_cuda else nullcontext())
    print('BENCHMARK FP16 AUTOCAST:', 'ENABLED (all variants)' if use_cuda else 'CPU FP32', flush=True)
    with context:
        return _run_inner(level, repeats, output, validate_only)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--level',choices=tuple(PROFILES),default='quick')
    parser.add_argument('--repeats',type=int,default=7)
    parser.add_argument('--output',type=Path,default=Path('/kaggle/working/bolt_mlbricks_vs_v05h.json'))
    parser.add_argument('--validate-only',action='store_true')
    args=parser.parse_args()
    run(args.level,args.repeats,args.output,args.validate_only)
