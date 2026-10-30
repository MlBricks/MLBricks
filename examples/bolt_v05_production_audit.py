"""BOLT v0.5b production-candidate CUDA audit using the real MLBricks API.

Runs CUDA in this fresh Python process; the Kaggle notebook bootstraps it without
importing torch into the notebook kernel. No trained-checkpoint claims are made.

Example:
  MLBRICKS_BOLT_V05_ALLOW_JIT=1 python examples/bolt_v05_production_audit.py \
      --level full --output /kaggle/working/bolt_v05_production_audit.json

Audit thresholds are engineering test criteria, NOT a trained-model quality guarantee.
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import os
from pathlib import Path
import statistics
import sys
import time
import traceback

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mlbricks import BoltModel  # noqa: E402

LOGIT_ATOL = 0.03
STATE_ATOL = 0.002
KV_ATOL = 0.03


def sync():
    torch.cuda.synchronize()


def scenarios(level):
    # 'tight'=16 latent features/head; 'normal'=32 features/head.
    basic = [
        dict(cache='tight', batch=1, prompt=128, tokens=32, seed=2026),
        dict(cache='normal', batch=1, prompt=128, tokens=32, seed=5505),
        dict(cache='tight', batch=2, prompt=256, tokens=32, seed=5505),
        dict(cache='normal', batch=1, prompt=512, tokens=64, seed=2026),
    ]
    if level == 'full':
        basic += [
            dict(cache='tight', batch=1, prompt=1024, tokens=64, seed=5505),
            dict(cache='normal', batch=2, prompt=512, tokens=64, seed=2026),
            dict(cache='tight', batch=4, prompt=128, tokens=32, seed=5505),
            dict(cache='normal', batch=1, prompt=2048, tokens=32, seed=2026),
        ]
    return basic


def finite_caches(caches):
    return all(bool(torch.isfinite(c.c).all().item()) and
               (c.state is None or bool(torch.isfinite(c.state).all().item()))
               for c in caches)


def state_snapshot(caches):
    return [c.state.detach().clone() if c.state is not None else None for c in caches]


def cache_delta(a, b):
    worst_s = worst_c = 0.0
    for ref, cand in zip(a, b):
        if ref.state is not None:
            if not bool(torch.isfinite(cand.state).all().item()):
                return float('inf'), float('inf')
            worst_s = max(worst_s, float((ref.state.float() - cand.state.float()).abs().max().item()))
        if not bool(torch.isfinite(cand.c).all().item()):
            return float('inf'), float('inf')
        active = min(ref.length, cand.length)
        if active > 0:
            worst_c = max(worst_c, float((ref.c[:, :, :active, :].float() -
                                         cand.c[:, :, :active, :].float()).abs().max().item()))
    return worst_s, worst_c


def make_model(s, *, layers=6, dim=384, heads=12, vocab=2048):
    torch.manual_seed(s['seed'])
    capacity = s['prompt'] + s['tokens'] + 8
    m = BoltModel(vocab_size=vocab, layers=layers, dim=dim, heads=heads,
                  context=capacity, bolt_mode='quality', bolt_cache=s['cache'],
                  avq_dim=198, position='none', dropout=0,
                  backend='pytorch').eval().to('cuda', dtype=torch.float16)
    with torch.no_grad():
        for block in m.blocks:
            a = block.attn._modern
            a.p.uniform_(.7, 1.3)
            a.avq.gamma.fill_(.21)
            a.avq.norm.weight.uniform_(.7, 1.3)
    m.enable_bolt_optimized_inference(True)
    actual = m.blocks[0].attn._modern.latent_dim
    assert actual == (16 if s['cache']=='tight' else 32), f'Wrong BOLT latent dimension: {actual}'
    return m


def configured(ref, gate, avq):
    # Crucial: independent Parameter storage and no shared graph buffers.
    m = copy.deepcopy(ref)
    if gate or avq:
        m.enable_bolt_v05_native(gate=gate, avq=avq)
        m.prepare_bolt_v05_native()
    return m


@torch.inference_mode()
def eager_trace(m, prompt, new_tokens):
    caches = m._generation_caches(prompt, prompt.size(1) + new_tokens)
    logits, caches = m.prefill(prompt, cache_buffers=caches)
    if not bool(torch.isfinite(logits).all().item()):
        raise AssertionError('Nonfinite reference prefill logits')
    cur = logits[:, -1:].argmax(dim=-1)
    gen = [cur.detach().clone()]
    out_logits, out_states = [], []
    for i in range(new_tokens-1):
        logits, caches = m.decode_step(cur, caches, position=prompt.size(1) + i)
        last = logits[:, -1, :]
        if not bool(torch.isfinite(last).all().item()):
            raise AssertionError(f'Nonfinite eager logits at step {i}')
        if not finite_caches(caches):
            raise AssertionError(f'Nonfinite eager KV/AVQ state at step {i}')
        cur = last.argmax(dim=-1, keepdim=True)
        gen.append(cur.detach().clone())
        out_logits.append(last.detach().clone())
        out_states.append(state_snapshot(caches))
    return dict(tokens=torch.cat([prompt, torch.cat(gen, dim=1)], dim=1),
                logits=out_logits, states=out_states, caches=caches)


@torch.inference_mode()
def graph_trace(m, prompt, new_tokens):
    # Hook records the tensor produced by the *captured* lm_head call. Holding
    # that tensor keeps its CUDA Graph output storage alive across graph replay.
    graph_outputs = []

    def hook(_module, _inputs, output):
        if output.ndim == 3 and output.size(1) == 1:
            graph_outputs.append(output)

    handle = m.lm_head.register_forward_hook(hook)
    try:
        sess = m.prepare_cuda_graph_generation(prompt, new_tokens)
    finally:
        handle.remove()
    if new_tokens < 2 or sess.graph is None:
        raise RuntimeError('Graph trace requires >=2 generated tokens')
    if not graph_outputs:
        raise RuntimeError('Graph capture did not expose its lm_head output')
    captured_logits = graph_outputs[-1]  # capture follows warmup; prefill is longer
    graph_outputs.clear()
    out_logits, out_states = [], []
    for i in range(new_tokens-1):
        sess.graph.replay()
        # Clone before next replay overwrites captured buffers.
        last = captured_logits[:, -1, :]
        if not bool(torch.isfinite(last).all().item()):
            raise AssertionError(f'Graph nonfinite logits at step {i}')
        if not finite_caches(sess.caches):
            raise AssertionError(f'Graph nonfinite cache/state at step {i}')
        out_logits.append(last.detach().clone())
        out_states.append(state_snapshot(sess.caches))
    sync()
    return dict(tokens=sess.tokens.detach().clone(), logits=out_logits,
                states=out_states, caches=sess.caches,
                _session=sess, _captured_logits=captured_logits)


def compare(base, test, *, mode):
    token_diff = int((base['tokens'] != test['tokens']).sum().item())
    worst_logit = 0.0
    worst_state = 0.0
    first_bad = None
    if len(base['logits']) != len(test['logits']):
        raise AssertionError('Trace lengths differ')
    for i, (a, b) in enumerate(zip(base['logits'], test['logits'])):
        diff = float((a.float() - b.float()).abs().max().item())
        worst_logit = max(worst_logit, diff)
        if (not torch.isfinite(b).all().item() or diff > LOGIT_ATOL) and first_bad is None:
            first_bad = {'step': i+2, 'operation': 'logits', 'delta': diff}
        for layer, (sa, sb) in enumerate(zip(base['states'][i], test['states'][i])):
            if sa is None and sb is None:
                continue
            if sa is None or sb is None:
                raise AssertionError('AVQ state presence mismatch')
            delta = float((sa.float() - sb.float()).abs().max().item())
            worst_state = max(worst_state, delta)
            if (not torch.isfinite(sb).all().item() or delta > STATE_ATOL) and first_bad is None:
                first_bad = {'step': i+2, 'layer': layer, 'operation': 'avq_state', 'delta': delta}
    final_state, kv_delta = cache_delta(base['caches'], test['caches'])
    # A graph cache does not track .length in Python; compare *all active slots*
    # separately in the graph path (including first prefill and generated tokens).
    if mode == 'graph':
        kv_delta = 0.0
        context = base['tokens'].size(1) - 1
        for ca, cb in zip(base['caches'], test['caches']):
            delta = float((ca.c[:, :, :context].float() - cb.c[:, :, :context].float()).abs().max().item())
            kv_delta = max(kv_delta, delta)
    good = (token_diff == 0 and first_bad is None and
            worst_logit <= LOGIT_ATOL and worst_state <= STATE_ATOL and
            final_state <= STATE_ATOL and kv_delta <= KV_ATOL)
    return dict(passed=bool(good), mode=mode, token_mismatches=token_diff,
                max_logit_delta=round(worst_logit, 8),
                max_avq_state_delta=round(worst_state, 8),
                final_state_delta=round(final_state, 8),
                max_kv_delta=round(kv_delta, 8), first_failure=first_bad)


def stress_one_shot(ref, prompt, new_tokens):
    # Separate instance from graph_trace, so this checks the public one-shot API.
    m = configured(ref, gate=True, avq=True)
    with torch.inference_mode():
        expected = eager_trace(ref, prompt, new_tokens)['tokens']
        s = m.prepare_cuda_graph_generation(prompt, new_tokens)
        result = s.generate().clone()
        if not torch.equal(expected, result):
            raise AssertionError('Public session.generate differs from eager')
        try:
            s.generate()
            raise AssertionError('One-shot session unexpectedly generated twice')
        except RuntimeError as ex:
            if 'one-shot' not in str(ex):
                raise
    return True


def memory_stability(ref, prompt, tokens, sessions):
    # Repeatedly capture on new models and *different prompts*, then release
    # sessions and check output correctness, resource cleanup and memory drift.
    gc.collect(); torch.cuda.empty_cache(); sync()
    before = int(torch.cuda.memory_allocated())
    all_equal = True
    peak_alloc = before
    first_bad_request = None
    for i in range(sessions):
        torch.manual_seed(9012+i)
        request = torch.randint(0, 2048, prompt.shape, device='cuda')
        m = configured(ref, gate=True, avq=True)
        baseline_model = configured(ref,gate=False,avq=False)
        with torch.inference_mode():
            expected = baseline_model.generate(request,tokens,temperature=0)
            session = m.prepare_cuda_graph_generation(request, tokens)
            out = session.generate()
            exact = bool(torch.equal(out,expected))
            states_ok=finite_caches(session.caches)
            all_equal = all_equal and exact and states_ok
            if (not exact or not states_ok) and first_bad_request is None:
                first_bad_request=i
        peak_alloc = max(peak_alloc, int(torch.cuda.max_memory_allocated()))
        del session, m, out, baseline_model, expected, request
        gc.collect(); torch.cuda.empty_cache(); sync()
    after = int(torch.cuda.memory_allocated())
    growth = after-before
    return dict(sessions=sessions, output_and_state_parity=all_equal,
                first_bad_request=first_bad_request,
                allocated_before_bytes=before, allocated_after_bytes=after,
                growth_bytes=growth, peak_allocated_bytes=peak_alloc,
                passed=(all_equal and growth < 128*1024**2))


def graph_safety_contract(ref,prompt,tokens):
    checks={}
    model=configured(ref,gate=True,avq=True)
    with torch.inference_mode():
        s=model.prepare_cuda_graph_generation(prompt,tokens)
        result=s.generate().clone()
        checks['normal_generation']=result.shape==(prompt.shape[0],prompt.shape[1]+tokens)
        try:
            s.generate()
        except RuntimeError as ex:
            checks['one_shot_rejected']=('one-shot' in str(ex))
        else:
            checks['one_shot_rejected']=False
    del s,model
    # Parameter mutation must invalidate a captured graph, not replay stale
    # folded weights. Construct model outside inference_mode so version exists.
    mut_model=configured(ref,gate=True,avq=True)
    s=mut_model.prepare_cuda_graph_generation(prompt,tokens)
    with torch.no_grad():
        next(mut_model.parameters()).add_(.1)
    try:
        s.generate()
    except RuntimeError as ex:
        checks['parameter_mutation_rejected']=('changed' in str(ex))
    else:
        checks['parameter_mutation_rejected']=False
    del s,mut_model
    # Overflow must be rejected; don't allow a cache overwrite or truncated
    # output when the requested budget exceeds the model's context.
    limits=configured(ref,gate=True,avq=True)
    try:
        limits.prepare_cuda_graph_generation(prompt,limits.config.context-prompt.size(1)+1)
    except ValueError:
        checks['context_overflow_rejected']=True
    else:
        checks['context_overflow_rejected']=False
    del limits
    checks['passed']=all(checks.values())
    return checks

def timing(fn, reps):
    samples = []
    peaks = []
    for _ in range(reps):
        gc.collect(); torch.cuda.empty_cache(); sync()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        fn()
        sync()
        samples.append((time.perf_counter()-start)*1000)
        peaks.append(torch.cuda.max_memory_allocated())
    samples.sort()
    p95 = samples[min(len(samples)-1, int((len(samples)-1)*.95+.99999))]
    return dict(p50_ms=round(statistics.median(samples), 3),
                p95_ms=round(p95, 3),
                samples_ms=[round(x, 3) for x in samples],
                max_allocated_mb=round(max(peaks)/1024**2, 2))


@torch.inference_mode()
def eager_generate_no_audit(m, prompt, tokens):
    """Fair timed eager path: NO per-token GPU sync, snapshot or diagnostics."""
    buffers = m._generation_caches(prompt, prompt.size(1)+tokens)
    logits, caches = m.prefill(prompt, cache_buffers=buffers)
    current = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    for i in range(tokens-1):
        logits, caches = m.decode_step(current, caches, position=prompt.size(1)+i)
        current = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    return current


def performance_probe(s, rep):
    """Separate graph decode speed from the complete one-shot request cost."""
    torch.manual_seed(s['seed']+31)
    prompt = torch.randint(0, 2048, (s['batch'], s['prompt']), device='cuda')
    reference = make_model(s)
    # Precreate models to exclude Python construction and device transfers from
    # the inference comparison. Runtime E2E INCLUDES prefill, cache, graph capture.
    models = {
        'pytorch_eager': configured(reference,False,False),
        'native_eager': configured(reference,True,True),
        'pytorch_graph': configured(reference,False,False),
        'native_graph': configured(reference,True,True),
    }

    def eager_run(m):
        return eager_generate_no_audit(m,prompt,s['tokens'])

    def graph_run(m):
        session=m.prepare_cuda_graph_generation(prompt,s['tokens'])
        return session.generate()

    def profile_one(name, mode):
        m=models[name]
        action=(lambda: eager_run(m)) if mode=='eager' else (lambda: graph_run(m))
        # First run warms kernels, CUDA context and scratch; this warmup is not
        # counted in steady-state, but graph capture IS included in every timed
        # graph request, because the generator is a one-shot session.
        with torch.inference_mode():
            action(); sync()
            return timing(action,rep)

    result={}
    for name in ('pytorch_eager','native_eager','pytorch_graph','native_graph'):
        result[name+'_request_e2e']=profile_one(name, 'graph' if name.endswith('graph') else 'eager')
    # Separate decode-only measurements for comparison with earlier notebooks.
    def eager_decode():
        m=models['native_eager']
        buffers=m._generation_caches(prompt,prompt.size(1)+s['tokens'])
        logits,caches=m.prefill(prompt,cache_buffers=buffers)
        cur=logits[:,-1,:].argmax(dim=-1,keepdim=True)
        sync(); t0=time.perf_counter()
        for i in range(s['tokens']-1):
            logits,caches=m.decode_step(cur,caches,position=prompt.size(1)+i)
            cur=logits[:,-1,:].argmax(dim=-1,keepdim=True)
        sync(); return (time.perf_counter()-t0)*1000

    def graph_decode():
        m=models['native_graph']
        sess=m.prepare_cuda_graph_generation(prompt,s['tokens'])
        sync();t0=time.perf_counter()
        sess.generate()
        sync();return (time.perf_counter()-t0)*1000

    with torch.inference_mode():
        native_eager_decode=[eager_decode() for _ in range(rep)]
        native_graph_decode=[graph_decode() for _ in range(rep)]
    result['native_eager_decode_ms']=round(statistics.median(native_eager_decode),4)
    result['native_graph_decode_ms']=round(statistics.median(native_graph_decode),4)
    result['native_eager_decode_tok_s']=round((s['tokens']-1)*s['batch']*1000/statistics.median(native_eager_decode),3)
    result['native_graph_decode_tok_s']=round((s['tokens']-1)*s['batch']*1000/statistics.median(native_graph_decode),3)
    ref_time=result['pytorch_eager_request_e2e']['p50_ms']
    result['native_eager_request_speedup_vs_ref_eager']=round(ref_time/result['native_eager_request_e2e']['p50_ms'],4)
    result['native_graph_request_speedup_vs_ref_eager']=round(ref_time/result['native_graph_request_e2e']['p50_ms'],4)
    result['fastest_request_backend']=min(
        ('pytorch_eager','native_eager','pytorch_graph','native_graph'),
        key=lambda key: result[key+'_request_e2e']['p50_ms'])
    result['note']='Full request excludes model load/JIT build but INCLUDES prefill and one-shot graph capture.'
    return result

def save(path, document):
    out = Path(path); out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(document, indent=2, default=str))


def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError('Production GPU audit requires CUDA (Tesla T4 or newer)')
    from torch.utils.cpp_extension import CUDA_HOME
    if CUDA_HOME is None:
        raise RuntimeError('CUDA toolkit/nvcc unavailable; cannot compile native BOLT')
    if os.environ.get('MLBRICKS_BOLT_V05_ALLOW_JIT') != '1':
        raise RuntimeError('Set MLBRICKS_BOLT_V05_ALLOW_JIT=1 before launching Python')
    torch.manual_seed(2026)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.cuda.get_device_name(0)
    report = dict(title='BOLT v0.5b production-candidate audit',
                  hardware={'gpu':device,'torch':torch.__version__,'cuda':torch.version.cuda,
                            'compute_capability':torch.cuda.get_device_capability(0)},
                  config=dict(level=args.level, logit_atol=LOGIT_ATOL,
                              state_atol=STATE_ATOL, kv_atol=KV_ATOL,
                              dims=dict(layers=6, dim=384, heads=12, avq=198,
                                        precision='fp16', position='none')),
                  cases=[], auxiliary={}, performance={}, pending=[], verdict='NOT_EVALUATED')
    cases = scenarios(args.level)
    variants = [('pytorch',False,False),('ug_native',True,False),
                ('avq_native',False,True),('combined',True,True)]
    variants = variants if args.level=='full' else [variants[0],variants[-1]]
    print('GPU:',device,'torch:',torch.__version__,'CUDA:',torch.version.cuda, flush=True)
    print('Audit cases:',len(cases),'variants:',[x[0] for x in variants],flush=True)
    for n,s in enumerate(cases,1):
        print(f'\nCASE {n}/{len(cases)}: {s}',flush=True)
        case = dict(config=s, tests=[])
        report['cases'].append(case)
        try:
            torch.manual_seed(s['seed']+31)
            prompt = torch.randint(0,2048,(s['batch'],s['prompt']),device='cuda')
            ref=make_model(s)
            with torch.inference_mode():
                baseline=eager_trace(ref,prompt,s['tokens'])
            for name,gate,avq in variants:
                for mode in ('eager','graph'):
                    row=dict(variant=name, mode=mode)
                    try:
                        m=configured(ref,gate,avq)
                        with torch.inference_mode():
                            observed = (graph_trace(m,prompt,s['tokens']) if mode=='graph'
                                        else eager_trace(m,prompt,s['tokens']))
                        row.update(compare(baseline,observed,mode=mode))
                        if '_session' in observed:
                            del observed['_session'], observed['_captured_logits']
                        del m, observed
                    except Exception as e:
                        row.update(passed=False,error=f'{type(e).__name__}: {e}',
                                   traceback=traceback.format_exc(limit=4))
                    print(' ',name,mode, 'PASS' if row['passed'] else 'FAIL',
                          'dlogit=',row.get('max_logit_delta'),
                          'dstate=',row.get('max_avq_state_delta'),
                          'bad=',row.get('token_mismatches'),flush=True)
                    case['tests'].append(row)
            if n==1:
                report['auxiliary']['public_one_shot']=stress_one_shot(ref,prompt,s['tokens'])
                report['auxiliary']['graph_contract']=graph_safety_contract(ref,prompt,s['tokens'])
            del baseline, ref
        except Exception as e:
            case['error']=f'{type(e).__name__}: {e}'
            case['traceback']=traceback.format_exc(limit=5)
            print(' CASE ERROR',case['error'],flush=True)
        save(args.output,report)
        gc.collect(); torch.cuda.empty_cache()
    # One repeated-session stability check (full: more iterations).
    try:
        s=cases[0]; torch.manual_seed(s['seed']+31)
        prompt=torch.randint(0,2048,(s['batch'],s['prompt']),device='cuda')
        ref=make_model(s)
        report['auxiliary']['repeat_session_memory']=memory_stability(
            ref,prompt,s['tokens'],sessions=10 if args.level=='full' else 3)
        del ref
    except Exception as e:
        report['auxiliary']['repeat_session_memory']={'passed':False,'error':str(e)}
    save(args.output,report)
    # Separate production latency probe, not part of numerical trace overhead.
    try:
        perf = dict(cache='tight',batch=1,prompt=512,tokens=64,seed=2026)
        report['performance']=performance_probe(perf,args.perf_repeats)
    except Exception as e:
        report['performance']={'error':f'{type(e).__name__}: {e}'}
    numerical_ok=bool(report['cases']) and all(
        'error' not in c and c['tests'] and all(t.get('passed',False) for t in c['tests'])
        for c in report['cases'])
    auxiliary_ok=(report['auxiliary'].get('public_one_shot') is True and
                  report['auxiliary'].get('repeat_session_memory',{}).get('passed',False) and
                  report['auxiliary'].get('graph_contract',{}).get('passed',False))
    report['pending'] = [
        'Trained checkpoint output-quality/accuracy benchmarks are not provided by this random-weight test.',
        'Independent build/CI verification on supported GPUs, driver/CUDA/PyTorch versions is still needed.',
        'Production concurrency, service-level p95 latency, and multi-hour soak are not exercised.',
        'Only FP16, causal position=none, fixed prompt sizes, greedy decoding are supported/tested here.',
        'Extension prebuild/distribution and security review are not validated by Kaggle JIT compilation.',
    ]
    if numerical_ok and auxiliary_ok and 'error' not in report['performance']:
        report['verdict']='GPU_CANDIDATE_PASSED__PRODUCTION_APPROVAL_PENDING'
    else:
        report['verdict']='NO_GO__PRODUCTION_TEST_FAILURE'
    save(args.output,report)
    print('\n=== SUMMARY ===',flush=True)
    print('Numerical:',numerical_ok,'Auxiliary:',auxiliary_ok,
          'Performance measured:','error' not in report['performance'],flush=True)
    print('Verdict:',report['verdict'],flush=True)
    print('JSON:',args.output,flush=True)
    return 0 if report['verdict'].startswith('GPU_CANDIDATE') else 2


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--level',choices=['smoke','full'],default='full')
    p.add_argument('--perf-repeats',type=int,default=5)
    p.add_argument('--output',default='bolt_v05_production_audit.json')
    p.add_argument('--plan-only',action='store_true')
    args=p.parse_args()
    if args.perf_repeats<2: p.error('--perf-repeats must be >=2')
    if args.plan_only:
        print(json.dumps({'cases':scenarios(args.level),'variants':
                          ['pytorch','ug_native','avq_native','combined'] if args.level=='full'
                          else ['pytorch','combined']},indent=2))
        return 0
    return run(args)

if __name__=='__main__':
    sys.exit(main())
