"""BOLT v0.5 real-MLBricks native CUDA validation + decode benchmark.

Run on Kaggle from the patched repository root (no package install needed):
    MLBRICKS_BOLT_V05_ALLOW_JIT=1 python examples/bolt_v05_native_benchmark.py

Measures **decode only**: prefill and graph compilation are outside timing.
The first generated token is obtained from prefill; measured work consists of
``new_tokens-1`` greedy steps in both eager and CUDA Graph modes. Run with a
Tesla T4 or later. The tests reject non-finite state and token mismatches.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mlbricks import BoltModel  # noqa: E402


def synchronize():
    torch.cuda.synchronize()


def make_model(seed: int, *, cache: str, layers: int, dim: int, heads: int,
               vocab: int, context: int):
    torch.manual_seed(seed)
    m = BoltModel(vocab_size=vocab, layers=layers, dim=dim, heads=heads,
                  context=context, bolt_mode='quality', bolt_cache=cache,
                  avq_dim=198, position='none', dropout=0,
                  backend='pytorch').eval().to('cuda', dtype=torch.float16)
    with torch.no_grad():
        for block in m.blocks:
            a = block.attn._modern
            a.p.uniform_(.7, 1.3)
            a.avq.gamma.fill_(.21)
            a.avq.norm.weight.uniform_(.7, 1.3)
    m.enable_bolt_optimized_inference(True)
    return m


def finite(caches):
    return all(torch.isfinite(c.c).all().item() and (
        c.state is None or torch.isfinite(c.state).all().item()) for c in caches)


@torch.inference_mode()
def eager_generate_and_time(m, prompt: torch.Tensor, new_tokens: int):
    total = prompt.size(1) + new_tokens
    cache_buffers = m._generation_caches(prompt, total)
    logits, caches = m.prefill(prompt, cache_buffers=cache_buffers)
    first = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated = [first.clone()]
    current = first
    synchronize()
    start = time.perf_counter()
    for i in range(new_tokens - 1):
        logits, caches = m.decode_step(current, caches, position=prompt.size(1)+i)
        current = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(current.clone())
    synchronize()
    elapsed = time.perf_counter()-start
    return torch.cat((prompt, torch.cat(generated,dim=1)), dim=1), elapsed, finite(caches)


@torch.inference_mode()
def graph_generate_and_time(m, prompt: torch.Tensor, new_tokens: int):
    session = m.prepare_cuda_graph_generation(prompt, new_tokens)
    synchronize()
    start = time.perf_counter()
    tokens = session.generate()
    synchronize()
    elapsed = time.perf_counter()-start
    return tokens.clone(), elapsed, finite(session.caches)


def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError('Tesla T4 / CUDA GPU required; do not interpret CPU throughput as native GPU')
    if args.new_tokens < 2:
        raise ValueError('--new-tokens must be >=2 to measure decode steps')
    if os.environ.get('MLBRICKS_BOLT_V05_ALLOW_JIT') != '1':
        raise RuntimeError('Set MLBRICKS_BOLT_V05_ALLOW_JIT=1 before starting Python')
    print('Torch:',torch.__version__,'CUDA:',torch.version.cuda,'GPU:',torch.cuda.get_device_name(0))
    variants = [('pytorch',False,False),('ug_native',True,False),
                ('avq_native',False,True),('combined',True,True)]
    results=[]
    for cache in args.caches:
        for seed in args.seeds:
            print(f'\n=== BOLT {cache} / seed {seed} ===',flush=True)
            reference = make_model(seed, cache=cache, layers=args.layers,
                                   dim=args.dim, heads=args.heads,
                                   vocab=args.vocab,context=args.prompt+args.new_tokens+8)
            torch.manual_seed(seed+31)
            prompt = torch.randint(0,args.vocab,(args.batch,args.prompt),device='cuda')
            baseline, _, _ = eager_generate_and_time(reference, prompt, args.new_tokens)
            for name, gate, avq in variants:
                for mode in ('eager','graph'):
                    record=dict(cache=cache, seed=seed, variant=name, mode=mode,
                                measured_steps=args.new_tokens-1)
                    try:
                        durations=[]
                        successful=True
                        # New independent model instance for every rollout.
                        for rep in range(args.repeats):
                            m=copy.deepcopy(reference)
                            if gate or avq:
                                m.enable_bolt_v05_native(gate=gate, avq=avq)
                                m.prepare_bolt_v05_native()
                            torch.cuda.reset_peak_memory_stats()
                            if mode=='graph':
                                observed, sec, state_ok=graph_generate_and_time(m,prompt,args.new_tokens)
                            else:
                                observed, sec, state_ok=eager_generate_and_time(m,prompt,args.new_tokens)
                            if not state_ok or not torch.equal(observed, baseline):
                                successful=False
                                bad=(observed[:,args.prompt:] != baseline[:,args.prompt:]).sum().item()
                                record['mismatches']=int(bad)
                                record['state_finite']=bool(state_ok)
                                break
                            durations.append(sec)
                        record['passed'] = successful
                        if successful and durations:
                            sec = statistics.median(durations)
                            record['decode_ms'] = round(sec*1000,4)
                            record['tok_s'] = round((args.new_tokens-1)*args.batch/sec,3)
                        print(record,flush=True)
                    except Exception as exc:
                        record['passed']=False
                        record['error']=str(exc)[:1000]
                        print(record,flush=True)
                    results.append(record)
    path=Path(args.output)
    path.write_text(json.dumps(dict(configuration=vars(args),results=results),indent=2))
    print('\nSaved:',path)
    if any(not r.get('passed',False) for r in results):
        print('Some variants FAILED: do not deploy v0.5; inspect the failure table.')
    return results


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--caches', nargs='+',choices=('tight','normal'),default=['tight','normal'])
    parser.add_argument('--seeds', nargs='+',type=int,default=[2026,5505])
    parser.add_argument('--layers',type=int,default=6)
    parser.add_argument('--dim',type=int,default=384)
    parser.add_argument('--heads',type=int,default=12)
    parser.add_argument('--vocab',type=int,default=2048)
    parser.add_argument('--batch',type=int,default=1)
    parser.add_argument('--prompt',type=int,default=512)
    parser.add_argument('--new-tokens',type=int,default=64)
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--output',type=str,default='bolt_v05_mlbricks_gpu_validation.json')
    run(parser.parse_args())
