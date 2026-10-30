"""Opt-in BOLT v0.2 optimized inference demo (PyTorch only, no Triton).

Run: python examples/bolt_optimized_inference.py
Optional on CUDA: python examples/bolt_optimized_inference.py --graph
"""
import argparse
import time
import torch
from mlbricks import BoltModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--graph', action='store_true', help='Experimental CUDA Graph greedy decode')
    ap.add_argument('--prompt', type=int, default=128)
    ap.add_argument('--new-tokens', type=int, default=64)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()
    if args.prompt < 1 or args.new_tokens < 0:
        ap.error('prompt must be >=1 and new-tokens >=0')
    device = torch.device(args.device)
    model = BoltModel(vocab_size=2048,dim=384,heads=12,layers=6,
                      bolt_mode='quality',bolt_cache='tight',avq_dim=198,
                      context=args.prompt+args.new_tokens+8,
                      backend='pytorch',position='none',dropout=0).eval().to(device)
    if args.graph and device.type != 'cuda':
        ap.error('--graph requires CUDA')
    if device.type == 'cuda':
        model.half()  # optional: use .bfloat16() for supported GPUs
    tokens = torch.randint(0,2048,(1,args.prompt),device=device)

    with torch.inference_mode():
        t0=time.perf_counter()
        baseline=model.generate(tokens,args.new_tokens,temperature=0)
        if device.type=='cuda': torch.cuda.synchronize()
        original_ms=(time.perf_counter()-t0)*1000

        model.enable_bolt_optimized_inference(True)
        t0=time.perf_counter()
        optimized=model.generate(tokens,args.new_tokens,temperature=0)
        if device.type=='cuda': torch.cuda.synchronize()
        optimized_ms=(time.perf_counter()-t0)*1000
        assert torch.equal(baseline,optimized), 'Optimized eager token mismatch'
        print(f'Original greedy generation (includes prefill): {original_ms:.3f} ms')
        print(f'Optimized greedy generation (includes prefill): {optimized_ms:.3f} ms')
        if args.graph:
            # Setup includes prefill, warmup and capture; benchmark replay separately.
            t0=time.perf_counter()
            session=model.prepare_cuda_graph_generation(tokens,args.new_tokens)
            torch.cuda.synchronize()
            setup_ms=(time.perf_counter()-t0)*1000
            t0=time.perf_counter()
            graph_tokens=session.generate()
            torch.cuda.synchronize()
            replay_ms=(time.perf_counter()-t0)*1000
            print(f'CUDA Graph setup: {setup_ms:.3f} ms')
            print(f'CUDA Graph greedy replay ({max(args.new_tokens-1,0)} steps): {replay_ms:.3f} ms')
            assert torch.equal(baseline,graph_tokens), 'CUDA Graph token mismatch'
            print('Graph parity: passed')
        print('Eager parity: passed')


if __name__=='__main__':
    main()
