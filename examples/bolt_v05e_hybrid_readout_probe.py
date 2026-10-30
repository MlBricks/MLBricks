"""BOLT v0.5e targeted precision probe, using pinned v0.5c MLBricks source.

Compares reference PyTorch AVQ, native AVQ, and hybrid AVQ (native recurrent
update, PyTorch norm/read gate), under teacher-forced greedy inputs.
No trained-checkpoint or production-readiness claims.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import types
from pathlib import Path
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from examples.bolt_v05_production_audit import make_model, configured, scenarios
from mlbricks.bolt import native_v05

LOGIT_ATOL = 0.03

def delta(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.shape != b.shape:
        return float('inf')
    if not bool(torch.isfinite(a).all()) or not bool(torch.isfinite(b).all()):
        return float('inf')
    return (a.float() - b.float()).abs().max().item()


def hybrid_avq_step(self, o, state, *, inplace=False):
    # This trial deliberately recomputes readout in PyTorch while preserving
    # the exact same native CUDA AVQ state update. Never used for training.
    if not inplace or self.training or torch.is_grad_enabled():
        return self._v05e_original_step(o, state, inplace=inplace)
    if not self.fold_output_inference:
        raise RuntimeError('Hybrid AVQ requires optimized inference')
    if not o.is_cuda or state.dtype != torch.float32 or not state.is_contiguous():
        raise RuntimeError('Hybrid AVQ requires CUDA and contiguous FP32 cache state')
    raw = self.project(o).contiguous()
    shape = (o.shape[0], 1, self.width)
    if (self._v05_features is None or self._v05_features.shape != shape or
            self._v05_features.device != o.device):
        self._v05_features = torch.empty(shape, device=o.device, dtype=torch.float32)
    native_v05.avq_out(raw, state, self._v05_features, self.norm.eps)
    # Exact reference PyTorch arithmetic order for the normalization/read gate.
    r = torch.tanh(raw[..., 2 * self.width:]).float()
    sf = state[:, None, :]
    features = (sf * torch.rsqrt(sf.square().mean(-1, keepdim=True) + self.norm.eps)) * (1. + r)
    # Only the readout is changed. Preserve the same folded back weights and
    # residual cast as PostSoftmaxAVQ._readout.
    result = o + F.linear(features.to(o.dtype), self._inference_back()).to(o.dtype)
    return result, state


def enable_hybrid(model):
    for block in model.blocks:
        avq = block.attn._modern.avq
        assert avq.v05_native_avq
        avq._v05e_original_step = avq.step
        avq.step = types.MethodType(hybrid_avq_step, avq)
    return model


def install_capture(model, records):
    for layer, block in enumerate(model.blocks):
        avq = block.attn._modern.avq
        original = avq.step
        def wrapped(o, state, *, inplace=False, _avq=avq, _old=original, _layer=layer):
            result, updated = _old(o, state, inplace=inplace)
            if inplace:
                # Raw and native scratch are examined AFTER all operations.
                # Capture only the readout result and state, not a new projection.
                records.append({'layer':_layer, 'readout':result.detach().clone(),
                                'state': updated.detach().clone()})
            return result, updated
        avq.step = wrapped


def comparison(reference, candidate):
    return {
        'max_logit_abs': delta(reference, candidate),
        'mean_logit_abs': (reference.float()-candidate.float()).abs().mean().item(),
        'max_reference_abs': reference.float().abs().max().item(),
        'token_match': bool(torch.equal(reference.argmax(-1), candidate.argmax(-1))),
    }


@torch.inference_mode()
def run_case(config):
    torch.manual_seed(config['seed'] + 31)
    prompt = torch.randint(0, 2048, (config['batch'], config['prompt']), device='cuda')
    ref = make_model(config)
    native = configured(ref, gate=False, avq=True)
    hybrid = enable_hybrid(configured(ref, gate=False, avq=True))
    models = {'reference':ref, 'native':native, 'hybrid':hybrid}
    traces = {key: [] for key in models}
    for key, m in models.items():
        install_capture(m, traces[key])
    caches = {}
    prefill_logits = {}
    for key, m in models.items():
        cache = m._generation_caches(prompt, prompt.shape[1]+config['tokens'])
        prefill_logits[key], caches[key] = m.prefill(prompt, cache_buffers=cache)
    # Teacher-force identical reference greedy inputs to every decoder.
    cur = prefill_logits['reference'][:, -1:].argmax(-1)
    rows = []
    first_fail_native = first_fail_hybrid = None
    for i in range(config['tokens'] - 1):
        outputs = {}
        for key, m in models.items():
            traces[key].clear()
            out, caches[key] = m.decode_step(cur, caches[key], position=prompt.shape[1]+i)
            outputs[key] = out[:, -1, :].detach().clone()
        native_cmp = comparison(outputs['reference'], outputs['native'])
        hybrid_cmp = comparison(outputs['reference'], outputs['hybrid'])
        lstate = [delta(a['state'], b['state']) for a,b in zip(traces['reference'],traces['native'])]
        hstate = [delta(a['state'], b['state']) for a,b in zip(traces['reference'],traces['hybrid'])]
        lread = [delta(a['readout'], b['readout']) for a,b in zip(traces['reference'],traces['native'])]
        hread = [delta(a['readout'], b['readout']) for a,b in zip(traces['reference'],traces['hybrid'])]
        row = {'generated_token_index':i+2, 'native':native_cmp, 'hybrid':hybrid_cmp,
               'native_max_state_abs':max(lstate), 'hybrid_max_state_abs':max(hstate),
               'native_max_readout_abs':max(lread), 'hybrid_max_readout_abs':max(hread)}
        rows.append(row)
        if first_fail_native is None and (native_cmp['max_logit_abs']>LOGIT_ATOL or not native_cmp['token_match']):
            first_fail_native = row
        if first_fail_hybrid is None and (hybrid_cmp['max_logit_abs']>LOGIT_ATOL or not hybrid_cmp['token_match']):
            first_fail_hybrid = row
        if i in (0,1,2,config['tokens']-2):
            print('   token',i+2,'native',native_cmp['max_logit_abs'],
                  'hybrid',hybrid_cmp['max_logit_abs'], 'hybrid-state',max(hstate),flush=True)
        cur = outputs['reference'].argmax(-1, keepdim=True)
    return {
        'config':config,
        'first_native_failure':first_fail_native,
        'first_hybrid_failure':first_fail_hybrid,
        'native_worst_logit_abs':max(x['native']['max_logit_abs'] for x in rows),
        'hybrid_worst_logit_abs':max(x['hybrid']['max_logit_abs'] for x in rows),
        'native_worst_state_abs':max(x['native_max_state_abs'] for x in rows),
        'hybrid_worst_state_abs':max(x['hybrid_max_state_abs'] for x in rows),
        'all_native_tokens_match':all(x['native']['token_match'] for x in rows),
        'all_hybrid_tokens_match':all(x['hybrid']['token_match'] for x in rows),
        'rows':rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cases',default='1,4,5,7,8')
    parser.add_argument('--output',default='/kaggle/working/bolt_v05e_hybrid_precision.json')
    a = parser.parse_args()
    if not torch.cuda.is_available() or os.environ.get('MLBRICKS_BOLT_V05_ALLOW_JIT')!='1':
        raise RuntimeError('Requires CUDA T4 and MLBRICKS_BOLT_V05_ALLOW_JIT=1')
    torch.backends.cuda.matmul.allow_tf32=False
    all_scenarios = scenarios('full')
    results=[]
    for idx in [int(x.strip()) for x in a.cases.split(',')]:
        config=all_scenarios[idx-1]
        print('\nCASE',idx,config,flush=True)
        try:
            record=run_case(config)
            results.append(record)
            print('  WORST logits (native, hybrid)',record['native_worst_logit_abs'],
                  record['hybrid_worst_logit_abs'],flush=True)
            print('  FIRST failing tokens',
                  record['first_native_failure']['generated_token_index'] if record['first_native_failure'] else None,
                  record['first_hybrid_failure']['generated_token_index'] if record['first_hybrid_failure'] else None,flush=True)
        except Exception as exc:
            print('  ERROR',repr(exc),flush=True)
            results.append({'config':config,'error':repr(exc)})
    path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(results, indent=2))
    print('\nSaved:',path)
    return 1 if any('error' in r for r in results) else 0

if __name__=='__main__':
    raise SystemExit(main())
