"""GPU acceptance test: FP32 training master -> auto FP16 graph mirror.

Only the FP16 graph/eager comparison is expected to be bitwise exact on tested
T4. FP32 reference is intentionally NOT claimed to match FP16 inference.

Run from repo root:
  MLBRICKS_AUTO_CUDA=1 PYTHONPATH=. python examples/mlbricks_default_fp32_master_t4_audit.py --prompt 128 --tokens 16 --layers 2 --dim 96 --heads 6
"""
import argparse
import copy
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from mlbricks import BoltModel, AttentionModel
from mlbricks.bolt.inference_mirror import _tensor_signature
from examples.bolt_v05_production_audit import (
    eager_trace as bolt_eager_trace, graph_trace as bolt_graph_trace,
    compare as bolt_compare,
)
from examples.mlbricks_auto_inference_gpu_audit import (
    qkv_eager_trace, qkv_graph_trace, qkv_compare,
)


def audit(name, args):
    torch.cuda.empty_cache()
    torch.manual_seed(2026 if name!='bolt32' else 5505)
    common={'vocab_size':2048, 'dim':args.dim, 'heads':args.heads,
            'layers':args.layers, 'context':args.prompt+args.tokens+8}
    if name=='qkv':
        master=AttentionModel(**common)
    else:
        master=BoltModel(**common, bolt_mode='quality',
                         bolt_cache='tight' if name=='bolt16' else 'normal',
                         avq_dim=198, position='none', dropout=0,
                         backend='pytorch')
    assert next(master.parameters()).device.type=='cuda'
    assert next(master.parameters()).dtype==torch.float32
    assert master.training
    prompt=torch.randint(0,2048,(1,args.prompt),dtype=torch.long)
    prompt_cuda=prompt.to('cuda')
    original_state={k:v.clone() for k,v in master.state_dict().items()}
    original_keys=tuple(original_state)
    original_sig=_tensor_signature(master)
    # Build a completely independent FP16 eager reference BEFORE generating.
    eager=copy.deepcopy(master).to(dtype=torch.float16).eval()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
        if name=='qkv':
            reference=qkv_eager_trace(eager,prompt_cuda,args.tokens)
        else:
            reference=bolt_eager_trace(eager,prompt_cuda,args.tokens)
        # No .cuda(), .half(), .eval(), explicit temperature or config flags.
        generated=master.generate(prompt,args.tokens)
    info=master.inference_info()
    assert info['last_execution']=='strict_cuda_graph',info
    assert info['precision_policy']=='fp32_master_fp16_inference_mirror',info
    assert info['inference_dtype']=='torch.float16',info
    assert master.training
    assert next(master.parameters()).dtype==torch.float32
    assert '_mlbricks_fp16_inference_mirror' not in master._modules
    assert tuple(master.state_dict())==original_keys
    assert all(torch.equal(v,master.state_dict()[key])
               for key,v in original_state.items())
    assert _tensor_signature(master)==original_sig
    assert torch.equal(generated,reference['tokens']), f'{name} token mismatch'
    mirror=master.__dict__['_mlbricks_fp16_inference_mirror'][1]
    mirror_id=id(mirror)
    assert next(mirror.parameters()).dtype==torch.float16
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
        if name=='qkv':
            actual=qkv_graph_trace(mirror,prompt_cuda,args.tokens)
            delta=qkv_compare(reference,actual)
            full_exact=bool(delta['passed'])
        else:
            mirror.enable_bolt_strict_inference(True)
            actual=bolt_graph_trace(mirror,prompt_cuda,args.tokens)
            delta=bolt_compare(reference,actual,mode='graph')
            full_exact=bool(delta['max_logit_delta']==0
                            and delta['max_avq_state_delta']==0
                            and delta['max_kv_delta']==0
                            and delta['final_state_delta']==0
                            and delta['token_mismatches']==0)
    assert full_exact, f'{name} full-state/logit FP16 parity mismatch: {delta}'
    # Non-mutating second generation must reuse identical mirror parameters.
    with torch.no_grad():
        again=master.generate(prompt,args.tokens)
    assert torch.equal(generated,again)
    assert id(master.__dict__['_mlbricks_fp16_inference_mirror'][1])==mirror_id
    # Direct optimizer-style inplace parameter update invalidates signature.
    # Keep original mirror alive so id-reuse cannot create a false positive.
    with torch.no_grad():
        next(master.parameters()).add_(0.01)
    assert _tensor_signature(master)!=original_sig
    with torch.no_grad():
        master.generate(prompt,args.tokens)
    assert id(master.__dict__['_mlbricks_fp16_inference_mirror'][1])!=mirror_id
    assert next(master.parameters()).dtype==torch.float32
    master.train()
    assert '_mlbricks_fp16_inference_mirror' not in master.__dict__
    # Do not compare FP32 to FP16 logits, as the cast is not bitwise-parity safe.
    report={
        'architecture':name,
        'gpu':torch.cuda.get_device_name(),
        'master_dtype':'float32',
        'inference_dtype':'float16',
        'dispatch':info['last_execution'],
        'full_fp16_eager_vs_graph_parity':full_exact,
        'public_output_matches_fp16_eager':True,
        'master_parameters_unchanged_during_generate':True,
        'checkpoint_keys_unchanged':True,
        'mirror_reused_when_unchanged':True,
        'mirror_rebuilt_after_weight_change':True,
        'mirror_cleared_on_train':True,
        'delta':{k:v for k,v in delta.items() if isinstance(v,(bool,int,float,str,type(None)))},
        'passed':True,
    }
    print(json.dumps(report,indent=2),flush=True)
    return report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--prompt',type=int,default=128)
    parser.add_argument('--tokens',type=int,default=16)
    parser.add_argument('--layers',type=int,default=2)
    parser.add_argument('--dim',type=int,default=96)
    parser.add_argument('--heads',type=int,default=6)
    parser.add_argument('--output',type=str,default='mlbricks_v3_t4_audit.json')
    a=parser.parse_args()
    assert a.dim%a.heads==0 and a.tokens>=2
    if not torch.cuda.is_available():
        raise RuntimeError('GPU unavailable; enable Tesla T4 on Kaggle')
    if torch.cuda.get_device_capability()!=(7,5):
        raise RuntimeError('Strict graph is currently validated only on Tesla T4')
    reports=[]
    for arch in ['bolt16','bolt32','qkv']:
        reports.append(audit(arch,a))
    Path(a.output).write_text(json.dumps({'passed':all(r['passed'] for r in reports),
                                        'reports':reports},indent=2))
    print('SUCCESS: BOLT16, BOLT32, QKV full FP16 strict-graph parity',flush=True)

if __name__=='__main__':main()
