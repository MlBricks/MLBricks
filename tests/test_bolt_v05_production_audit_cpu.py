"""CPU coverage of the production-audit gate (does not claim GPU parity)."""
import copy
from types import SimpleNamespace
import torch
from examples import bolt_v05_production_audit as audit


def make_trace():
    cache = SimpleNamespace(
        c=torch.zeros((1,2,5,16),dtype=torch.float32),
        state=torch.zeros((1,198),dtype=torch.float32),
        length=4,
    )
    return {
        'tokens':torch.tensor([[11,12,13,14,15]]),
        'logits':[torch.ones((1,17)),torch.ones((1,17))],
        'states':[[torch.zeros((1,198))],[torch.zeros((1,198))]],
        'caches':[cache],
    }


def test_scenarios_cover_bolt16_bolt32_batch_and_long_context():
    cases=audit.scenarios('full')
    assert {'tight','normal'}=={s['cache'] for s in cases}
    assert max(s['batch'] for s in cases)>=4
    assert max(s['prompt'] for s in cases)>=2048


def test_compare_rejects_logits_state_token_and_kv_corruption():
    base=make_trace()
    result=audit.compare(base,copy.deepcopy(base),mode='eager')
    assert result['passed']
    for corrupt in ('token','logit','state','kv','nan'):
        cand=copy.deepcopy(base)
        if corrupt=='token': cand['tokens'][0,-1]=999
        if corrupt=='logit': cand['logits'][0][0,0]=1.5
        if corrupt=='state': cand['states'][0][0][0,0]=.3
        if corrupt=='kv': cand['caches'][0].c[0,0,0,0]=1.0
        if corrupt=='nan': cand['states'][0][0][0,0]=float('nan')
        verdict=audit.compare(base,cand,mode='eager')
        assert not verdict['passed'],(corrupt,verdict)


def test_graph_comparison_checks_full_active_kv_region():
    ref=make_trace()
    observed=copy.deepcopy(ref)
    observed['caches'][0].length=2  # graph replay doesn't update Python length
    observed['caches'][0].c[0,0,3,0]=.9
    verdict=audit.compare(ref,observed,mode='graph')
    assert not verdict['passed']
    assert verdict['max_kv_delta']>.8
