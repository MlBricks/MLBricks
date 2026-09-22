import copy

import torch
from torch import nn

from mlbricks import ElasticBit
from mlbricks.elasticbit.stage2g import (
    STAGE2G_NOTEBOOK_CELL_SHA256,
    Stage2PackedLinear,
    UniversalStage2CoreCompiler,
)


EXPECTED_NOTEBOOK_HASHES = {
    "core": "61c4b679c03adcea1e1d0fac46431c96827c6dc651a7b6923340f2fc49c974cd",
    "transactional_hotfix": "bb70cd67ede7e5914df7e771bad7fa5300058c9c33a4f3b933e8b9e1da5c8287",
    "public_reduce": "b321f3c4ebe3da0eb9bf9fd72b6dd983464f3c24b3a60489767e4751085b74a6",
}


class _BranchMerge(nn.Module):
    def __init__(self, d: int = 16):
        super().__init__()
        self.a = nn.Linear(d, d)
        self.b = nn.Linear(d, d)
        self.c = nn.Linear(d, d)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.a(x) + self.b(x) + self.c(x) + 10.0)


class _BetweenBoundaries(nn.Module):
    def __init__(self, d: int = 16, mid: int = 12):
        super().__init__()
        self.pre = nn.Linear(d, d)
        self.left = nn.ReLU()
        self.a = nn.Linear(d, mid)
        self.b = nn.Linear(mid, d)
        self.right = nn.SiLU()

    def forward(self, x):
        h = self.left(self.pre(x))
        h = self.a(h)
        h = self.b(h)
        return self.right(h)


class _PureFanout(nn.Module):
    def __init__(self, d: int = 16):
        super().__init__()
        self.a = nn.Linear(d, d)
        self.b = nn.Linear(d, d // 2)
        self.c = nn.Linear(d, d // 4)

    def forward(self, x):
        return self.a(x), self.b(x), self.c(x)


class _ResultantFanout(nn.Module):
    def __init__(self, d: int = 16):
        super().__init__()
        self.p = nn.Linear(d, d)
        self.a = nn.Linear(d, d)
        self.b = nn.Linear(d, d // 2)
        self.c = nn.Linear(d, d // 4)

    def forward(self, x):
        h = self.p(x)
        return self.a(h), self.b(h), self.c(h)


class _ExternalConsumer(nn.Module):
    def __init__(self, d: int = 16):
        super().__init__()
        self.a = nn.Linear(d, d)
        self.b = nn.Linear(d, d)
        self.act = nn.ReLU()

    def forward(self, x):
        h = self.a(x)
        reduced_candidate = self.act(self.b(h))
        return reduced_candidate, h


def _flat(value):
    if torch.is_tensor(value):
        return [value]
    if isinstance(value, (tuple, list)):
        out = []
        for item in value:
            out.extend(_flat(item))
        return out
    raise TypeError(type(value))


def _assert_close(reference, candidate):
    ref = _flat(reference)
    got = _flat(candidate)
    assert len(ref) == len(got)
    for a, b in zip(ref, got):
        assert torch.allclose(a, b, rtol=2e-3, atol=2e-3)


def test_stage2g_reference_hashes_are_frozen():
    assert STAGE2G_NOTEBOOK_CELL_SHA256 == EXPECTED_NOTEBOOK_HASHES


def test_public_reduce_branch_merge_matches_frozen_rules():
    torch.manual_seed(20260922)
    model = _BranchMerge().eval()
    x = torch.randn(4, 16)

    with torch.inference_mode():
        reference = model(x)

    reduced = ElasticBit.reduce(model, x)
    report = reduced._elasticbit_reduce_report

    with torch.inference_mode():
        candidate = reduced(x)

    _assert_close(reference, candidate)

    assert report["verified"] is True
    assert report["no_op"] is False
    assert [r["kind"] for r in report["rewrites"]] == ["boundary"]
    assert report["rewrites"][0]["old_params"] == 816
    assert report["rewrites"][0]["new_params"] == 272

    modules = dict(reduced.named_modules())
    assert "a" not in modules
    assert "b" not in modules
    assert "c" not in modules
    assert not any(isinstance(m, Stage2PackedLinear) for m in reduced.modules())


def test_public_reduce_between_boundaries_composes_linears():
    torch.manual_seed(20260922)
    model = _BetweenBoundaries().eval()
    x = torch.randn(4, 16)

    with torch.inference_mode():
        reference = model(x)

    reduced = ElasticBit.reduce(model, x)
    report = reduced._elasticbit_reduce_report

    with torch.inference_mode():
        candidate = reduced(x)

    _assert_close(reference, candidate)
    assert report["verified"] is True
    assert report["no_op"] is False
    assert any(r["kind"] == "boundary" and r["accepted"] for r in report["rewrites"])


def test_public_reduce_pure_same_input_fanout_is_noop():
    torch.manual_seed(20260922)
    model = _PureFanout().eval()
    x = torch.randn(4, 16)

    reduced = ElasticBit.reduce(model, x)
    report = reduced._elasticbit_reduce_report

    assert report["verified"] is True
    assert report["no_op"] is True
    assert report["rewrites"] == []
    assert not any(isinstance(m, Stage2PackedLinear) for m in reduced.modules())

    annotations = report["analysis_before"]["same_input_fanout_annotations"]
    assert annotations
    assert all(item["action"] == "ANNOTATE_ONLY_NO_PACKING" for item in annotations)


def test_public_reduce_resultant_propagation_stays_unpacked():
    torch.manual_seed(20260922)
    model = _ResultantFanout().eval()
    x = torch.randn(4, 16)

    with torch.inference_mode():
        reference = model(x)

    reduced = ElasticBit.reduce(model, x)
    report = reduced._elasticbit_reduce_report

    with torch.inference_mode():
        candidate = reduced(x)

    _assert_close(reference, candidate)

    rewrites = [
        r for r in report["rewrites"]
        if r.get("kind") == "resultant_propagation"
    ]
    assert rewrites
    assert all(r["packed"] is False for r in rewrites)
    assert not any(isinstance(m, Stage2PackedLinear) for m in reduced.modules())


def test_external_consumer_safety_prevents_illegal_elimination():
    torch.manual_seed(20260922)
    model = _ExternalConsumer().eval()
    x = torch.randn(4, 16)

    with torch.inference_mode():
        reference = model(x)

    reduced = ElasticBit.reduce(model, x)

    with torch.inference_mode():
        candidate = reduced(x)

    _assert_close(reference, candidate)
    report = reduced._elasticbit_reduce_report
    assert report["verified"] is True


def test_exact_compiler_defaults_match_notebook():
    torch.manual_seed(20260922)
    model = _BranchMerge().eval()
    x = torch.randn(2, 16)

    compiler = UniversalStage2CoreCompiler(model, (x,))

    assert compiler.max_param_growth == 0.0
    assert compiler.max_resultant_bytes == 512 * 1024**2
    assert compiler.max_passes == 32
