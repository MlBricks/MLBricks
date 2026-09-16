from __future__ import annotations

import pytest
import torch

from mlbricks import ESA, ESAConfig
from mlbricks.backends.thunder import thunder_scan
from mlbricks.generation import parse_engine_spec
from mlbricks.planner import AUTO_COMPASS_CANDIDATES, ceil_div


def test_auto_compass_is_public_only_through_esa_layer():
    config = ESAConfig(embd=32, head=4)
    assert config.compass == "auto"

    layer = ESA(
        embd=32,
        head=4,
        precision="fp32",
        device=None,
    )
    assert layer.compass == "auto"
    assert layer.layer.compass == "auto"


def test_low_level_thunder_scan_requires_manual_integer():
    A = torch.ones(1, 8, 2, 4)
    B = torch.zeros_like(A)
    with pytest.raises(ValueError, match="requires an explicit"):
        thunder_scan(A, B)


def test_low_level_thunder_scan_rejects_public_auto_string():
    A = torch.ones(1, 8, 2, 4)
    B = torch.zeros_like(A)
    with pytest.raises(ValueError, match=r"ESA\(.*compass='auto'"):
        thunder_scan(A, B, compass="auto")


def test_thunder_auto_engine_name_is_not_public():
    with pytest.raises(ValueError, match="not a public engine name"):
        parse_engine_spec("thunder_auto")


def test_auto_candidate_set_matches_validated_compasses():
    assert AUTO_COMPASS_CANDIDATES == (8, 16, 32, 64)
    assert ceil_div(65_536, 64) == 1024


def test_planned_backward_uses_chunked_kernel_for_hierarchical_context(monkeypatch):
    import mlbricks.native as native

    class FakeNative:
        def __init__(self):
            self.calls = 0

        def thunder_scan_backward_chunked(self, A, states, grad_out, compass):
            self.calls += 1
            assert compass == 32
            return torch.full_like(A, 3.0), torch.full_like(A, 4.0)

        def thunder_reverse_prepare(self, *args, **kwargs):
            raise AssertionError("hierarchical backward must not materialize reverse tensors")

        def thunder_reverse_finish(self, *args, **kwargs):
            raise AssertionError("hierarchical backward must not use reverse_finish")

    fake = FakeNative()
    monkeypatch.setattr(native, "_C", fake)

    A = torch.ones(1, 8, 2, 4)
    states = torch.zeros_like(A)
    grad_out = torch.ones_like(A)

    class Ctx:
        compass = 32
        # Deliberately keep the old metadata shape to model a hierarchical case.
        hierarchy_depth = 2
        direct_budget = 1024
        saved_tensors = (A, states)

    grad_A, grad_B, grad_compass, grad_budget = native._PlannedThunderScanFn.backward(
        Ctx(), grad_out
    )

    assert fake.calls == 1
    assert torch.equal(grad_A, torch.full_like(A, 3.0))
    assert torch.equal(grad_B, torch.full_like(A, 4.0))
    assert grad_compass is None
    assert grad_budget is None
