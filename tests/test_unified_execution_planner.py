from __future__ import annotations

import torch

from mlbricks import (
    EXECUTION_PLANNER,
    MLBricksExecutionPlanner,
    Vesa,
    VisionBolt,
)
from mlbricks.esa.planner import EXECUTION_PLANNER as ESA_PLANNER
from mlbricks.planner import AUTO_OPERATOR_DEFAULTS


def setup_function():
    EXECUTION_PLANNER.clear()


def test_esa_and_library_share_one_planner_instance():
    assert ESA_PLANNER is EXECUTION_PLANNER
    assert isinstance(EXECUTION_PLANNER, MLBricksExecutionPlanner)


def test_initial_operator_policy_matches_qualified_routes():
    assert AUTO_OPERATOR_DEFAULTS["vision_scan"] == "pytorch"
    assert AUTO_OPERATOR_DEFAULTS["bolt_full"] == "pytorch"
    assert AUTO_OPERATOR_DEFAULTS["vision_position_2d"] == "native"
    assert AUTO_OPERATOR_DEFAULTS["ffnbrick_state"] == "native"
    assert AUTO_OPERATOR_DEFAULTS["ffnbrick_micro"] == "native"
    assert AUTO_OPERATOR_DEFAULTS["rescontroller"] == "native"
    assert AUTO_OPERATOR_DEFAULTS["elastic_linear"] == "pytorch"


def test_auto_route_is_shape_mode_and_operator_aware():
    x = torch.randn(2, 96, 64)
    route_scan = EXECUTION_PLANNER.select_operator(
        "vision_scan", x, requested_backend="auto", native_available=True,
        native_supports_training=True, training=False,
    )
    route_pos = EXECUTION_PLANNER.select_operator(
        "vision_position_2d", x, requested_backend="auto", native_available=True,
        native_supports_training=True, training=False,
    )
    assert route_scan == "pytorch"
    assert route_pos == "native"
    assert len(EXECUTION_PLANNER.operator_cache) == 2


def test_benchmark_result_overrides_heuristic():
    x = torch.randn(2, 96, 64)
    winner = EXECUTION_PLANNER.record_operator_benchmark(
        "bolt_full", x, {"native": 0.5, "pytorch": 0.8}, training=False
    )
    assert winner == "native"
    assert EXECUTION_PLANNER.select_operator(
        "bolt_full", x, requested_backend="auto", native_available=True,
        native_supports_training=True, training=False,
    ) == "native"


def test_training_safety_keeps_inference_only_component_on_pytorch():
    x = torch.randn(2, 16, 32)
    route = EXECUTION_PLANNER.select_operator(
        "ffnbrick_state", x, requested_backend="auto", native_available=True,
        native_supports_training=False, training=True,
    )
    assert route == "pytorch"


def test_explicit_backends_are_never_overridden():
    x = torch.randn(2, 16, 32)
    assert EXECUTION_PLANNER.select_operator(
        "vision_scan", x, requested_backend="pytorch", native_available=True,
        native_supports_training=True, training=False,
    ) == "pytorch"
    assert EXECUTION_PLANNER.select_operator(
        "vision_scan", x, requested_backend="native", native_available=True,
        native_supports_training=True, training=False,
    ) == "native"


def test_model_execution_plan_reports_planner_routes():
    x = torch.randn(2, 64, 32)
    EXECUTION_PLANNER.select_operator(
        "vision_position_2d", x, requested_backend="auto", native_available=True,
        native_supports_training=True, training=False,
    )
    vesa = Vesa(image_size=16, patch_size=4, dim=32, depth=1, num_classes=3,
                engine="Serpentine", backend="pytorch")
    bolt = VisionBolt(image_size=16, patch_size=4, dim=32, depth=1, heads=4,
                      latent_dim=8, num_classes=3, engine="ViT", backend="pytorch")
    assert vesa.execution_plan().planner_routes >= 1
    assert bolt.execution_plan().planner_routes >= 1


def test_operator_decision_report_contains_reason():
    x = torch.randn(1, 32, 16)
    EXECUTION_PLANNER.select_operator(
        "rescontroller", x, requested_backend="auto", native_available=True,
        native_supports_training=False, training=False,
    )
    rows = EXECUTION_PLANNER.operator_decisions()
    assert len(rows) == 1
    assert rows[0]["route"] == "native"
    assert str(rows[0]["reason"]).startswith("heuristic:")


def test_component_local_auto_route_cache_is_sticky_and_revision_safe(monkeypatch):
    planner = MLBricksExecutionPlanner()
    x = torch.randn(2, 32, 16)

    class Owner:
        pass

    owner = Owner()
    calls = {"count": 0}
    original = planner.select_operator

    def counted(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(planner, "select_operator", counted)

    first = planner.select_operator_cached(
        owner, "rescontroller", x, requested_backend="auto",
        native_available=True, native_supports_training=False, training=False,
        extra=(16,),
    )
    second = planner.select_operator_cached(
        owner, "rescontroller", x, requested_backend="auto",
        native_available=True, native_supports_training=False, training=False,
        extra=(16,),
    )
    assert first == second == "native"
    assert calls["count"] == 1

    # Explicit calibration changes the global winner and bumps the revision.
    winner = planner.record_operator_benchmark(
        "rescontroller", x, {"native": 0.8, "pytorch": 0.2},
        training=False, extra=(16,),
    )
    assert winner == "pytorch"

    third = planner.select_operator_cached(
        owner, "rescontroller", x, requested_backend="auto",
        native_available=True, native_supports_training=False, training=False,
        extra=(16,),
    )
    assert third == "pytorch"
    assert calls["count"] == 2


def test_bolt_full_auto_skips_native_wrapper_when_cached_route_is_pytorch(monkeypatch):
    from mlbricks import Bolt
    import mlbricks.vision_native as vision_native

    def should_not_run(*args, **kwargs):
        raise AssertionError("native Bolt wrapper should be skipped for cached PyTorch route")

    monkeypatch.setattr(vision_native, "bolt_full", should_not_run)
    model = Bolt(32, 4, latent_dim=8, dropout=0.0, backend="auto",
                           native_full_sequence=True).eval()
    x = torch.randn(2, 16, 32)
    with torch.no_grad():
        y = model(x)
    assert y.shape == x.shape
    decisions = [r for r in EXECUTION_PLANNER.operator_decisions()
                 if len(r["key"]) > 1 and r["key"][1] == "bolt_full"]
    assert decisions
    assert decisions[-1]["route"] == "pytorch"
