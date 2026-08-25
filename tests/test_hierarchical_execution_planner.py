import torch
import pytest

from mlbricks import VisionBolt, EXECUTION_PLANNER
from mlbricks.planner import MLBricksExecutionPlanner
from mlbricks.runtime import apply_execution_route, reset_execution_route


def tiny_model():
    return VisionBolt(
        image_size=8,
        patch_size=4,
        in_channels=3,
        num_classes=3,
        dim=16,
        depth=1,
        heads=4,
        latent_dim=4,
        perspective_groups=4,
        engine="ViT",
        backend="auto",
    )


def test_model_benchmark_cache_selects_fastest_route_cpu():
    planner = MLBricksExecutionPlanner()
    model = tiny_model().eval()
    x = torch.randn(2, 3, 8, 8)
    winner = planner.record_model_benchmark(
        model,
        x,
        {"operator": 2.0, "native": 1.5, "pytorch": 1.0},
    )
    assert winner == "pytorch"
    assert planner.select_model_route(model, x) == "pytorch"
    rows = planner.model_decisions()
    assert len(rows) == 1
    assert rows[0]["reason"].startswith("benchmark:")


def test_prepare_execution_benchmarks_composed_routes_and_keeps_requested_auto():
    EXECUTION_PLANNER.clear()
    model = tiny_model().eval()
    x = torch.randn(2, 3, 8, 8)
    plan = model.prepare_execution(
        x,
        candidates=("operator", "pytorch"),
        warmup=0,
        trials=1,
        force=True,
    )
    assert plan.requested_backend == "auto"
    assert plan.model_route in {"operator", "pytorch"}
    assert plan.model_benchmarked is True
    assert plan.model_benchmarked_routes >= 1
    assert getattr(model, "_mlbricks_model_timings")
    y = model(x)
    assert y.shape == (2, 3)


def test_prepare_execution_reuses_cached_model_route():
    EXECUTION_PLANNER.clear()
    model = tiny_model().eval()
    x = torch.randn(1, 3, 8, 8)
    first = model.prepare_execution(
        x,
        candidates=("operator", "pytorch"),
        warmup=0,
        trials=1,
        force=True,
    )
    timings = dict(model._mlbricks_model_timings)
    second = model.prepare_execution(x, warmup=0, trials=1, force=False)
    assert second.model_route == first.model_route
    assert model._mlbricks_model_timings == timings


def test_prepare_execution_requires_eval_mode():
    model = tiny_model().train()
    x = torch.randn(1, 3, 8, 8)
    with pytest.raises(RuntimeError, match="model.eval"):
        model.prepare_execution(x, warmup=0, trials=1)


def test_explicit_set_backend_clears_prepared_route_state():
    model = tiny_model().eval()
    apply_execution_route(model, "pytorch")
    model._mlbricks_requested_backend = "auto"
    model._mlbricks_model_timings = {"pytorch": 1.0}
    model.set_backend("pytorch")
    plan = model.execution_plan()
    assert plan.requested_backend == "pytorch"
    assert plan.model_route == "pytorch"
    assert plan.model_benchmarked is False


def test_reset_execution_restores_operator_auto():
    model = tiny_model().eval()
    apply_execution_route(model, "pytorch")
    reset_execution_route(model)
    plan = model.execution_plan()
    assert plan.requested_backend == "auto"
    assert plan.model_route == "operator"
