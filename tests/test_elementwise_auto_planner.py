import time

import torch

from mlbricks import EXECUTION_PLANNER, SOUP
from mlbricks.planner import MLBricksExecutionPlanner


def test_qualify_operator_once_benchmarks_then_freezes_fastest_route():
    planner = MLBricksExecutionPlanner()
    x = torch.randn(2, 4, 8)

    class Owner:
        pass

    owner = Owner()

    def fast():
        time.sleep(0.0005)
        return x

    def slow():
        time.sleep(0.003)
        return x

    route = planner.qualify_operator_once(
        owner,
        "unit_element",
        x,
        {"native": slow, "pytorch": fast},
        requested_backend="auto",
        native_available=True,
        training=False,
        default_auto="native",
        warmup=0,
        trials=3,
        switch_margin=0.0,
    )
    assert route == "pytorch"
    assert planner.owner_routes(owner) == {("unit_element", False): "pytorch"}

    # Flip which callable is faster. A frozen element must not re-benchmark or
    # switch because transient runtime load changes later.
    route_again = planner.qualify_operator_once(
        owner,
        "unit_element",
        x,
        {"native": fast, "pytorch": slow},
        requested_backend="auto",
        native_available=True,
        training=False,
        default_auto="native",
        warmup=0,
        trials=1,
        switch_margin=0.0,
    )
    assert route_again == "pytorch"


def test_qualified_routes_are_independent_per_element_owner():
    planner = MLBricksExecutionPlanner()
    x = torch.randn(2, 4, 8)

    class Owner:
        pass

    first = Owner()
    second = Owner()

    def fast():
        time.sleep(0.0005)
        return x

    def slow():
        time.sleep(0.003)
        return x

    first_route = planner.qualify_operator_once(
        first,
        "same_op",
        x,
        {"native": fast, "pytorch": slow},
        native_available=True,
        training=False,
        warmup=0,
        trials=3,
        switch_margin=0.0,
    )
    second_route = planner.qualify_operator_once(
        second,
        "same_op",
        x,
        {"native": slow, "pytorch": fast},
        native_available=True,
        training=False,
        warmup=0,
        trials=3,
        switch_margin=0.0,
    )

    assert first_route == "native"
    assert second_route == "pytorch"
    assert planner.owner_routes(first) == {("same_op", False): "native"}
    assert planner.owner_routes(second) == {("same_op", False): "pytorch"}


def test_frozen_auto_routes_are_owned_per_element():
    EXECUTION_PLANNER.clear()
    x = torch.randn(2, 4, 8)

    class Owner:
        pass

    first = Owner()
    second = Owner()
    route1 = EXECUTION_PLANNER.select_operator_once(
        first,
        "unit_element",
        x,
        requested_backend="auto",
        native_available=True,
        training=False,
        default_auto="native",
    )
    route2 = EXECUTION_PLANNER.select_operator_once(
        second,
        "unit_element",
        x,
        requested_backend="auto",
        native_available=True,
        training=False,
        default_auto="pytorch",
        extra=("different-element",),
    )

    assert route1 == "native"
    assert route2 == "pytorch"
    assert EXECUTION_PLANNER.owner_routes(first) != EXECUTION_PLANNER.owner_routes(second)


def test_soup_keeps_backend_policy_per_mixer_element():
    model = SOUP(
        dim=16,
        width=24,
        depth=2,
        mixer=["esa", "bolt"],
        ffn=["ffn", "ffn"],
        backend="auto",
        precision="fp32",
        memory_dim=8,
        fusion_hidden=32,
    )

    model.set_backend("pytorch")
    assert str(model.layers[0].mixer.backend) == "pytorch"
    assert str(model.layers[1].mixer.backend) == "pytorch"

    model.set_backend("auto")
    rows = model.element_backends()
    mixers = [row for row in rows if row["kind"] == "mixer"]
    assert len(mixers) == 2
    assert all(row["requested"] == "auto" for row in mixers)
