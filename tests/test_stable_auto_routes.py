import torch

from mlbricks import EXECUTION_PLANNER


class Owner:
    pass


def test_auto_route_is_frozen_on_owner_until_explicit_reset():
    EXECUTION_PLANNER.clear()
    owner = Owner()
    x = torch.randn(2, 8)

    first = EXECUTION_PLANNER.select_operator_once(
        owner, "unit_test_op", x, requested_backend="auto",
        native_available=True, training=False, default_auto="native",
    )
    assert first == "native"

    # Change the global planner decision after the component has started.
    EXECUTION_PLANNER.set_operator_route(
        "unit_test_op", x, "pytorch", training=False, reason="test-change"
    )
    still_first = EXECUTION_PLANNER.select_operator_once(
        owner, "unit_test_op", x, requested_backend="auto",
        native_available=True, training=False, default_auto="native",
    )
    assert still_first == "native"

    # An explicit reset is the only way auto is allowed to choose again.
    EXECUTION_PLANNER.clear_owner_routes(owner)
    replanned = EXECUTION_PLANNER.select_operator_once(
        owner, "unit_test_op", x, requested_backend="auto",
        native_available=True, training=False, default_auto="native",
    )
    assert replanned == "pytorch"
