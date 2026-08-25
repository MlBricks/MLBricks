from __future__ import annotations

import inspect

from mlbricks.model import ESAModel


def test_generation_fast_defaults() -> None:
    signature = inspect.signature(ESAModel.generate)
    assert signature.parameters["compile"].default is True
    assert signature.parameters["compile_mode"].default == "default"


def test_compile_generation_fast_default() -> None:
    signature = inspect.signature(ESAModel.compile_generation)
    assert signature.parameters["mode"].default == "default"
