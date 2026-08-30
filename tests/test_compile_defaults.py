from __future__ import annotations

import inspect

from mlbricks import ESAModel, ESAModelConfig


def test_training_compile_mode_default() -> None:
    config = ESAModelConfig(vocab_size=128)
    assert config.training_compile is True
    assert config.training_compile_mode == "default"


def test_prefill_native_default_and_compatibility_kwargs() -> None:
    signature = inspect.signature(ESAModel.prefill)
    assert signature.parameters["engine"].default == "thunder"
    # Retained as backward-compatible no-op kwargs in 1.0.x.
    assert signature.parameters["compile_mode"].default == "default"
    assert signature.parameters["fullgraph"].default is False
    assert signature.parameters["dynamic"].default is True


def test_compile_generation_mode_default() -> None:
    signature = inspect.signature(ESAModel.compile_generation)
    assert signature.parameters["mode"].default == "default"


def test_generate_compile_mode_default() -> None:
    signature = inspect.signature(ESAModel.generate)
    assert signature.parameters["compile"].default is True
    assert signature.parameters["compile_mode"].default == "default"
