
import time
import torch
import pytest

from mlbricks.benchmark import TrainingIntervalTimer


def test_training_timer_excludes_paused_time():
    timer = TrainingIntervalTimer()
    time.sleep(0.01)
    timer.pause()
    before = timer.elapsed
    time.sleep(0.02)
    after = timer.elapsed
    assert after - before < 0.01
    timer.resume()
    time.sleep(0.01)
    assert timer.elapsed > after


def test_legacy_bounded_gate_native_readout_is_not_exposed():
    import mlbricks.native as native
    for name in (
        "should_use_fused_readout",
        "thunder_fused_readout",
        "thunder_forward_hierarchical",
        "lightning_fused_step",
    ):
        assert not hasattr(native, name)
