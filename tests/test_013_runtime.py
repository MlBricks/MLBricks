
import time
import torch
import pytest

from mlbricks.elasticbit import ElasticBitConfig
from mlbricks.benchmark import TrainingIntervalTimer


def test_elasticbit_runtime_manifest_roundtrip():
    cfg = ElasticBitConfig(bits=4, group_size=32, runtime="packed", cache_dequantized=False)
    restored = ElasticBitConfig.from_manifest(cfg.to_manifest())
    assert restored.runtime == "packed"
    assert restored.cache_dequantized is False


def test_elasticbit_runtime_validation():
    with pytest.raises(ValueError, match="runtime"):
        ElasticBitConfig(runtime="unknown")


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


def test_short_prefill_route_avoids_fused(monkeypatch):
    import mlbricks.native as native
    monkeypatch.setattr(native, "fused_enabled_for", lambda tensor: True)
    monkeypatch.delenv("MLBRICKS_FUSED_READOUT", raising=False)
    native.EXECUTION_PLANNER.route_cache.clear()
    short = torch.empty(1, 256, 3 * 64)
    larger = torch.empty(4, 256, 3 * 64)
    assert native.should_use_fused_readout(short, 64, 16) is False
    assert native.should_use_fused_readout(larger, 64, 16) is True
