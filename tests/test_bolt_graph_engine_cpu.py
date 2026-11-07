"""CPU contract tests for the opt-in graph engine (GPU qualification separate)."""
from __future__ import annotations

import pytest
import torch
from torch import nn

from mlbricks.bolt.graph_engine import BoltGraphEngine, GraphCaptureRecovered


class FakeCache:
    def __init__(self, capacity, length=5):
        self.k = torch.zeros((1, 1, capacity, 1))
        self.v = torch.zeros_like(self.k)
        self.length = length


class FakeLM(nn.Module):
    kind = "qkv"  # eager fallback

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Identity()])
        self.weight = nn.Parameter(torch.tensor(0.), requires_grad=False)

    def decode_one(self, token, caches, position):
        assert caches[0].length == position
        caches[0].length += 1
        return (token + 1) % 50257, None

    def prefill(self, prompt, capacity):
        return prompt[:, -1:], [FakeCache(capacity, prompt.shape[1])]


class FakeBoltModel(nn.Module):
    """Public BoltModel's decode_step/prefill method contract, on CPU."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.), requires_grad=False)
        self._prepared_generation = None

    def prepare_generation(self, *, batch_size, max_context):
        assert batch_size == 1
        self._prepared_generation = {"caches": [FakeCache(max_context, 0)]}

    def prefill(self, ids, *, cache_buffers):
        cache_buffers[0].length = ids.shape[1]
        logits = torch.zeros((1, ids.shape[1], 16))
        logits[:, -1, int(ids[0, -1])] = 1
        return logits, cache_buffers

    def decode_step(self, token, caches, *, position):
        assert caches[0].length == position
        caches[0].length += 1
        logits = torch.zeros((1, 1, 16))
        logits[0, 0, int((int(token.item()) + 1) % 16)] = 1
        return logits, caches


@pytest.mark.parametrize("n", [0, 1, 10, 100, 1201])
def test_variable_length_and_one_token_yields(n):
    engine = BoltGraphEngine(FakeLM().eval(), [FakeCache(1230)], torch.tensor([[7]]), 5)
    result = list(engine.generate(n))
    assert result == [(7 + i) % 50257 for i in range(n)]
    assert engine.position == 5 + max(n - 1, 0)
    assert engine.status()["pointer_stable"]
    assert not engine.using_graph


def test_eos_before_second_token():
    engine = BoltGraphEngine(FakeLM().eval(), [FakeCache(50)], torch.tensor([[7]]), 5)
    assert list(engine.generate(20, eos_id=7)) == [7]
    assert engine.position == 5


def test_eos_later():
    engine = BoltGraphEngine(FakeLM().eval(), [FakeCache(50)], torch.tensor([[7]]), 5)
    assert list(engine.generate(20, eos_id=9)) == [7, 8, 9]
    assert engine.position == 7


def test_capacity_rejected_before_mutation():
    engine = BoltGraphEngine(FakeLM().eval(), [FakeCache(7)], torch.tensor([[7]]), 5)
    with pytest.raises(ValueError, match="capacity"):
        list(engine.generate(4))
    assert engine.position == 5
    assert not engine._sent_first


def test_unsupported_strict_graph_fails():
    with pytest.raises(RuntimeError, match="unavailable"):
        BoltGraphEngine(FakeLM().eval(), [FakeCache(7)], torch.tensor([[7]]), 5,
                        strict_graph=True)


def test_from_prompt_eager_qkv_protocol():
    model = FakeLM().eval()
    prompt = torch.tensor([[4, 5, 7]], dtype=torch.long)
    engine = BoltGraphEngine.from_prompt(model, prompt, max_new_tokens=4)
    assert list(engine.generate(4)) == [7, 8, 9, 10]


def test_from_prompt_boltmodel_eager_fallback():
    model = FakeBoltModel().eval()
    prompt = torch.tensor([[4, 5, 7]], dtype=torch.long)
    engine = BoltGraphEngine.from_prompt(model, prompt, max_new_tokens=4)
    assert list(engine.generate(4)) == [7, 8, 9, 10]
    assert not engine.using_graph
    assert engine.status()["fallback_reason"]


def test_capture_snapshot_rollback_preserves_addresses():
    cache = FakeCache(50)
    eng = BoltGraphEngine(FakeLM().eval(), [cache], torch.tensor([[7]]), 5)
    before = eng._pointer_signature()
    snap = eng._save_state()
    cache.k.fill_(25)
    cache.v.fill_(-3)
    cache.length = 18
    eng.input_token.fill_(99)
    eng.pos_gpu.fill_(18)
    eng.position = 18
    eng._restore_state(snap)
    assert cache.length == 5
    assert not cache.k.any() and not cache.v.any()
    assert eng.position == 5 and int(eng.input_token.item()) == 7
    assert eng._pointer_signature() == before


def test_module_export():
    from mlbricks.bolt import BoltGraphEngine as public
    assert public is BoltGraphEngine


def test_negative_length():
    engine = BoltGraphEngine(FakeLM().eval(), [FakeCache(50)], torch.tensor([[7]]), 5)
    with pytest.raises(ValueError):
        list(engine.generate(-1))


def test_prompt_capacity_preflight():
    with pytest.raises(ValueError, match="max_context"):
        BoltGraphEngine.from_prompt(FakeLM().eval(), torch.tensor([[1, 2, 3]]),
                                    max_new_tokens=10, max_context=5)


def test_real_public_boltmodel_from_prompt_eager_fallback_cpu():
    from mlbricks import BoltModel, BoltConfig
    config = BoltConfig(vocab_size=64, dim=32, heads=4, layers=2,
                        context=24, device='cpu', backend='pytorch',
                        precision='fp32', position='none')
    model = BoltModel(config).eval().requires_grad_(False)
    prompt = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    engine = BoltGraphEngine.from_prompt(model, prompt,
                                         max_new_tokens=4, max_context=10)
    results = list(engine.generate(4))
    assert len(results) == 4 and all(0 <= v < 64 for v in results)
    assert not engine.using_graph
    assert engine.position == 7
