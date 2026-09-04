from __future__ import annotations

import torch

from mlbricks import ESA, Bolt, Vesa, VisionBolt
from mlbricks.vesa.layers.mixer import ESAMixer


COMMON = dict(
    image_size=8,
    patch_size=2,
    in_channels=3,
    num_classes=5,
    dim=16,
    depth=2,
    heads=4,
    perspective_groups=4,
    vocab_size=37,
    latent_dim=4,
    backend="pytorch",
    position=None,
    scan="cross",
)


def test_vesa_classifier_uses_canonical_esa_core() -> None:
    model = Vesa(engine="ViT", **COMMON)
    mixers = [block.mixer for block in model.blocks]
    assert mixers
    assert all(isinstance(mixer, ESAMixer) for mixer in mixers)
    assert all(isinstance(mixer.core_engine, ESA) for mixer in mixers)
    assert all(mixer.core_engine.layer.qgv.out_features == 3 * COMMON["dim"] for mixer in mixers)


def test_visionbolt_classifier_uses_canonical_bolt_core() -> None:
    model = VisionBolt(engine="ViT", **COMMON)
    mixers = [block.mixer for block in model.engine_model.blocks]
    assert mixers
    assert all(isinstance(mixer, Bolt) for mixer in mixers)


def test_diffusion_engines_use_canonical_core_mixers() -> None:
    vesa = Vesa(engine="Diffusion", **COMMON)
    bolt = VisionBolt(engine="Diffusion", **COMMON)
    assert all(isinstance(block.mixer, ESAMixer) for block in vesa.engine_model.blocks)
    assert all(isinstance(block.mixer.core_engine, ESA) for block in vesa.engine_model.blocks)
    assert all(isinstance(block.mixer, Bolt) for block in bolt.engine_model.blocks)


def test_visual_ar_prefill_decode_matches_full_forward_for_both_cores() -> None:
    prompt = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    for cls in (Vesa, VisionBolt):
        torch.manual_seed(123)
        model = cls(engine="AR", **COMMON).eval()
        engine = model.engine_model
        with torch.no_grad():
            full_prompt = model(prompt)[:, -1]
            prefill_logits, caches = engine.prefill(prompt)
            torch.testing.assert_close(prefill_logits, full_prompt, rtol=1e-4, atol=1e-5)

            token = prefill_logits.argmax(dim=-1)
            extended = torch.cat([prompt, token[:, None]], dim=1)
            full_next = model(extended)[:, -1]
            decode_logits, _ = engine.decode_step(token, prompt.shape[1], caches)
            torch.testing.assert_close(decode_logits, full_next, rtol=1e-4, atol=1e-5)


def test_visual_ar_generation_uses_core_prefill_decode_contract() -> None:
    prompt = torch.tensor([[1, 2, 3]])
    for cls in (Vesa, VisionBolt):
        model = cls(engine="AR", **COMMON).eval()
        generated = model.generate(prompt, 3)
        assert generated.shape == (1, 3)
        if cls is Vesa:
            assert all(isinstance(block.mixer.core_engine, ESA) for block in model.engine_model.blocks)
        else:
            assert all(isinstance(block.mixer, Bolt) for block in model.engine_model.blocks)


def test_canonical_esa_forward_with_state_matches_forward() -> None:
    torch.manual_seed(321)
    engine = ESA(embd=16, head=4, backend="pytorch", precision="fp32", device=None).eval()
    x = torch.randn(2, 7, 16)
    direct = engine(x)
    stateful, state = engine.forward_with_state(x)
    torch.testing.assert_close(stateful, direct, rtol=1e-5, atol=1e-6)
    assert state.shape == (2, 4, 4)


def test_canonical_esa_forward_with_state_continuation_matches_one_pass() -> None:
    torch.manual_seed(654)
    engine = ESA(embd=16, head=4, backend="pytorch", precision="fp32", device=None).eval()
    prefix = torch.randn(2, 5, 16)
    continuation = torch.randn(2, 3, 16)
    _, state = engine.forward_with_state(prefix)
    tail, final_state = engine.forward_with_state(continuation, state=state)
    combined, combined_state = engine.forward_with_state(torch.cat([prefix, continuation], dim=1))
    torch.testing.assert_close(tail, combined[:, -continuation.shape[1]:], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(final_state, combined_state, rtol=1e-5, atol=1e-6)
