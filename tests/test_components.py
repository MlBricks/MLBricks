import torch
import torch.nn as nn


def test_top_level_component_imports_and_aliases():
    from mlbricks import (
        FFN,
        Embedding,
        LMHead,
        Linear,
        LayerNorm,
        RMSNorm,
        Residual,
        ffn,
        embedding,
        embeddings,
        lmhead,
        linear,
        layernorm,
        rmsnorm,
        residual,
    )

    assert ffn is FFN
    assert embedding is Embedding
    assert embeddings is Embedding
    assert lmhead is LMHead
    assert linear is Linear
    assert layernorm is LayerNorm
    assert rmsnorm is RMSNorm
    assert residual is Residual


def test_components_are_pytorch_native():
    from mlbricks import Embedding, FFN, LMHead, Linear

    embedding = Embedding(31, 12)
    ffn = FFN(12, 24)
    lm_head = LMHead(12, 31)
    linear = Linear(12, 7)

    assert isinstance(embedding, nn.Embedding)
    assert isinstance(lm_head, nn.Linear)
    assert isinstance(linear, nn.Linear)

    ids = torch.randint(0, 31, (2, 5))
    hidden = embedding(ids)
    output = ffn(hidden)
    logits = lm_head(output)
    assert logits.shape == (2, 5, 31)


def test_lmhead_weight_tying():
    from mlbricks import Embedding, LMHead

    embedding = Embedding(41, 16)
    head = LMHead(16, 41, tie_to=embedding)
    assert head.weight is embedding.weight


def test_ffn_default_matches_explicit_pytorch_path():
    from mlbricks import FFN

    torch.manual_seed(3)
    layer = FFN(8, 20, activation="gelu", dropout=0.0, bias=True).eval()
    x = torch.randn(2, 4, 8)
    expected = layer.proj(torch.nn.functional.gelu(layer.fc(x)))
    torch.testing.assert_close(layer(x), expected)


def test_swiglu_style_ffn():
    from mlbricks import FFN

    layer = FFN(8, 16, activation="silu", gated=True)
    x = torch.randn(2, 3, 8, requires_grad=True)
    y = layer(x)
    assert y.shape == x.shape
    y.sum().backward()
    assert x.grad is not None


def test_norms_and_residual():
    from mlbricks import LayerNorm, RMSNorm, Residual

    x = torch.randn(2, 3, 8)
    update = torch.randn_like(x)
    assert LayerNorm(8)(x).shape == x.shape
    assert RMSNorm(8)(x).shape == x.shape
    torch.testing.assert_close(Residual()(x, update), x + update)


def test_esa_model_uses_public_pytorch_components():
    from mlbricks import Embedding, ESAModel, ESAModelConfig, FFN, LMHead

    model = ESAModel(
        ESAModelConfig(
            vocab_size=37,
            block=8,
            n_layer=1,
            head=2,
            embd=8,
            dropout=0.0,
            precision="fp32",
            training_compile=False,
        ),
        device="cpu",
    )
    assert isinstance(model.wte, Embedding)
    assert isinstance(model.wpe, Embedding)
    assert isinstance(model.blocks[0].mlp, FFN)
    assert isinstance(model.lm_head, LMHead)
