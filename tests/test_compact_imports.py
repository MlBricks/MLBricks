def test_compact_imports():
    from mlbricks import ESA, FFN, Embedding, LMHead, esa, ffn, embeddings, lmhead

    assert esa is ESA
    assert ffn is FFN
    assert embeddings is Embedding
    assert lmhead is LMHead
