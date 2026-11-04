def test_compact_imports():
    from mlbricks import ESA, FFN, Embedding, LMHead, ffn, embeddings, lmhead
    from mlbricks.esa.fast_engine.public_api import ESA as NativeESA
    assert ESA is NativeESA
    assert ffn is FFN
    assert embeddings is Embedding
    assert lmhead is LMHead
