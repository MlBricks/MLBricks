from __future__ import annotations

import inspect

import pytest
import torch

from mlbricks import ElasticBit
from mlbricks.elasticbit import stage2f


def test_optimize_public_api_is_explicit_and_small():
    signature = inspect.signature(ElasticBit.optimize)
    assert list(signature.parameters) == ["model", "force"]
    assert signature.parameters["force"].default is False


def test_stage2f_notebook_constants_are_frozen():
    assert stage2f.S2F_M_VALUES == (1, 16, 128, 512)
    assert stage2f.S2F_ITERS == {1: 240, 16: 120, 128: 45, 512: 14}
    assert stage2f.S2F_ROUNDS == 5
    assert stage2f.S2F_MAX_GROUP_REL_L2 == 2e-2

    assert stage2f.S2F_TOP_K == 3
    assert stage2f.S2F_FULL_LAYER_MARGIN == 0.02
    assert stage2f.S2F_FULL_ITERS == {1: 60, 16: 40, 128: 14, 512: 4}
    assert stage2f.S2F_FULL_ROUNDS == 3

    assert stage2f.S2F_FINAL_ITERS == {1: 100, 16: 70, 128: 20, 512: 6}
    assert stage2f.S2F_FINAL_ROUNDS == 5
    assert stage2f.S2F_MAX_FULL_REL_L2 == 2e-2
    assert stage2f.S2F_FINAL_DEPLOY_MARGIN == 0.01


def test_stage2f_candidate_sets_match_notebook():
    assert stage2f.S2F_QKV_CANDIDATES == (
        "eb_separate",
        "dense_separate",
        "dense_packed",
        "dense_packed_contiguous",
        "q_plus_kv",
        "qk_plus_v",
        "qv_plus_k",
    )
    assert stage2f.S2F_QKV_M1_CANDIDATES == (
        "dense_separate_mv",
        "dense_packed_mv",
        "q_plus_kv_mv",
    )
    assert stage2f.S2F_GATEUP_CANDIDATES == (
        "eb_separate",
        "dense_separate",
        "dense_packed",
        "dense_packed_contiguous",
    )
    assert stage2f.S2F_GATEUP_M1_CANDIDATES == (
        "dense_separate_mv",
        "dense_packed_mv",
    )


def test_optimize_rejects_uncompressed_model_before_any_planning():
    model = torch.nn.Linear(4, 4)
    with pytest.raises(ValueError, match=r"compress\(\) or ElasticBit.load\(\)"):
        ElasticBit.optimize(model)
