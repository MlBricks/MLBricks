"""Regression for versionless parameters introduced by inference_mode deepcopies."""
import copy

import torch
from mlbricks import BoltModel
from mlbricks.bolt.tensor_versions import inference_safe_version


def test_inference_safe_version_returns_none_only_for_inference_tensors():
    regular = torch.nn.Parameter(torch.randn(3))
    assert isinstance(inference_safe_version(regular), int)
    with torch.inference_mode():
        inferred = torch.nn.Parameter(torch.randn(3))
        assert torch.is_inference(inferred)
        assert inference_safe_version(inferred) is None
        assert isinstance(inference_safe_version(regular), int)


def test_deepcopy_under_inference_mode_preserves_optimized_bolt_forward():
    torch.manual_seed(5507)
    model = BoltModel(vocab_size=127, dim=48, heads=3, layers=1,
                      context=32, bolt_mode='quality', bolt_cache='tight',
                      position='none', backend='pytorch', dropout=0).eval()
    model.enable_bolt_optimized_inference(True)
    ids = torch.randint(0, 127, (1, 8))
    with torch.inference_mode():
        inferred = copy.deepcopy(model)
        assert all(torch.is_inference(p) for p in inferred.parameters())
        baseline = model(ids)
        observed = inferred(ids)
        torch.testing.assert_close(baseline, observed, rtol=1e-5, atol=1e-5)
        # Second forward tests reuse of cached prepacked projections/AVQ gain.
        torch.testing.assert_close(baseline, inferred(ids), rtol=1e-5, atol=1e-5)
