import torch

import mlbricks
from mlbricks import SOUP, soup


def test_soup_top_level_exports_and_forward():
    model = SOUP(dim=16, width=24, depth=1, mixer="esa", backend="pytorch", precision="fp32")
    model = model.cpu()
    x = torch.randn(2, 5, 16)
    y = model(x)
    assert y.shape == x.shape
    assert mlbricks.soup is soup


def test_soup_layerwise_mixer_selection():
    model = soup(
        dim=16, width=[24, 20], depth=2,
        mixer=["esa", "bolt"], ffn=["saffn", "ffn"],
        backend="pytorch", precision="fp32",
    ).cpu()
    x = torch.randn(1, 4, 16)
    y = model(x)
    assert y.shape == x.shape
    assert model.mixer_names == ["esa", "bolt"]
