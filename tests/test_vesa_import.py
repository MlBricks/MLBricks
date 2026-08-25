import mlbricks


def test_top_level_vesa_api():
    from mlbricks import Vesa, VesaConfig

    cfg = VesaConfig()
    model = Vesa(cfg)

    assert cfg.backend == "auto"
    assert model.config is cfg
    assert isinstance(model, mlbricks.vesa.VisionESAClassifier)


def test_vesa_default_constructor_is_auto():
    from mlbricks import Vesa

    model = Vesa()
    assert model.config.backend == "auto"


def test_advanced_vesa_namespace_is_kept():
    from mlbricks import vesa

    assert vesa.__version__ == "1.0.0"
    assert vesa.Vesa is mlbricks.Vesa
    assert vesa.VesaConfig is mlbricks.VesaConfig
    assert vesa.AR is vesa.ESAARModel
    assert vesa.Classifier is vesa.VisionESAClassifier
    assert vesa.Diffusion is vesa.ESADiffusionModel


def test_attention_baselines_not_in_public_vesa_api():
    from mlbricks import vesa

    assert "AttentionARModel" not in vesa.__all__
    assert "AttentionDiffusionModel" not in vesa.__all__


def test_vesa_direct_keyword_constructor():
    from mlbricks import Vesa

    model = Vesa(
        image_size=32,
        num_classes=10,
        dim=192,
        depth=4,
    )

    assert model.config.image_size == 32
    assert model.config.num_classes == 10
    assert model.config.dim == 192
    assert model.config.depth == 4
    assert model.config.backend == "auto"


def test_vesa_rejects_config_plus_keyword_overrides():
    import pytest
    from mlbricks import Vesa, VesaConfig

    with pytest.raises(TypeError, match="either a config object"):
        Vesa(VesaConfig(), depth=4)
