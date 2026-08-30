import torch

import mlbricks as mlb


def _mixed_model():
    dim = 16
    return mlb.Bricks(
        vocab_size=32,
        dim=dim,
        context=8,
        layers=[
            mlb.Brick(
                mixer=mlb.ESA(embd=dim, head=2, backend="pytorch", precision="fp32"),
                ffn=mlb.FFN(dim, 32),
                dim=dim,
            ),
            mlb.Brick(
                mixer=mlb.Bolt(dim, 2, latent_dim=4, backend="pytorch"),
                ffn=mlb.FFN(dim, 32),
                dim=dim,
            ),
        ],
        backend="pytorch",
    )


def test_package_level_save_load_mixed_model(tmp_path):
    torch.manual_seed(7)
    model = _mixed_model().eval()
    ids = torch.randint(0, 32, (2, 6))
    ref = model(ids)

    path = mlb.save(model, tmp_path / "mixed")
    loaded = mlb.load(path, device="cpu").eval()
    got = loaded(ids)

    assert isinstance(loaded, mlb.Bricks)
    assert isinstance(loaded.layers[0].mixer, mlb.ESA)
    assert isinstance(loaded.layers[1].mixer, mlb.Bolt)
    assert torch.allclose(ref, got, atol=0, rtol=0)
    info = mlb.inspect(path)
    assert info["format"] == "mlbricks.model"
    assert info["architecture"].endswith(".Bricks")


def test_generic_trainer_fit_and_resume(tmp_path):
    torch.manual_seed(8)
    model = _mixed_model()
    ids = torch.randint(0, 32, (2, 6))
    targets = torch.randint(0, 32, (2, 6))

    trainer = mlb.Trainer(
        model,
        optimizer="adamw",
        lr=1e-3,
        checkpoint_dir=tmp_path / "checkpoints",
        save_last=True,
    )
    summary = trainer.fit([(ids, targets)], steps=1)
    assert summary["step"] == 1

    resumed = mlb.Trainer.resume(tmp_path / "checkpoints" / "last", device="cpu")
    assert resumed.state.step == 1
    assert isinstance(resumed.model, mlb.Bricks)
    assert resumed.optimizer is not None


def test_old_esa_lifecycle_methods_are_removed():
    assert not hasattr(mlb.ESAModel, "save")
    assert not hasattr(mlb.ESAModel, "load")
    import mlbricks.esa as esa_package
    assert not hasattr(esa_package, "Trainer")
