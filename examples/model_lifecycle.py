import torch
import mlbricks as mlb


config = mlb.ESAModelConfig(
    vocab_size=128,
    block=32,
    n_layer=2,
    head=4,
    embd=64,
    backend="auto",
    precision="fp32",
)
model = mlb.ESAModel(config, device="cpu")

# Any MLBricks architecture uses the same lifecycle.
trainer = mlb.Trainer(
    model,
    optimizer="adamw",
    lr=3e-4,
    checkpoint_dir="checkpoints",
    save_every=100,
)

x = torch.randint(0, 128, (4, 16))
y = torch.randint(0, 128, (4, 16))
train_data = [(x, y)]
trainer.fit(train_data, steps=1)

mlb.save(model, "my_model")
model = mlb.load("my_model", device="cpu")

# Exact interrupted-run resume.
trainer = mlb.Trainer.resume("checkpoints/last", device="cpu")
