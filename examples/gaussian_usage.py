import torch
from mlbricks import Gaussian

model = Gaussian(
    vocab_size=50_257, context=1024, layers=6, dim=384, heads=6,
    latent_dim=32, position="rope", ffn="saffn", residual="rescontroller",
)
ids = torch.randint(0, 50_257, (1, 32))
logits = model(ids)
print(logits.shape)
print(model.backend_report())
