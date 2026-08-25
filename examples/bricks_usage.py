import torch
from mlbricks import Bricks, Brick, ESA, Bolt, FFN, StateAwareFFN

model = Bricks(
    vocab_size=4096, dim=256, context=512,
    layers=[
        Brick(mixer=ESA(embd=256, head=4), ffn=StateAwareFFN(256), dim=256),
        Brick(mixer=Bolt(256, 4, latent_dim=32), ffn=FFN(256, 1024), position="rope", dim=256),
    ],
)
ids = torch.randint(0, 4096, (1, 16))
print(model(ids).shape)
