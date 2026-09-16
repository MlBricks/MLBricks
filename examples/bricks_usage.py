import torch
from mlbricks import Bricks, Brick, ESA, Bolt, FFN, StateAwareFFN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Bricks(
    vocab_size=4096, dim=256, context=512,
    layers=[
        # Inside a composed Bricks model, let the parent own device placement.
        Brick(
            mixer=ESA(embd=256, head=4, device=None),
            ffn=StateAwareFFN(256),
            dim=256,
        ),
        Brick(
            mixer=Bolt(256, 4, latent_dim=32),
            ffn=FFN(256, 1024),
            position="rope",
            dim=256,
        ),
    ],
).to(device)

ids = torch.randint(0, 4096, (1, 16), device=device)
print(model(ids).shape)
