from mlbricks import Bolt, BoltModel

# Standalone Bolt attention component. backend="auto" is implicit.
bolt = Bolt(d_model=384, num_heads=6, latent_dim=32, position="rope")

# Ready-made causal LM using Bolt blocks.
model = BoltModel(
    vocab_size=50257,
    context=2048,
    layers=6,
    dim=384,
    heads=6,
    latent_dim=32,
    position="rope",
)
