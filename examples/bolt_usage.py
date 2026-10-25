"""BOLT preset examples — default is quality AVQ-198, normal cache 32D."""
from mlbricks import Bolt, BoltModel

# Default: BOLT-32 with AVQ-198 memory per layer.
bolt = Bolt(embed_dim=384, num_heads=12)  # PyTorch-style: quality+normal by default
# Equivalent positional or legacy-keyword forms:
# Bolt(384, 12)
# Bolt(d_model=384, num_heads=12)
print("BOLT backend:", bolt.resolved_backend())  # PyTorch SDPA for BOLT/AVQ-198

# Same quality, tighter attention cache (16 instead of 32 features per head).
tight_quality = Bolt(384, 12, mode="quality", cache="tight")

# Faster inference: no AVQ recurrent state or AVQ projections.
fast = Bolt(384, 12, mode="fast", cache="normal")

# Compact fast profile combines 16D cache with no AVQ.
fast_tight = Bolt(384, 12, mode="fast", cache="tight")

# Ready-made language model, same default quality+normal preset.
model = BoltModel(vocab_size=50257, context=2048, layers=6, dim=384, heads=12)

# Load OLD original-normalization BOLT checkpoints ONLY with legacy mode.
legacy = Bolt(384, 12, latent_dim=32, mode="legacy")
legacy_model = BoltModel(vocab_size=50257, context=2048, layers=6,
                         dim=384, heads=12, latent_dim=32, bolt_mode="legacy")
