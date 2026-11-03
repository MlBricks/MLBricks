"""Standalone BOLT model using the official API."""
import torch
from mlbricks import BoltModel, BoltConfig

config = BoltConfig(vocab_size=512, context=128, layers=2, dim=64, heads=4,
                    profile="standard", state_memory=True, device="auto")
model = BoltModel(config)
tokens = torch.randint(0, config.vocab_size, (1, 8))
print(model(tokens).shape)
