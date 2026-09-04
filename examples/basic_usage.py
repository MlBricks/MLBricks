import torch
from mlbricks import ESA

x = torch.randn(2, 128, 64)

# Default: backend="auto" with automatic Compass planning.
layer = ESA(embd=64, head=4, precision="fp32", device=None)
y = layer(x)
print(y.shape, layer.backend, layer.compass)

# Advanced/manual Compass override.
manual = ESA(embd=64, head=4, compass=32, precision="fp32", device=None)
y_manual = manual(x)
print(y_manual.shape, manual.compass)
