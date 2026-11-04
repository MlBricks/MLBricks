"""Direct original Fast-EsaQ ESA under MLBricks — CPU runnable."""
import torch
from mlbricks import ESA, ESAAdapter

native = ESA(embd=24, head=6, backend="pytorch", device="cpu", variant="current")
print("native class:", type(native).__name__)
print("native checkpoint keys:", list(native.state_dict())[:4])
x = torch.randn(2, 4, 24)
with torch.inference_mode():
    y, state = native(x)
    step, state = native.decode_step(x[:, -1], state)
assert y.shape == (2, 4, 24) and step.shape == (2, 1, 24)
print("direct native:", y.shape, step.shape, state.shape)

# Existing MLBricks ESAModel, VESA, SOUP can retain their checkpoint/API bridge:
adapter = ESAAdapter(embd=24, head=6, backend="pytorch", device="cpu")
with torch.inference_mode():
    old_y = adapter(x)
assert old_y.shape == x.shape
print("compatibility adapter:", old_y.shape)
