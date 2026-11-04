"""Native Fast-EsaQ in MLBricks, CPU example."""
import torch
from mlbricks import ESA

mixer = ESA(embd=24, head=6, variant="current", backend="pytorch", device="cpu")
x = torch.randn(2,4,24)
with torch.inference_mode():
    output,state = mixer(x)
    step,state = mixer.decode_step(x[:,-1],state)
print("ESA:",type(mixer).__name__)
print("Output:",output.shape,"Step:",step.shape,"State:",state.shape)
print("Checkpoint keys:",list(mixer.state_dict())[:4])
