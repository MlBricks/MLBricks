"""Position transform defined by the *model user*, not implemented by BOLT."""
import torch
from torch import nn


class ExternalQK(nn.Module):
    """Deterministic token-local Q/K scale, valid for prefill and cached decode."""
    def forward(self, x, *, start_pos=0):
        pos = torch.arange(start_pos, start_pos + x.size(-2),
                           device=x.device, dtype=torch.float32)
        weights = (1.0 + 0.08 * torch.sin(pos * 0.37)).to(x.dtype)
        return x * weights.view(1, 1, -1, 1)
