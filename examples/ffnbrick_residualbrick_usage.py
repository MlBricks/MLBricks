import torch

from mlbricks import ResController, StateAwareFFN

batch, seq, dim, state_dim = 2, 16, 192, 64
x = torch.randn(batch, seq, dim)
esa = torch.randn_like(x)
previous_esa = torch.zeros_like(x)
state = torch.zeros(batch, seq, state_dim)

ffn = StateAwareFFN(
    d_model=dim,
    state_dim=state_dim,
    depth_embedding_dim=32,
    layer_index=0,
    total_layers=6,
)
residual = ResController(update_ratio=0.18)

update, next_state = ffn(x, esa, previous_esa, state)
out = residual(x, update)

print("output:", out.shape)
print("next_state:", next_state.shape)
