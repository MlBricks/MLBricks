"""Current half-life ESA in MLBricks (opt-in)."""
import torch
from mlbricks import (ESA, LegacyESA, RetentionESA, make_current_esa_model,
                      current_esa_forward_tokens, DEFAULT_CURRENT_HALF_LIVES)

current = ESA(embd=384, head=6, backend='pytorch', device='cpu')
assert isinstance(current, RetentionESA)
print('Nominal initial half-lives:', current.log_half_life.detach().exp().tolist())
assert tuple(DEFAULT_CURRENT_HALF_LIVES) == (.125, .5, 2, 8, 32, 64)
x = torch.randn(1, 12, 384)
y, state = current.prefill(x)
print('Prefill:', y.shape, 'State:', state.shape)
y_step, state = current.decode_step(torch.randn(1,384),state)
print('Decode:', y_step.shape, 'State:', state.shape)
model = make_current_esa_model(vocab_size=256, dim=384, heads=6, layers=2,
                               scan='reference')
logits, states = current_esa_forward_tokens(model, torch.randint(0,256,(1,8)))
print('LM:', logits.shape, 'State layers:',len(states))

legacy = ESA(embd=384, head=4, mode='legacy', backend='pytorch', device='cpu')
assert isinstance(legacy, LegacyESA)
print('Legacy shape:', legacy(torch.randn(1, 12, 384)).shape)
