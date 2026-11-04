"""Modern BOLT model API. CUDA selected automatically when present."""
import torch
from mlbricks import BoltConfig, BoltModel, Bolt, BoltStateMemory

config = BoltConfig(
    vocab_size=50257, context=2048, layers=6, dim=384, heads=12,
    profile='standard', state_memory=True, state_dim=198,
    backend='auto', device='auto', precision='auto',
)
model = BoltModel(config)
input_ids = torch.randint(0, config.vocab_size, (1, 32))
logits = model(input_ids)  # inputs moved to model.device automatically
print('BOLT device:', model.device, 'logits:', logits.shape)

# Disable AVQ explicitly: same compact attention, no state memory allocation.
attention_only_cfg = BoltConfig(
    vocab_size=50257, context=2048, layers=6, dim=384, heads=12,
    profile='standard', state_memory=False,
)
attention_only = BoltModel(attention_only_cfg)
assert attention_only_cfg.state_memory is False
assert all(block.attn._modern.avq is None for block in attention_only.blocks)

# Smaller cache profile with the same state memory.
compact = Bolt(embed_dim=384,num_heads=12,profile='compact',device='auto')
print('Compact layer device:', next(compact.parameters()).device)
