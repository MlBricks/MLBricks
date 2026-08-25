import torch

from mlbricks import (
    ESAModel,
    ESAModelConfig,
    Trainer,
)

config = ESAModelConfig(
    vocab_size=50257,
    block=512,
    n_layer=6,
    head=6,
    embd=384,
    backend="thunder",
    precision="fp16",
)

model = ESAModel(
    config
).cuda()

trainer = Trainer(
    model,
    checkpoint_dir="checkpoints",
    save_every=1000,
    save_at=[
        5000,
        10000,
        20000,
    ],
    save_best=True,
    save_last=True,
    keep_last_n=3,
)

# Save:
model.save(
    "my_model"
)

# Load:
model = ESAModel.load(
    "my_model",
    device="cuda",
)

# Generate:
# text = model.generate(
#     prompt="Once upon a time",
#     tokenizer=tokenizer,
#     max_new_tokens=4096,
#     temperature=0.75,
#     top_k=40,
# )
