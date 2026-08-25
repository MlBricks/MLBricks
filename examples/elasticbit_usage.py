import torch

from mlbricks import ElasticBit, ESAModel, ESAModelConfig

config = ESAModelConfig(vocab_size=50_257, n_layer=6, head=6, embd=384)
model = ESAModel(config, device="cpu").eval()

# One top-level ElasticBit object controls packing and model conversion.
elastic = ElasticBit(bits=4, group_size=128)
elastic.quantize_module(model, include_embeddings=False)

input_ids = torch.randint(0, config.vocab_size, (1, 32))
with torch.no_grad():
    logits = model(input_ids)
print(logits.shape)

# On a native CUDA build, choose the true low-memory bitstream path explicitly:
#   ElasticBit(bits=4, group_size=128, runtime="packed", cache_dequantized=False)
# This avoids constructing a full dequantized weight. The default "auto" policy
# remains speed-first when dequant caching is enabled.
