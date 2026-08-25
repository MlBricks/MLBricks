"""Build a small PyTorch-native MLBricks component pipeline."""

import torch

from mlbricks import ESA, FFN, embeddings, lmhead, rmsnorm


vocab_size = 50_257
hidden_size = 384

embedding = embeddings(vocab_size, hidden_size)
esa = ESA(embd=hidden_size, head=6, precision="fp32", device="cpu")
feed_forward = FFN(hidden_size, 4 * hidden_size, activation="gelu")
norm = rmsnorm(hidden_size)
output_head = lmhead(hidden_size, vocab_size, tie_to=embedding)

input_ids = torch.randint(0, vocab_size, (2, 32))
x = embedding(input_ids)
x = x + esa(norm(x))
x = x + feed_forward(norm(x))
logits = output_head(norm(x))

print(logits.shape)
