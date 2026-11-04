"""Run: python examples/fast_esaq_migration.py

Fast-EsaQ v0.4.0 is the only MLBricks ESA mixer.
"""
import torch
from mlbricks import ESA, ESAModel, ESAModelConfig


def main() -> None:
    torch.manual_seed(12)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    precision = torch.float16 if device.type == 'cuda' else torch.float32
    layer = ESA(embd=48, head=6, backend='auto', precision='fp16' if device.type == 'cuda' else 'fp32', cumpass=16, parallel_chunk=128, cuda_warps=8, device=device).to(dtype=precision).eval()
    x = torch.randn(2, 8, 48, device=device, dtype=precision)
    with torch.no_grad():
        output = layer(x)
        prefix, state = layer.prefill(x[:, :7])
        last, next_state = layer.decode_step(x[:, 7], state)
        joined = torch.cat((prefix, last[:, None, :]), dim=1)
    print('Engine:', layer.layer.__class__.__name__)
    print('Layer output:', tuple(output.shape), 'cache:', tuple(next_state.shape), next_state.dtype)
    print('Prefill + decode max difference:', (joined.float() - output.float()).abs().max().item())
    model = ESAModel(ESAModelConfig(
        vocab_size=128, embd=48, head=6, n_layer=2, block=32,
        precision='fp16' if device.type == 'cuda' else 'fp32',
        training_compile=False,
    ), device=str(device))
    ids = torch.randint(0, 128, (2, 7), device=device)
    logits, loss = model(ids, ids)
    print('MLBricks language model:', tuple(logits.shape), 'training loss:', float(loss.detach()))


if __name__ == '__main__':
    main()
