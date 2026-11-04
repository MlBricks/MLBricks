"""BOLT auto should select hardware-native BF16, not emulated BF16."""
import torch
from mlbricks.bolt.precision import native_bf16_supported


def test_auto_precision_rejects_emulated_bf16_on_t4(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda device=None: (7, 5))
    # PyTorch 2.x may report emulated BF16 as supported by the default API.
    monkeypatch.setattr(torch.cuda, 'is_bf16_supported', lambda including_emulation=True: including_emulation)
    assert torch.cuda.is_bf16_supported() is True
    assert native_bf16_supported('cuda:0') is False


def test_auto_precision_accepts_native_bf16_on_ampere(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda device=None: (8, 0))
    assert native_bf16_supported('cuda:1') is True


def test_auto_precision_returns_false_without_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    assert native_bf16_supported() is False


def test_auto_precision_qualifies_device_independently(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    seen = []
    def capability(device=None):
        seen.append(device)
        return (7, 5) if str(device) == 'cuda:0' else (9, 0)
    monkeypatch.setattr(torch.cuda, 'get_device_capability', capability)
    assert native_bf16_supported(torch.device('cuda:0')) is False
    assert native_bf16_supported(torch.device('cuda:1')) is True
    assert len(seen) == 2
