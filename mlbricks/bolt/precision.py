"""Hardware-native precision qualification for BOLT automatic mixed precision.

PyTorch's torch.cuda.is_bf16_supported() defaults to including BF16 emulation.
That check is not suitable for choosing a fast default on Tesla T4 (SM 7.5),
where BF16 tensor allocation may work but native BF16 Tensor Cores do not.
"""
from __future__ import annotations

import torch


def native_bf16_supported(device: torch.device | str | None = None) -> bool:
    """Whether automatic BOLT precision should prefer BF16 on this device.

    For NVIDIA CUDA use native BF16 Tensor Core architecture (SM 8.0+),
    not emulated storage support. ROCm retains PyTorch's BF16 support probe.
    Explicit precision='bf16' is still handled by the existing policy.
    """
    if not torch.cuda.is_available():
        return False
    if torch.version.hip is not None:
        return bool(torch.cuda.is_bf16_supported(including_emulation=False))
    return torch.cuda.get_device_capability(device)[0] >= 8
