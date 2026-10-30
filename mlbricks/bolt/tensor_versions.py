"""Safe tensor version stamps for graph/prepacked inference weights.

PyTorch inference tensors have no version counter. Cache invalidation can use
pointer + dtype/device for those tensors; in-place weight mutations of an
inference tensor cannot be detected. Build/rebuild the model outside
``torch.inference_mode`` when weight mutation tracking is required.
"""
import torch


def inference_safe_version(tensor: torch.Tensor) -> int | None:
    """Return the tracked version, or None for versionless inference tensors."""
    return None if torch.is_inference(tensor) else int(tensor._version)
