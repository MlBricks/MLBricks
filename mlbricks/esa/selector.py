"""Public ESA mode selector without wrapping or changing checkpoint key names.

* retention (default): validated FastESA half-life ESA, outputs (y, state)
* legacy: Thunder/Lightning MLBricks ESA, outputs y

Constructed objects retain their original concrete Python types. Historical
ESAModel, VESA, and SOUP wrappers continue importing the legacy class directly.
"""
from __future__ import annotations

import torch
from torch import nn

from .layer import ESA as LegacyESA
from .half_life import CurrentESA as RetentionESA


class _ESASelectorMeta(type(nn.Module)):
    def __call__(cls, embd: int = 384, head: int | None = None,
                 backend: str = "auto", device: str | torch.device | None = None,
                 *, mode: str = "retention", **kwargs):
        if cls is not ESA:
            return super().__call__(embd=embd, head=head, backend=backend,
                                    device=device, **kwargs)
        if not isinstance(mode, str):
            raise ValueError("ESA mode must be 'retention' or 'legacy'")
        selected = mode.strip().lower()
        if selected == "retention":
            heads = 6 if head is None else head
            return RetentionESA(embd=embd, head=heads, backend=backend,
                                device=device, **kwargs)
        if selected == "legacy":
            heads = 4 if head is None else head
            return LegacyESA(embd=embd, head=heads, backend=backend,
                             device=device, **kwargs)
        raise ValueError("ESA mode must be 'retention' or 'legacy'")

    def __instancecheck__(cls, instance):
        if cls is ESA:
            return isinstance(instance, (RetentionESA, LegacyESA))
        return super().__instancecheck__(instance)


class ESA(nn.Module, metaclass=_ESASelectorMeta):
    """Construct a concrete ESA implementation, defaulting to retention.

    ``ESA(embd=384, head=6, backend='auto')`` creates retention ESA.
    ``ESA(embd=384, head=4, mode='legacy')`` creates Thunder legacy ESA.

    The return signatures differ: retention returns ``(y, state)`` from
    ``forward(x, state=None)``; legacy returns only ``y`` from ``forward(x)``.
    State dict keys and checkpoint types are left untouched. This class is a
    constructor selector; the result is a RetentionESA or LegacyESA instance.
    """

    def forward(self, *args, **kwargs):
        raise RuntimeError("ESA is a constructor selector; its concrete engine handles forward")


__all__ = ["ESA", "RetentionESA", "LegacyESA"]
