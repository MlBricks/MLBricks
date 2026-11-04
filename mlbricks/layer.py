"""Direct Fast-EsaQ ESA entry point.

ESA is Fast-EsaQ's original v0.4.0 public implementation, returned without
an intervening nn.Module or decode-step wrapper. For the historical MLBricks
single-output contract use ``ESAAdapter`` explicitly.
"""
from .esa.fast_engine.public_api import ESA, CurrentESA, BaseESA
from .esa.layer import ESA as ESAAdapter
__all__ = ["ESA", "ESAAdapter", "CurrentESA", "BaseESA"]
