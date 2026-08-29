from __future__ import annotations

import sys
import types

from .core import SOUP, soup

__version__ = "0.1.0a3"
__all__ = ["SOUP", "soup"]


class _CallableModule(types.ModuleType):
    """Allow the intended API: ``from mlbricks import soup; soup(...)``."""

    def __call__(self, *args, **kwargs):
        return soup(*args, **kwargs)


sys.modules[__name__].__class__ = _CallableModule
