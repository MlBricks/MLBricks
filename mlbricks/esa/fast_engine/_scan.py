"""Compatibility import. Use ``fastesa.scan`` for new code."""
import sys as _sys
from importlib import import_module as _import_module
_impl = _import_module(".scan", __package__)
_sys.modules[__name__] = _impl
