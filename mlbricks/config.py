# Backward-compatibility module alias.
# Canonical implementation: mlbricks.esa.config
import sys as _sys
from .esa import config as _impl

_sys.modules[__name__] = _impl
