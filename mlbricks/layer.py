# Backward-compatibility module alias.
# Canonical implementation: mlbricks.esa.layer
import sys as _sys
from .esa import layer as _impl

_sys.modules[__name__] = _impl
