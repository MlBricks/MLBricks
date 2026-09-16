# Backward-compatibility module alias.
# Canonical implementation: mlbricks.esa.native
import sys as _sys
from .esa import native as _impl

_sys.modules[__name__] = _impl
