# Backward-compatibility module alias.
# Canonical implementation: mlbricks.esa.boost
import sys as _sys
from .esa import boost as _impl

_sys.modules[__name__] = _impl
