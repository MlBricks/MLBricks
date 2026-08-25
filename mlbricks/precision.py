# Backward-compatibility module alias.
# Canonical implementation: mlbricks.esa.precision
import sys as _sys
from .esa import precision as _impl

_sys.modules[__name__] = _impl
