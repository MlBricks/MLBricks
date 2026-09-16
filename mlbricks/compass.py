# Backward-compatibility module alias.
# Canonical implementation: mlbricks.esa.compass
import sys as _sys
from .esa import compass as _impl

_sys.modules[__name__] = _impl
