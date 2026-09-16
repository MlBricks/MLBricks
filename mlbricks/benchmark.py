# Backward-compatibility module alias.
# Canonical implementation: mlbricks.esa.benchmark
import sys as _sys
from .esa import benchmark as _impl

_sys.modules[__name__] = _impl
