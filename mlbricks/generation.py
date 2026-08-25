# Backward-compatibility module alias.
# Canonical implementation: mlbricks.esa.generation
import sys as _sys
from .esa import generation as _impl

_sys.modules[__name__] = _impl
