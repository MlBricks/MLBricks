# Backward-compatibility module alias.
# Canonical implementation: mlbricks.esa.trainer
import sys as _sys
from .esa import trainer as _impl

_sys.modules[__name__] = _impl
