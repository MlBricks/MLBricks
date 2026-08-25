# Backward-compatibility module alias.
# Canonical implementation: mlbricks.esa.constants
import sys as _sys
from .esa import constants as _impl

_sys.modules[__name__] = _impl
