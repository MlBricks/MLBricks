# Backward-compatibility module alias.
# Canonical implementation: mlbricks.esa.model
# Compatibility note: fixed-shape ESA-Lightning decode step; compile mode default remains unchanged.
import sys as _sys
from .esa import model as _impl

_sys.modules[__name__] = _impl
