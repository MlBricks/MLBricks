"""ElasticBit quantization component package."""

from . import core as _core

# Re-export the complete compatibility surface, including private helpers used
# by existing tests/tools such as _pack_unsigned and _unpack_unsigned.
globals().update({
    name: value
    for name, value in vars(_core).items()
    if not name.startswith("__")
})

__all__ = getattr(_core, "__all__", [])
