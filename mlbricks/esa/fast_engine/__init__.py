"""MLBricks ESA engine. Half-life ESA is current; BaseESA preserves history.

The existing ``fastesa`` import is retained for backwards compatibility.
Internal module names are now core/base/scan/decode/graph/model/config.
"""
from .core import (ESAModel, BaseESAModel,
                   make_esa, make_model, forward_tokens, load_checkpoint,
                   normalize_variant, normalize_scan,
                   FastESA, QualityESA, FastESAModel, QualityESAModel)
from .public_api import ESA, CurrentESA, BaseESA, save_esa_checkpoint, load_esa_checkpoint, detect_esa_variant
from .config import ESAConfig
from .decode import prepare_decode, prepare_quality_decode
from .graph import GraphDecoder, CUDAGreedyGraph
from .scan import (parallel_scan, short_scan, sequential_scan,
                   hierarchical_scan, adaptive_scan, prepare_scan,
                   prepare_short_scan, select_scan_backend,
                   prepare_swqr, prepare_swqr_short, select_swqr_backend)
__all__ = [
  "ESA", "CurrentESA", "BaseESA", "ESAModel", "BaseESAModel", "ESAConfig",
  "save_esa_checkpoint", "load_esa_checkpoint", "detect_esa_variant",
  "make_esa", "make_model", "forward_tokens", "load_checkpoint",
  "normalize_variant", "normalize_scan", "parallel_scan",
  "short_scan", "sequential_scan", "hierarchical_scan", "adaptive_scan",
  "prepare_scan", "prepare_short_scan", "select_scan_backend",
  "prepare_decode", "GraphDecoder",
  # Deprecated compatibility aliases, retained until a major release.
  "FastESA", "QualityESA", "FastESAModel", "QualityESAModel",
  "CUDAGreedyGraph", "prepare_quality_decode", "prepare_swqr",
  "prepare_swqr_short", "select_swqr_backend",
]
__version__ = "0.4.0"
