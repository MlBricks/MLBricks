"""Unified MLBricks model lifecycle API.

Use ``mlbricks.save`` and ``mlbricks.load`` from the package root in normal
code. This module exists only as the implementation namespace, not as an ESA
alias.
"""
from .lifecycle import compile, generate, inspect, load, predict, quantize, save

__all__ = ["save", "load", "inspect", "predict", "generate", "compile", "quantize"]
