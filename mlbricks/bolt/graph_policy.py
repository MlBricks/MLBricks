"""Per-layer CUDA Graph preference. No capture occurs merely by constructing a layer.

The model/session-level controller owns CUDA Graph capture, stable buffers and
cache positions.  An individual BOLT layer can therefore safely use graph='auto'
inside arbitrary hybrid PyTorch models without forcing a separate graph.
"""
from __future__ import annotations


def normalize_graph_policy(graph: str | bool) -> str:
    if isinstance(graph, bool):
        return "on" if graph else "off"
    if not isinstance(graph, str):
        raise TypeError("graph must be 'auto', 'on', 'off', True or False")
    value = graph.strip().lower()
    value = {"true": "on", "false": "off"}.get(value, value)
    if value not in {"auto", "on", "off"}:
        raise ValueError("graph must be 'auto', 'on', 'off', True or False")
    return value


def layer_graph_permission(graph: str | bool, backend: str, *, training: bool) -> bool:
    """Permission for a qualified outer graph controller, not capture eligibility."""
    policy = normalize_graph_policy(graph)
    return policy != "off" and backend != "pytorch" and not training
