# Copyright 2026 Zameer Hussain and Akhtar Hussain
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# See LICENSE and LICENSING_NOTICE.md; commercial use requires a separate written license.

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Iterable


@dataclass
class CompassResult:
    """Result returned by ``compass()`` for Thunder configuration selection."""

    recommended: int
    best_quality: int
    fastest: int
    rows: list[dict[str, Any]]
    precision: str
    quality_tolerance: float
    recommendation: str

    def summary(self) -> str:
        return self.recommendation

    def to_dataframe(self):
        try:
            import pandas as pd
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("pandas is required for to_dataframe().") from exc
        return pd.DataFrame(self.rows)

    def to_dict(self) -> dict[str, Any]:
        return {
            "recommended": self.recommended,
            "best_quality": self.best_quality,
            "fastest": self.fastest,
            "rows": self.rows,
            "precision": self.precision,
            "quality_tolerance": self.quality_tolerance,
            "recommendation": self.recommendation,
        }


def _as_float(row: dict[str, Any], *names: str) -> float | None:
    for name in names:
        value = row.get(name)
        if value is not None:
            try:
                return float(value)
            except Exception:
                return None
    return None


def _normalise_row(row: dict[str, Any]) -> dict[str, Any]:
    row = dict(row)
    if "ppl" not in row and "val_loss" in row:
        try:
            row["ppl"] = float(math.exp(float(row["val_loss"])))
        except Exception:
            pass
    if "tok_per_sec" not in row:
        for alias in ("tokens_per_second", "tok_s", "throughput", "tokens_sec"):
            if alias in row:
                row["tok_per_sec"] = row[alias]
                break
    return row


def compass(
    *,
    evaluate_fn: Callable[..., dict[str, Any]],
    c_candidates: Iterable[int] = (8, 16, 32, 64),
    precision: str = "fp16",
    quality_tolerance: float = 0.02,
    metric: str = "ppl",
    speed_metric: str = "tok_per_sec",
    **evaluate_kwargs: Any,
) -> CompassResult:
    """Benchmark manual Thunder Compass values and recommend one.

    Normal users should leave ``ESA(compass="auto")`` unchanged. This utility
    is for controlled experiments or deployment tuning. ``evaluate_fn`` is
    called with ``backend="auto"``, ``c=<candidate>`` and ``precision``.
    """
    c_values = tuple(int(c) for c in c_candidates)
    if not c_values:
        raise ValueError("c_candidates must contain at least one value.")
    if any(c <= 0 for c in c_values):
        raise ValueError(f"all c values must be positive integers, got {c_values}")
    if quality_tolerance < 0:
        raise ValueError("quality_tolerance must be >= 0.")

    rows: list[dict[str, Any]] = []
    for c in c_values:
        row = evaluate_fn(
            backend="auto",
            c=c,
            precision=precision,
            **evaluate_kwargs,
        )
        row = _normalise_row(row)
        row.setdefault("backend", "auto")
        row["c"] = c
        rows.append(row)

    def quality_value(row: dict[str, Any]) -> float:
        value = _as_float(row, metric, "ppl", "val_loss")
        if value is None:
            raise ValueError(
                f"Each row must contain {metric!r}, 'ppl', or 'val_loss'. Bad row: {row}"
            )
        return value

    def speed_value(row: dict[str, Any]) -> float:
        value = _as_float(
            row,
            speed_metric,
            "tok_per_sec",
            "tokens_per_second",
            "tok_s",
            "throughput",
            "tokens_sec",
        )
        if value is None:
            raise ValueError(
                f"Each row must contain {speed_metric!r} or a supported speed alias. Bad row: {row}"
            )
        return value

    best_quality_row = min(rows, key=quality_value)
    fastest_row = max(rows, key=speed_value)
    threshold = quality_value(best_quality_row) * (1.0 + quality_tolerance)
    acceptable = [row for row in rows if quality_value(row) <= threshold]
    recommended_row = max(acceptable, key=speed_value)

    for row in rows:
        row["acceptable"] = row in acceptable

    recommended = int(recommended_row["c"])
    best_quality = int(best_quality_row["c"])
    fastest = int(fastest_row["c"])
    recommendation = (
        f"Use compass={recommended}. It is the fastest tested setting within "
        f"{quality_tolerance * 100:.2f}% of the best observed {metric}. "
        f"Best quality={best_quality}; fastest={fastest}."
    )
    return CompassResult(
        recommended=recommended,
        best_quality=best_quality,
        fastest=fastest,
        rows=rows,
        precision=precision,
        quality_tolerance=quality_tolerance,
        recommendation=recommendation,
    )
