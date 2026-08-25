# Copyright 2026 Zameer Hussain and Akhtar Hussain
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# See LICENSE and LICENSING_NOTICE.md; commercial use requires a separate written license.

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Any, Callable, Iterable

import torch

from .model import ESAModel
from ..optim import stabilize_optimizer


@dataclass
class TrainerState:
    step: int = 0
    best_val_loss: float = float("inf")


class Trainer:
    """
    Training/checkpoint manager for ESAModel.

    Supports:
      - save_every=N
      - save_at=[...]
      - save_best=True
      - save_last=True
      - keep_last_n=N
      - exact resume including optimizer/scheduler/scaler and RNG state
    """

    def __init__(
        self,
        model: ESAModel,
        *,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any | None = None,
        scaler: Any | None = None,
        checkpoint_dir: str | Path = "checkpoints",
        save_every: int | None = None,
        save_at: Iterable[int] | None = None,
        save_best: bool = True,
        save_last: bool = True,
        keep_last_n: int | None = 3,
    ):
        self.model = model
        self.optimizer = optimizer
        if self.optimizer is not None:
            # Direct FP16 Adam/AdamW with eps=1e-8 can produce first-step
            # NaNs because epsilon underflows to zero.  Keep external
            # optimizers usable while preserving all other settings.
            stabilize_optimizer(self.optimizer)
        self.scheduler = scheduler
        self.scaler = scaler
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.save_every = int(save_every) if save_every else None
        self.save_at = {int(x) for x in (save_at or [])}
        self.save_best = bool(save_best)
        self.save_last = bool(save_last)
        self.keep_last_n = keep_last_n
        self.state = TrainerState()

    def _rng_state(self) -> dict[str, Any]:
        state = {
            "python": random.getstate(),
            "torch": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            state["cuda"] = torch.cuda.get_rng_state_all()
        return state

    def _restore_rng_state(self, state: dict[str, Any]) -> None:
        if "python" in state:
            random.setstate(state["python"])
        if "torch" in state:
            torch.set_rng_state(state["torch"])
        if "cuda" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda"])

    def save_checkpoint(
        self,
        *,
        step: int | None = None,
        name: str | None = None,
        protected: bool = False,
        extra: dict[str, Any] | None = None,
    ) -> Path:
        """Save a complete training checkpoint."""
        step = int(self.state.step if step is None else step)
        name = name or f"step_{step:06d}"
        path = self.checkpoint_dir / name
        self.model.save(path)

        payload = {
            "step": step,
            "best_val_loss": self.state.best_val_loss,
            "optimizer": self.optimizer.state_dict() if self.optimizer is not None else None,
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            "scaler": self.scaler.state_dict() if self.scaler is not None else None,
            "rng_state": self._rng_state(),
            "protected": bool(protected),
            "extra": extra or {},
        }
        torch.save(payload, path / "training_state.pt")
        return path

    def maybe_save(self, *, step: int, val_loss: float | None = None) -> list[Path]:
        """Apply the configured checkpoint policy for the current training step."""
        self.state.step = int(step)
        saved: list[Path] = []

        exact = step in self.save_at
        periodic = self.save_every is not None and step % self.save_every == 0

        if exact or periodic:
            saved.append(
                self.save_checkpoint(
                    step=step,
                    protected=exact,
                )
            )

        if self.save_best and val_loss is not None and val_loss < self.state.best_val_loss:
            self.state.best_val_loss = float(val_loss)
            saved.append(self.save_checkpoint(step=step, name="best", protected=True))

        self._prune_periodic()
        return saved

    def save_final(self) -> Path | None:
        """Save the final training checkpoint."""
        if not self.save_last:
            return None
        return self.save_checkpoint(step=self.state.step, name="last", protected=True)

    def _prune_periodic(self) -> None:
        if self.keep_last_n is None or self.keep_last_n < 0:
            return

        candidates = []
        for path in self.checkpoint_dir.glob("step_*"):
            state_file = path / "training_state.pt"
            if not state_file.exists():
                continue
            try:
                payload = torch.load(state_file, map_location="cpu", weights_only=False)
                if payload.get("protected", False):
                    continue
                candidates.append((int(payload.get("step", -1)), path))
            except Exception:
                continue

        candidates.sort()
        for _, path in candidates[:-self.keep_last_n] if self.keep_last_n else candidates:
            import shutil
            shutil.rmtree(path, ignore_errors=True)

    def load_checkpoint(
        self,
        path: str | Path,
        *,
        device: str | torch.device | None = None,
        restore_rng: bool = True,
    ) -> "Trainer":
        """Load a checkpoint into the model and available training-state objects."""
        path = Path(path)
        if device is None:
            device = self.model.device

        loaded = ESAModel.load(path, device=device)
        self.model.load_state_dict(loaded.state_dict())

        payload = torch.load(path / "training_state.pt", map_location="cpu", weights_only=False)
        self.state.step = int(payload.get("step", 0))
        self.state.best_val_loss = float(payload.get("best_val_loss", float("inf")))

        if self.optimizer is not None and payload.get("optimizer") is not None:
            self.optimizer.load_state_dict(payload["optimizer"])
        if self.scheduler is not None and payload.get("scheduler") is not None:
            self.scheduler.load_state_dict(payload["scheduler"])
        if self.scaler is not None and payload.get("scaler") is not None:
            self.scaler.load_state_dict(payload["scaler"])
        if restore_rng and payload.get("rng_state") is not None:
            self._restore_rng_state(payload["rng_state"])
        return self

    def resume_from(
        self,
        value: str | Path,
        *,
        device: str | torch.device | None = None,
    ) -> "Trainer":
        """Resume training from a named or explicit checkpoint."""
        value = str(value)
        if value in {"latest", "last"}:
            path = self.checkpoint_dir / "last"
            if not path.exists():
                steps = sorted(self.checkpoint_dir.glob("step_*"))
                if not steps:
                    raise FileNotFoundError("No checkpoints found.")
                path = steps[-1]
        elif value == "best":
            path = self.checkpoint_dir / "best"
        else:
            path = Path(value)
        return self.load_checkpoint(path, device=device)
