# Copyright 2026 Zameer Hussain and Akhtar Hussain
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# See LICENSE.md and LICENSING_NOTICE.md; commercial use requires a separate written license.

"""Architecture-agnostic training and checkpointing for MLBricks."""
from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path
import random
from typing import Any, Callable, Iterable, Iterator

import torch
from torch import nn

from .lifecycle import load as load_model
from .lifecycle import save as save_model
from .optim import Adam, AdamW, stabilize_optimizer


@dataclass
class TrainerState:
    step: int = 0
    epoch: int = 0
    best_val_loss: float = float("inf")


def _device_of(model: nn.Module) -> torch.device:
    for parameter in model.parameters():
        return parameter.device
    for buffer in model.buffers():
        return buffer.device
    return torch.device("cpu")


def _move(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {k: _move(v, device) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(_move(v, device) for v in value)
    if isinstance(value, list):
        return [_move(v, device) for v in value]
    return value


def _scalar_loss(output: Any) -> torch.Tensor | None:
    if isinstance(output, torch.Tensor) and output.ndim == 0:
        return output
    if isinstance(output, dict):
        loss = output.get("loss")
        return loss if isinstance(loss, torch.Tensor) else None
    loss = getattr(output, "loss", None)
    if isinstance(loss, torch.Tensor):
        return loss
    if isinstance(output, (tuple, list)):
        # MLBricks language models conventionally return (logits, loss).
        if len(output) > 1 and isinstance(output[1], torch.Tensor) and output[1].ndim == 0:
            return output[1]
        if output and isinstance(output[0], torch.Tensor) and output[0].ndim == 0:
            return output[0]
    return None


def _logits(output: Any) -> torch.Tensor | None:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, dict):
        value = output.get("logits")
        return value if isinstance(value, torch.Tensor) else None
    value = getattr(output, "logits", None)
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    return None


def _optimizer_spec(optimizer: torch.optim.Optimizer | None) -> dict[str, Any] | None:
    if optimizer is None:
        return None
    defaults = dict(getattr(optimizer, "defaults", {}))
    # Keep only values torch.save can reliably round-trip and constructor kwargs
    # can normally consume.
    defaults.pop("params", None)
    return {
        "module": type(optimizer).__module__,
        "class": type(optimizer).__qualname__,
        "defaults": defaults,
    }


def _build_optimizer(
    model: nn.Module,
    optimizer: str | torch.optim.Optimizer | None,
    *,
    lr: float,
    optimizer_kwargs: dict[str, Any] | None = None,
) -> torch.optim.Optimizer | None:
    if optimizer is None:
        return None
    if isinstance(optimizer, torch.optim.Optimizer):
        stabilize_optimizer(optimizer)
        return optimizer
    name = str(optimizer).strip().lower().replace("_", "")
    kwargs = dict(optimizer_kwargs or {})
    kwargs.setdefault("lr", lr)
    if name in {"adamw", "mlbricksadamw"}:
        return AdamW(model.parameters(), **kwargs)
    if name in {"adam", "mlbricksadam"}:
        return Adam(model.parameters(), **kwargs)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), **kwargs)
    raise ValueError("optimizer must be an Optimizer or one of: adamw, adam, sgd")


def _restore_optimizer_from_spec(model: nn.Module, spec: dict[str, Any] | None) -> torch.optim.Optimizer | None:
    if not spec:
        return None
    module_name = str(spec.get("module", ""))
    class_name = str(spec.get("class", ""))
    kwargs = dict(spec.get("defaults", {}))
    try:
        module = importlib.import_module(module_name)
        cls: Any = module
        for part in class_name.split("."):
            cls = getattr(cls, part)
        import inspect as _inspect
        signature = _inspect.signature(cls.__init__)
        if not any(p.kind == _inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()):
            kwargs = {k: v for k, v in kwargs.items() if k in signature.parameters}
        optimizer = cls(model.parameters(), **kwargs)
        stabilize_optimizer(optimizer, warn=False)
        return optimizer
    except Exception as exc:
        raise RuntimeError(
            f"Could not reconstruct optimizer {module_name}.{class_name}; "
            "pass an optimizer when constructing Trainer and call resume_from()."
        ) from exc


class Trainer:
    """Generic MLBricks trainer for ESA, Bolt, Bricks, SOUP, VESA and future modules."""

    def __init__(
        self,
        model: nn.Module,
        *,
        optimizer: str | torch.optim.Optimizer | None = "adamw",
        lr: float = 3e-4,
        optimizer_kwargs: dict[str, Any] | None = None,
        scheduler: Any | None = None,
        scaler: Any | None = None,
        loss_fn: Callable[[Any, Any], torch.Tensor] | None = None,
        checkpoint_dir: str | Path = "checkpoints",
        save_every: int | None = None,
        save_at: Iterable[int] | None = None,
        save_best: bool = True,
        save_last: bool = True,
        keep_last_n: int | None = 3,
        grad_clip: float | None = None,
    ) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError("Trainer model must be a torch.nn.Module")
        self.model = model
        self.optimizer = _build_optimizer(
            model, optimizer, lr=float(lr), optimizer_kwargs=optimizer_kwargs
        )
        self.scheduler = scheduler
        self.scaler = scaler
        self.loss_fn = loss_fn
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_every = int(save_every) if save_every else None
        self.save_at = {int(x) for x in (save_at or [])}
        self.save_best = bool(save_best)
        self.save_last = bool(save_last)
        self.keep_last_n = keep_last_n
        self.grad_clip = None if grad_clip is None else float(grad_clip)
        self.state = TrainerState()

    def _rng_state(self) -> dict[str, Any]:
        state = {"python": random.getstate(), "torch": torch.get_rng_state()}
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

    def _forward_batch(self, batch: Any) -> tuple[Any, torch.Tensor]:
        device = _device_of(self.model)
        batch = _move(batch, device)
        target = None

        if isinstance(batch, dict):
            kwargs = dict(batch)
            target = kwargs.get("targets", kwargs.get("labels"))
            try:
                output = self.model(**kwargs)
            except TypeError:
                if "labels" in kwargs and "targets" not in kwargs:
                    kwargs["targets"] = kwargs.pop("labels")
                    output = self.model(**kwargs)
                else:
                    raise
        elif isinstance(batch, (tuple, list)) and len(batch) == 2:
            x, target = batch
            try:
                output = self.model(x, targets=target)
            except TypeError:
                output = self.model(x, target)
        elif isinstance(batch, (tuple, list)):
            output = self.model(*batch)
        else:
            output = self.model(batch)

        loss = _scalar_loss(output)
        if loss is None and self.loss_fn is not None and target is not None:
            logits = _logits(output)
            if logits is None:
                raise RuntimeError("Could not extract model predictions for loss_fn")
            loss = self.loss_fn(logits, target)
        if loss is None:
            raise RuntimeError(
                "Trainer could not find a scalar loss. Return loss/(logits, loss), "
                "pass batches as (inputs, targets), or provide loss_fn=."
            )
        return output, loss

    def fit(
        self,
        train_loader: Iterable[Any],
        *,
        steps: int | None = None,
        epochs: int | None = None,
        val_loader: Iterable[Any] | None = None,
        validate_every: int | None = None,
        log_every: int | None = None,
        callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Train the current model and return a compact run summary."""
        if self.optimizer is None:
            raise RuntimeError("Trainer.fit() requires an optimizer")
        if steps is not None and steps <= 0:
            raise ValueError("steps must be positive")
        if epochs is not None and epochs <= 0:
            raise ValueError("epochs must be positive")
        if steps is None and epochs is None:
            epochs = 1

        losses: list[float] = []
        stop = False
        epoch_index = self.state.epoch

        while not stop:
            epoch_index += 1
            self.state.epoch = epoch_index
            saw_batch = False
            for batch in train_loader:
                saw_batch = True
                self.model.train(True)
                self.optimizer.zero_grad(set_to_none=True)
                _, loss = self._forward_batch(batch)

                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                    if self.grad_clip is not None:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    if self.grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.optimizer.step()

                if self.scheduler is not None:
                    self.scheduler.step()

                self.state.step += 1
                loss_value = float(loss.detach().cpu())
                losses.append(loss_value)

                val_loss = None
                if val_loader is not None and validate_every and self.state.step % validate_every == 0:
                    metrics = self.evaluate(val_loader)
                    val_loss = metrics["loss"]

                self.maybe_save(step=self.state.step, val_loss=val_loss)

                if callback is not None and (log_every is None or self.state.step % log_every == 0):
                    callback({
                        "step": self.state.step,
                        "epoch": self.state.epoch,
                        "loss": loss_value,
                        "val_loss": val_loss,
                    })

                if steps is not None and self.state.step >= steps:
                    stop = True
                    break

            if not saw_batch:
                raise RuntimeError("train_loader produced no batches")
            if steps is None and epochs is not None and epoch_index >= epochs:
                stop = True
            elif steps is not None and self.state.step >= steps:
                stop = True

        final_path = self.save_final()
        return {
            "step": self.state.step,
            "epoch": self.state.epoch,
            "loss": losses[-1] if losses else None,
            "mean_loss": sum(losses) / len(losses) if losses else None,
            "best_val_loss": self.state.best_val_loss,
            "checkpoint": str(final_path) if final_path is not None else None,
        }

    @torch.no_grad()
    def evaluate(self, data_loader: Iterable[Any]) -> dict[str, float]:
        was_training = self.model.training
        self.model.eval()
        losses: list[float] = []
        try:
            for batch in data_loader:
                _, loss = self._forward_batch(batch)
                losses.append(float(loss.detach().cpu()))
        finally:
            self.model.train(was_training)
        if not losses:
            raise RuntimeError("data_loader produced no batches")
        return {"loss": sum(losses) / len(losses), "batches": float(len(losses))}

    def save_checkpoint(
        self,
        *,
        step: int | None = None,
        name: str | None = None,
        protected: bool = False,
        extra: dict[str, Any] | None = None,
    ) -> Path:
        step = int(self.state.step if step is None else step)
        name = name or f"step_{step:06d}"
        path = self.checkpoint_dir / name
        save_model(self.model, path, metadata={"kind": "training_checkpoint", "step": step})

        payload = {
            "step": step,
            "epoch": self.state.epoch,
            "best_val_loss": self.state.best_val_loss,
            "optimizer": self.optimizer.state_dict() if self.optimizer is not None else None,
            "optimizer_spec": _optimizer_spec(self.optimizer),
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            "scaler": self.scaler.state_dict() if self.scaler is not None else None,
            "rng_state": self._rng_state(),
            "protected": bool(protected),
            "extra": extra or {},
            "trainer": {
                "save_every": self.save_every,
                "save_at": sorted(self.save_at),
                "save_best": self.save_best,
                "save_last": self.save_last,
                "keep_last_n": self.keep_last_n,
                "grad_clip": self.grad_clip,
            },
        }
        torch.save(payload, path / "training_state.pt")
        return path

    def save(self, name: str = "last", *, extra: dict[str, Any] | None = None) -> Path:
        """Explicitly save the current complete training state."""
        return self.save_checkpoint(name=name, protected=True, extra=extra)

    def maybe_save(self, *, step: int, val_loss: float | None = None) -> list[Path]:
        self.state.step = int(step)
        saved: list[Path] = []
        exact = step in self.save_at
        periodic = self.save_every is not None and step % self.save_every == 0
        if exact or periodic:
            saved.append(self.save_checkpoint(step=step, protected=exact))
        if self.save_best and val_loss is not None and val_loss < self.state.best_val_loss:
            self.state.best_val_loss = float(val_loss)
            saved.append(self.save_checkpoint(step=step, name="best", protected=True))
        self._prune_periodic()
        return saved

    def save_final(self) -> Path | None:
        if not self.save_last:
            return None
        return self.save_checkpoint(step=self.state.step, name="last", protected=True)

    def _prune_periodic(self) -> None:
        if self.keep_last_n is None or self.keep_last_n < 0:
            return
        candidates: list[tuple[int, Path]] = []
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
        doomed = candidates[:-self.keep_last_n] if self.keep_last_n else candidates
        for _, path in doomed:
            import shutil
            shutil.rmtree(path, ignore_errors=True)

    def load_checkpoint(
        self,
        path: str | Path,
        *,
        device: str | torch.device | None = None,
        restore_rng: bool = True,
    ) -> "Trainer":
        path = Path(path)
        target = _device_of(self.model) if device is None else torch.device(device)
        loaded = load_model(path, device=target)
        self.model.load_state_dict(loaded.state_dict())
        self.model.to(target)

        payload = torch.load(path / "training_state.pt", map_location="cpu", weights_only=False)
        self.state.step = int(payload.get("step", 0))
        self.state.epoch = int(payload.get("epoch", 0))
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

    @classmethod
    def resume(
        cls,
        path: str | Path,
        *,
        device: str | torch.device | None = "auto",
        loss_fn: Callable[[Any, Any], torch.Tensor] | None = None,
        scheduler: Any | None = None,
        scaler: Any | None = None,
        restore_rng: bool = True,
    ) -> "Trainer":
        """Create a Trainer directly from a saved MLBricks checkpoint."""
        path = Path(path)
        model = load_model(path, device=device)
        payload = torch.load(path / "training_state.pt", map_location="cpu", weights_only=False)
        optimizer = _restore_optimizer_from_spec(model, payload.get("optimizer_spec"))
        cfg = dict(payload.get("trainer", {}))
        trainer = cls(
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            loss_fn=loss_fn,
            checkpoint_dir=path.parent,
            save_every=cfg.get("save_every"),
            save_at=cfg.get("save_at"),
            save_best=cfg.get("save_best", True),
            save_last=cfg.get("save_last", True),
            keep_last_n=cfg.get("keep_last_n", 3),
            grad_clip=cfg.get("grad_clip"),
        )
        trainer.state.step = int(payload.get("step", 0))
        trainer.state.epoch = int(payload.get("epoch", 0))
        trainer.state.best_val_loss = float(payload.get("best_val_loss", float("inf")))
        if trainer.optimizer is not None and payload.get("optimizer") is not None:
            trainer.optimizer.load_state_dict(payload["optimizer"])
        if scheduler is not None and payload.get("scheduler") is not None:
            scheduler.load_state_dict(payload["scheduler"])
        if scaler is not None and payload.get("scaler") is not None:
            scaler.load_state_dict(payload["scaler"])
        if restore_rng and payload.get("rng_state") is not None:
            trainer._restore_rng_state(payload["rng_state"])
        return trainer


def train(
    model: nn.Module,
    train_loader: Iterable[Any],
    *,
    steps: int | None = None,
    epochs: int | None = None,
    val_loader: Iterable[Any] | None = None,
    validate_every: int | None = None,
    **trainer_kwargs: Any,
) -> Trainer:
    """One-call convenience API: build a Trainer and run ``fit``."""
    trainer = Trainer(model, **trainer_kwargs)
    trainer.fit(
        train_loader,
        steps=steps,
        epochs=epochs,
        val_loader=val_loader,
        validate_every=validate_every,
    )
    return trainer


__all__ = ["Trainer", "TrainerState", "train"]
