from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

import torch
from torch import nn


class ModelEMA:
    """
    Exponential moving average of trainable parameters for more stable validation.

    Tradeoff: extra memory (~1x model params) and small CPU/GPU sync cost per step.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError("decay must be in (0, 1)")
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {}
        self._backup: dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.detach().clone()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self.decay
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name not in self.shadow:
                self.shadow[name] = param.detach().clone()
            self.shadow[name].mul_(d).add_(param.detach(), alpha=1.0 - d)

    @torch.no_grad()
    def apply_to_model(self, model: nn.Module) -> None:
        """Copy EMA weights into the live model (destructive). Use before save/eval if desired."""

        for name, param in model.named_parameters():
            if name in self.shadow:
                param.data.copy_(self.shadow[name])

    def state_dict(self) -> dict[str, Any]:
        return {"decay": self.decay, "shadow": {k: v.cpu() for k, v in self.shadow.items()}}

    def load_state_dict(self, state: dict[str, Any], model: nn.Module) -> None:
        self.decay = float(state["decay"])
        self.shadow = {k: v.clone() for k, v in state["shadow"].items()}
        # Drop stale keys if architecture changed
        valid = {n for n, _ in model.named_parameters()}
        self.shadow = {k: v for k, v in self.shadow.items() if k in valid}


@contextmanager
def ema_eval_scope(model: nn.Module, ema: ModelEMA) -> Iterator[None]:
    """Temporarily load EMA weights into the live model for eval/save; restores after."""

    backup: dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if name in ema.shadow:
            backup[name] = param.data.clone()
            param.data.copy_(ema.shadow[name])
    try:
        yield
    finally:
        for name, param in model.named_parameters():
            if name in backup:
                param.data.copy_(backup[name])
