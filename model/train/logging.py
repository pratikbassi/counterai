from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class StepStats:
    avg_step_s: float
    imgs_per_s: float


class StepLogger:
    """
    Prints periodic training/eval progress with simple throughput estimates.
    """

    def __init__(self, *, prefix: str, batch_size: int, log_every: int) -> None:
        self.prefix = prefix
        self.batch_size = batch_size
        self.log_every = log_every
        self._last_log_t = time.perf_counter()
        self._last_log_steps = 0

    def maybe_log(
        self,
        *,
        batch_idx: int,
        num_batches: int,
        loss: float,
        acc: float,
    ) -> None:
        if self.log_every <= 0:
            return
        if not (batch_idx % self.log_every == 0 or batch_idx == num_batches):
            return

        now = time.perf_counter()
        dt = now - self._last_log_t
        steps_in_window = batch_idx - self._last_log_steps
        avg_step_s = dt / max(1, steps_in_window)
        imgs_per_s = (self.batch_size * steps_in_window) / max(dt, 1e-9)

        print(
            f"  {self.prefix} step {batch_idx}/{num_batches} | "
            f"loss={loss:.4f} acc={acc:.4f} | "
            f"{imgs_per_s:.1f} imgs/s | {avg_step_s:.3f}s/step"
        )

        self._last_log_t = now
        self._last_log_steps = batch_idx

