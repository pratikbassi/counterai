from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader

from .logging import StepLogger
from .metrics import balanced_accuracy_from_confusion, macro_f1_from_confusion

if TYPE_CHECKING:
    from .ema import ModelEMA


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def train_one_epoch(
    *,
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    batch_size: int,
    amp_enabled: bool,
    log_every: int,
    ema: Optional["ModelEMA"] = None,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    steps = 0

    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    logger = StepLogger(prefix="train", batch_size=batch_size, log_every=log_every)

    for batch_idx, (images, labels) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            logits = model(images)
            loss = criterion(logits, labels)

        batch_acc = accuracy(logits.detach(), labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if ema is not None:
            ema.update(model)

        total_loss += loss.item()
        total_acc += batch_acc
        steps += 1

        logger.maybe_log(
            batch_idx=batch_idx,
            num_batches=len(loader),
            loss=loss.item(),
            acc=batch_acc,
        )

    return total_loss / max(steps, 1), total_acc / max(steps, 1)


@torch.no_grad()
def evaluate_metrics(
    *,
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    batch_size: int,
    amp_enabled: bool,
    log_every: int,
    class_names: Optional[list[str]] = None,
) -> dict:
    """
    Validation metrics in one pass: loss, acc, balanced accuracy, macro-F1, confusion matrix.
    """

    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    steps = 0
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    logger = StepLogger(prefix="val", batch_size=batch_size, log_every=log_every)

    with torch.inference_mode():
        for batch_idx, (images, labels) in enumerate(loader, start=1):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=amp_enabled):
                logits = model(images)
                loss = criterion(logits, labels)

            preds = logits.argmax(dim=1)
            batch_acc = accuracy(logits, labels)
            total_loss += loss.item()
            total_acc += batch_acc
            steps += 1

            flat = (labels.to(torch.int64) * num_classes) + preds.to(torch.int64)
            counts = torch.bincount(flat.cpu(), minlength=num_classes * num_classes)
            confusion += counts.reshape(num_classes, num_classes)

            logger.maybe_log(
                batch_idx=batch_idx,
                num_batches=len(loader),
                loss=loss.item(),
                acc=batch_acc,
            )

    cf = confusion.float()
    out: dict = {
        "loss": total_loss / max(steps, 1),
        "acc": total_acc / max(steps, 1),
        "balanced_acc": balanced_accuracy_from_confusion(cf),
        "macro_f1": macro_f1_from_confusion(cf),
        "confusion_matrix": confusion.tolist(),
    }
    if class_names is not None:
        per_class = {}
        for i in range(num_classes):
            row_sum = confusion[i].sum().clamp(min=1).float()
            name = class_names[i] if i < len(class_names) else str(i)
            per_class[name] = (confusion[i, i].float() / row_sum).item()
        out["per_class_accuracy"] = per_class
    return out


@torch.no_grad()
def evaluate(
    *,
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    batch_size: int,
    amp_enabled: bool,
    log_every: int,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    steps = 0

    logger = StepLogger(prefix="val", batch_size=batch_size, log_every=log_every)

    with torch.inference_mode():
        for batch_idx, (images, labels) in enumerate(loader, start=1):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=amp_enabled):
                logits = model(images)
                loss = criterion(logits, labels)

            batch_acc = accuracy(logits, labels)
            total_loss += loss.item()
            total_acc += batch_acc
            steps += 1

            logger.maybe_log(
                batch_idx=batch_idx,
                num_batches=len(loader),
                loss=loss.item(),
                acc=batch_acc,
            )

    return total_loss / max(steps, 1), total_acc / max(steps, 1)


@torch.no_grad()
def evaluate_confusion_matrix(
    *,
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    class_names: list[str],
    batch_size: int,
    amp_enabled: bool,
) -> dict:
    """
    Same as evaluate_metrics with per-class recall and no per-step val logging.
    """

    return evaluate_metrics(
        model=model,
        loader=loader,
        criterion=criterion,
        device=device,
        num_classes=num_classes,
        class_names=class_names,
        batch_size=batch_size,
        amp_enabled=amp_enabled,
        log_every=0,
    )

