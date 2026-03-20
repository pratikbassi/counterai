from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader

from train_logging import StepLogger


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
        # torch.cuda.amp.autocast is deprecated in recent PyTorch versions.
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            logits = model(images)
            loss = criterion(logits, labels)

        batch_acc = accuracy(logits.detach(), labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

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

            # torch.cuda.amp.autocast is deprecated in recent PyTorch versions.
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
    Computes:
      - overall loss/acc
      - per-class accuracy (recall per true class)
      - confusion matrix [true_class][pred_class]
    """

    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    steps = 0

    # confusion[true, pred] counts.
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    per_class_correct = torch.zeros((num_classes,), dtype=torch.int64)
    per_class_total = torch.zeros((num_classes,), dtype=torch.int64)

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=amp_enabled):
                logits = model(images)
                loss = criterion(logits, labels)

            preds = logits.argmax(dim=1)

            total_loss += loss.item()
            total_acc += accuracy(logits, labels)
            steps += 1

            # Vectorized confusion matrix update.
            # Map (true, pred) -> flat index: true * num_classes + pred
            flat = (labels.to(torch.int64) * num_classes) + preds.to(torch.int64)
            counts = torch.bincount(flat.cpu(), minlength=num_classes * num_classes)
            confusion += counts.reshape(num_classes, num_classes)

            per_class_total += torch.bincount(
                labels.to(torch.int64).cpu(), minlength=num_classes
            )
            per_class_correct += torch.bincount(
                labels.to(torch.int64).cpu()[preds.cpu() == labels.cpu()],
                minlength=num_classes,
            )

    per_class_acc = (
        per_class_correct.float() / torch.clamp(per_class_total.float(), min=1.0)
    ).tolist()

    # Human-friendly labels
    per_class = {
        class_names[i] if i < len(class_names) else str(i): per_class_acc[i]
        for i in range(num_classes)
    }

    return {
        "loss": total_loss / max(steps, 1),
        "acc": total_acc / max(steps, 1),
        "per_class_accuracy": per_class,
        "confusion_matrix": confusion.tolist(),
    }

