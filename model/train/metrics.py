from __future__ import annotations

import torch


def balanced_accuracy_from_confusion(confusion: torch.Tensor) -> float:
    """Mean per-class recall (diagonal / row sum)."""

    num_classes = confusion.shape[0]
    recalls = []
    for c in range(num_classes):
        row_sum = confusion[c].sum().clamp(min=1).float()
        recalls.append((confusion[c, c].float() / row_sum).item())
    return sum(recalls) / max(num_classes, 1)


def macro_f1_from_confusion(confusion: torch.Tensor) -> float:
    """Unweighted mean of per-class F1 (macro-F1)."""

    num_classes = confusion.shape[0]
    f1s: list[float] = []
    eps = 1e-8
    for c in range(num_classes):
        tp = confusion[c, c].float()
        fp = confusion[:, c].sum().float() - tp
        fn = confusion[c, :].sum().float() - tp
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = (2 * precision * recall) / (precision + recall + eps)
        f1s.append(f1.item())
    return sum(f1s) / max(num_classes, 1)
