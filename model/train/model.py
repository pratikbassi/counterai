from __future__ import annotations

from typing import Literal

import torch
from torch import nn
from torchvision import models


Architecture = Literal["resnet18", "resnet50", "efficientnet_b0"]


def freeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False


def create_classifier(
    architecture: str, *, num_classes: int, device: torch.device
) -> tuple[nn.Module, str]:
    """
    Build a pretrained classifier head for Real-vs-Fake (or general) num_classes.

    Returns (model, normalized_architecture_name).
    """

    arch = architecture.strip().lower().replace("-", "_")
    if arch == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif arch == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif arch in ("efficientnet_b0", "efficientnetb0"):
        arch = "efficientnet_b0"
        weights = models.EfficientNet_B0_Weights.DEFAULT
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(
            f"Unknown architecture {architecture!r}. "
            "Use: resnet18, resnet50, efficientnet_b0"
        )

    model.to(device)
    return model, arch


def apply_train_stage(
    model: nn.Module,
    architecture: str,
    *,
    stage: Literal["head", "full"],
    unfreeze_layer3: bool,
) -> None:
    """
    ResNet: head = fc only; full = layer4+fc, optionally layer3.
    EfficientNet-B0: head = classifier only; full = features + classifier.
    """

    freeze_all(model)
    arch = architecture.lower()

    if arch in ("resnet18", "resnet50"):
        if stage == "head":
            for p in model.fc.parameters():
                p.requires_grad = True
        else:
            if unfreeze_layer3:
                for p in model.layer3.parameters():
                    p.requires_grad = True
            for p in model.layer4.parameters():
                p.requires_grad = True
            for p in model.fc.parameters():
                p.requires_grad = True
    elif arch == "efficientnet_b0":
        if stage == "head":
            for p in model.classifier.parameters():
                p.requires_grad = True
        else:
            for p in model.features.parameters():
                p.requires_grad = True
            for p in model.classifier.parameters():
                p.requires_grad = True
    else:
        raise ValueError(f"Unsupported architecture: {architecture!r}")
