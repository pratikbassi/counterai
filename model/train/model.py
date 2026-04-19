from __future__ import annotations

from typing import Literal

import torch
from torch import nn
from torchvision import models


Architecture = Literal[
    "resnet18",
    "resnet50",
    "efficientnet_b0",
    "convnext_tiny",
    "efficientnet_v2_s",
    "vit_b_16",
    "scratch_cnn_v1",
]


def freeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False


def create_classifier(
    architecture: str, *, num_classes: int, device: torch.device
) -> tuple[nn.Module, str]:
    """
    Build a classifier model for Real-vs-Fake (or general) num_classes.

    Returns (model, normalized_architecture_name).

    Note: we intentionally pin ``IMAGENET1K_V1`` weights for the newer backbones
    (convnext_tiny, efficientnet_v2_s, vit_b_16). ``DEFAULT`` would silently swap
    in alternate preprocessing (e.g. ViT-B/16 ``DEFAULT`` is the SWAG variant
    with mean=0.5/std=0.5 and crop=384), which would mismatch the ImageNet
    mean/std + 224-crop pipeline in ``train/transforms.py``.

    ``scratch_cnn_v1`` is the from-scratch ablation arch (Phase H): a hand-
    authored ResNet-style CNN built in :mod:`train.scratch_cnn` with random
    init and **no** torchvision/ImageNet weights. See its module docstring
    and ``docs/MODEL_ABLATION_PLAN.md`` Phase H for the protocol.
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
    elif arch in ("convnext_tiny", "convnexttiny"):
        arch = "convnext_tiny"
        weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        model = models.convnext_tiny(weights=weights)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, num_classes)
    elif arch in ("efficientnet_v2_s", "efficientnetv2s", "efficientnet_v2s"):
        arch = "efficientnet_v2_s"
        weights = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
        model = models.efficientnet_v2_s(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    elif arch in ("vit_b_16", "vitb16", "vit_base_16"):
        arch = "vit_b_16"
        weights = models.ViT_B_16_Weights.IMAGENET1K_V1
        model = models.vit_b_16(weights=weights)
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)
    elif arch in ("scratch_cnn_v1", "scratchcnnv1", "scratch_cnn"):
        arch = "scratch_cnn_v1"
        # Local import keeps torchvision out of the scratch arch's dependency
        # graph and makes the "no pretrained weights" boundary explicit.
        from train.scratch_cnn import build_scratch_cnn_v1

        model = build_scratch_cnn_v1(num_classes=num_classes)
    else:
        raise ValueError(
            f"Unknown architecture {architecture!r}. "
            "Use: resnet18, resnet50, efficientnet_b0, "
            "convnext_tiny, efficientnet_v2_s, vit_b_16, scratch_cnn_v1"
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
    Stage-aware unfreeze:
      ResNet:           head = fc only;          full = layer4+fc (+ layer3 if requested)
      EfficientNet-B0:  head = classifier;       full = features + classifier
      EfficientNet-V2-S: head = classifier;      full = features + classifier
      ConvNeXt-Tiny:    head = classifier[2];    full = features + classifier
      ViT-B/16:         head = heads;            full = encoder + heads
      scratch_cnn_v1:   head = full;             full = full

    For ``scratch_cnn_v1`` there is no pretrained backbone to protect, so
    "head warmup" is meaningless and both stages unfreeze every parameter.
    The recommended recipe is ``--head-epochs 0`` so the head stage is
    skipped entirely; we still handle it correctly if it's not.
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
    elif arch in ("efficientnet_b0", "efficientnet_v2_s"):
        if stage == "head":
            for p in model.classifier.parameters():
                p.requires_grad = True
        else:
            for p in model.features.parameters():
                p.requires_grad = True
            for p in model.classifier.parameters():
                p.requires_grad = True
    elif arch == "convnext_tiny":
        if stage == "head":
            # The Linear is classifier[2]; train the trailing classifier block (LN+Flatten+Linear).
            for p in model.classifier.parameters():
                p.requires_grad = True
        else:
            for p in model.features.parameters():
                p.requires_grad = True
            for p in model.classifier.parameters():
                p.requires_grad = True
    elif arch == "vit_b_16":
        if stage == "head":
            for p in model.heads.parameters():
                p.requires_grad = True
        else:
            # Unfreeze everything: encoder + heads + patch projection (conv_proj)
            # + class_token. The latter two live at the model root, not inside
            # encoder, so iterating only model.encoder would leave them frozen.
            for p in model.parameters():
                p.requires_grad = True
    elif arch == "scratch_cnn_v1":
        # Random init -> no pretrained backbone to protect; both stages train
        # the full network. Pair with ``--head-epochs 0`` to skip the head
        # stage entirely (otherwise the "head" stage just runs full training
        # under the head LR, which is not what you want).
        for p in model.parameters():
            p.requires_grad = True
    else:
        raise ValueError(f"Unsupported architecture: {architecture!r}")
