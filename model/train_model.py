from __future__ import annotations

import torch
from torch import nn
from torchvision import models


def create_resnet18_classifier(*, num_classes: int, device: torch.device) -> nn.Module:
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.to(device)
    return model

