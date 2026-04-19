"""Hand-authored ResNet-style CNN used in the Phase H from-scratch ablation.

This module deliberately does NOT depend on torchvision: it owns its own
``BasicBlock`` and full network so we can isolate the effect of "no ImageNet
prior" from "different architecture family". Weights are random-initialised
(Kaiming for conv, ones/zeros for BN, default for the head Linear).

Target capacity is ~5-7M parameters at ``num_classes=2`` with a 224x224 input,
which sits between EfficientNet-B0 (~5.3M) and ResNet-18 (~11.7M) and keeps
the Phase H comparison against the EfficientNet-B0 G1 baseline a fair-ish
contest on capacity grounds.

See ``docs/MODEL_ABLATION_PLAN.md`` Phase H for the experimental protocol.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class BasicBlock(nn.Module):
    """Standard two-conv residual block (ResNet-18/34 style).

    Pre-skip activation order:
      Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> add(skip) -> ReLU
    """

    expansion: int = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Identity skip when shapes match; otherwise a 1x1 conv projection.
        if stride != 1 or in_channels != out_channels:
            self.downsample: nn.Module = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = out + identity
        return F.relu(out, inplace=True)


class ScratchCNNv1(nn.Module):
    """Hand-authored ResNet-style CNN (4 stages of 1 BasicBlock each).

    Channel widths (64, 128, 256, 512) and stage downsampling mirror ResNet-18
    for readability, but with 1 block per stage instead of 2 the parameter
    count lands near ~5-6M (close to EfficientNet-B0's 5.3M G1 baseline)
    rather than ~11.7M. The code is self-contained: no torchvision dependency,
    no pretrained weight loading.
    """

    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()

        # Stem: 224 -> 112 (conv stride 2) -> 56 (maxpool stride 2)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # 4 stages, 1 BasicBlock per stage. First block of stages 2-4 downsamples.
        # Depth was chosen to land at ~5-6M params (close to EfficientNet-B0's
        # 5.3M baseline) rather than ResNet-18's ~11.7M. The widths still match
        # the standard ResNet (64, 128, 256, 512) for readability; only the
        # blocks-per-stage count differs.
        self.stage1 = self._make_stage(in_ch=64, out_ch=64, blocks=1, stride=1)
        self.stage2 = self._make_stage(in_ch=64, out_ch=128, blocks=1, stride=2)
        self.stage3 = self._make_stage(in_ch=128, out_ch=256, blocks=1, stride=2)
        self.stage4 = self._make_stage(in_ch=256, out_ch=512, blocks=1, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

        self._init_weights()

    @staticmethod
    def _make_stage(
        *, in_ch: int, out_ch: int, blocks: int, stride: int
    ) -> nn.Sequential:
        layers = [BasicBlock(in_ch, out_ch, stride=stride)]
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_ch, out_ch, stride=1))
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def build_scratch_cnn_v1(num_classes: int) -> nn.Module:
    """Factory wired into ``train/model.py::create_classifier``."""

    return ScratchCNNv1(num_classes=num_classes)


def _smoke_test() -> None:
    """Self-contained sanity check.

    Run with::

        cd model && .venv/bin/python -m train.scratch_cnn

    Exercises the public ``create_classifier`` + ``apply_train_stage`` path
    so a regression in the factory wiring (not just this module) is caught.
    """

    from train.model import apply_train_stage, create_classifier

    device = torch.device("cpu")
    model, arch = create_classifier(
        "scratch_cnn_v1", num_classes=2, device=device
    )
    assert arch == "scratch_cnn_v1", arch

    n_params = sum(p.numel() for p in model.parameters())
    assert 4_000_000 <= n_params <= 8_000_000, (
        f"param count {n_params:,} outside expected 4-8M range"
    )

    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    assert tuple(y.shape) == (2, 2), tuple(y.shape)

    apply_train_stage(
        model, "scratch_cnn_v1", stage="full", unfreeze_layer3=False
    )
    frozen = [n for n, p in model.named_parameters() if not p.requires_grad]
    assert not frozen, f"unexpectedly frozen params: {frozen[:3]}..."

    apply_train_stage(
        model, "scratch_cnn_v1", stage="head", unfreeze_layer3=False
    )
    frozen_head = [n for n, p in model.named_parameters() if not p.requires_grad]
    assert not frozen_head, (
        "head stage should also unfreeze all for scratch_cnn_v1; "
        f"frozen: {frozen_head[:3]}..."
    )

    print(f"OK scratch_cnn_v1: params={n_params:,} ({n_params / 1e6:.2f}M), "
          f"forward shape={tuple(y.shape)}, all params trainable in both stages")


if __name__ == "__main__":
    _smoke_test()
