from __future__ import annotations

import io
import random
from typing import Callable, Tuple

from PIL import Image
from torchvision import transforms


def _maybe_jpeg_compress(img: Image.Image, quality_min: int, quality_max: int) -> Image.Image:
    buf = io.BytesIO()
    q = random.randint(quality_min, quality_max)
    img.save(buf, format="JPEG", quality=q)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def create_transforms(
    image_size: int,
    *,
    augment_strength: str = "default",
    randaugment: bool = False,
    randaugment_num_ops: int = 2,
    randaugment_magnitude: int = 9,
    jpeg_augment: bool = False,
    jpeg_quality_min: int = 40,
    jpeg_quality_max: int = 90,
    jpeg_prob: float = 0.25,
) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Train / eval transforms. ImageNet normalization for torchvision pretrained models.

    augment_strength:
      - default: mild ColorJitter (current baseline)
      - strong: wider ColorJitter + optional RandAugment / JPEG (see flags)

    Tradeoff: stronger aug can improve robustness but may slow convergence / need more epochs.
    """

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_blocks: list[Callable] = [
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
    ]

    if augment_strength == "strong":
        train_blocks.append(
            transforms.ColorJitter(
                brightness=0.25,
                contrast=0.25,
                saturation=0.25,
                hue=0.04,
            )
        )
    else:
        train_blocks.append(
            transforms.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.1,
            )
        )

    if randaugment:
        train_blocks.append(
            transforms.RandAugment(
                num_ops=randaugment_num_ops,
                magnitude=randaugment_magnitude,
            )
        )

    if jpeg_augment:
        train_blocks.append(
            transforms.RandomApply(
                [
                    transforms.Lambda(
                        lambda im: _maybe_jpeg_compress(
                            im, jpeg_quality_min, jpeg_quality_max
                        )
                    )
                ],
                p=jpeg_prob,
            )
        )

    train_blocks.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    train_tfms = transforms.Compose(train_blocks)
    eval_tfms = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return train_tfms, eval_tfms
