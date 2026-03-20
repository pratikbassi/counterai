from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from torch.utils.data import Dataset, random_split
from torchvision import datasets
from torchvision import transforms

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def has_images(path: Path) -> bool:
    for root, _dirs, files in os.walk(path):
        if any(Path(f).suffix.lower() in IMAGE_EXTENSIONS for f in files):
            return True
    return False


def looks_like_imagefolder_root(path: Path) -> bool:
    subdirs = [p for p in path.iterdir() if p.is_dir()]
    if len(subdirs) < 2:
        return False
    return all(has_images(d) for d in subdirs)


def find_imagefolder_root(base_path: Path, max_depth: int = 4) -> Path:
    if looks_like_imagefolder_root(base_path):
        return base_path

    queue: list[tuple[Path, int]] = [(base_path, 0)]
    while queue:
        current, depth = queue.pop(0)
        if depth > max_depth:
            continue
        try:
            if looks_like_imagefolder_root(current):
                return current
            for child in current.iterdir():
                if child.is_dir():
                    queue.append((child, depth + 1))
        except PermissionError:
            continue

    raise FileNotFoundError(
        f"Could not locate ImageFolder-compatible directory under: {base_path}"
    )


def classify_real_fake_dirname(dirname: str) -> Optional[str]:
    """
    Map a directory name to a binary class.

    Expected examples: Train_Real/Test_Fake/whatever_real/whatever_fake.
    """

    lower = dirname.lower()
    if "real" in lower:
        return "Real"
    if "fake" in lower:
        return "Fake"
    return None


def gather_real_fake_items(split_dir: Path) -> list[tuple[Path, int]]:
    """
    Walk a split directory and return (image_path, label_index) pairs.
    """

    items: list[tuple[Path, int]] = []
    for child in sorted(split_dir.iterdir()):
        if not child.is_dir():
            continue

        label_name = classify_real_fake_dirname(child.name)
        if label_name is None:
            continue

        label_idx = 0 if label_name == "Real" else 1

        for root, _dirs, files in os.walk(child):
            for f in files:
                suffix = Path(f).suffix.lower()
                if suffix in IMAGE_EXTENSIONS:
                    items.append((Path(root) / f, label_idx))

    # Ensure stable ordering for reproducible runs.
    items.sort(key=lambda x: str(x[0]))
    return items


def discover_train_test_real_fake(dataset_path: Path, max_depth: int = 6) -> Optional[dict]:
    """
    Discover datasets shaped like:
      <dataset>/.../train/*_Real, *_Fake
      <dataset>/.../test/*_Real, *_Fake
    """

    train_dir: Optional[Path] = None
    test_dir: Optional[Path] = None

    queue: list[tuple[Path, int]] = [(dataset_path, 0)]
    while queue:
        current, depth = queue.pop(0)
        if depth > max_depth:
            continue
        try:
            for child in current.iterdir():
                if not child.is_dir():
                    continue
                name = child.name.lower()
                if name == "train" and train_dir is None:
                    train_dir = child
                elif name == "test" and test_dir is None:
                    test_dir = child

                queue.append((child, depth + 1))
        except PermissionError:
            continue

        if train_dir is not None and test_dir is not None:
            break

    if train_dir is None or test_dir is None:
        return None

    train_items = gather_real_fake_items(train_dir)
    val_items = gather_real_fake_items(test_dir)
    if not train_items or not val_items:
        return None

    # Ensure both labels exist.
    train_labels = {label_idx for _p, label_idx in train_items}
    val_labels = {label_idx for _p, label_idx in val_items}
    if not ({0, 1}.issubset(train_labels) and {0, 1}.issubset(val_labels)):
        return None

    split_root = (
        train_dir.parent if train_dir.parent == test_dir.parent else dataset_path
    )
    return {
        "split_root": split_root,
        "train_dir": train_dir,
        "test_dir": test_dir,
        "train_items": train_items,
        "val_items": val_items,
        "class_names": ["Real", "Fake"],
    }


class RealFakePathsDataset(Dataset):
    def __init__(self, items: list[tuple[Path, int]], transform: transforms.Compose):
        self.items = items
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self.items[idx]
        img = Image.open(path).convert("RGB")
        img_tensor = self.transform(img)
        return img_tensor, label


def build_train_val_datasets(
    dataset_path: Path,
    *,
    train_tfms: transforms.Compose,
    eval_tfms: transforms.Compose,
    val_split: float,
    seed: int,
) -> dict:
    """
    Returns:
      - train_ds, val_ds, class_names, num_classes
      - imagefolder_root + data_loading metadata for train_metrics.json
    """

    split_data = discover_train_test_real_fake(dataset_path)
    if split_data is not None:
        class_names = split_data["class_names"]
        num_classes = len(class_names)
        train_ds = RealFakePathsDataset(
            split_data["train_items"], transform=train_tfms
        )
        val_ds = RealFakePathsDataset(
            split_data["val_items"], transform=eval_tfms
        )

        if len(train_ds) < 1 or len(val_ds) < 1:
            raise RuntimeError("Train/Test split discovery produced empty datasets.")

        return {
            "train_ds": train_ds,
            "val_ds": val_ds,
            "class_names": class_names,
            "num_classes": num_classes,
            "imagefolder_root": split_data["split_root"],
            "data_loading": {
                "mode": "train_test_real_fake",
                "train_dir": str(split_data["train_dir"]),
                "test_dir": str(split_data["test_dir"]),
            },
        }

    # Fallback: discover an ImageFolder root and do a random train/val split.
    root = find_imagefolder_root(dataset_path)
    base_ds = datasets.ImageFolder(root=str(root))
    num_samples = len(base_ds)
    if num_samples == 0:
        raise RuntimeError("Dataset is empty after discovery.")

    class_names = base_ds.classes
    num_classes = len(class_names)
    if num_classes < 2:
        raise RuntimeError(f"Expected >=2 classes, got {num_classes}: {class_names}")

    val_size = max(1, int(num_samples * val_split))
    train_size = num_samples - val_size
    if train_size < 1:
        raise RuntimeError("Train split too small. Reduce --val-split.")

    g = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(base_ds, [train_size, val_size], generator=g)

    # Re-wrap subsets with distinct transforms for train/val.
    train_ds = datasets.ImageFolder(root=str(root), transform=train_tfms)
    val_ds = datasets.ImageFolder(root=str(root), transform=eval_tfms)
    train_ds = torch.utils.data.Subset(train_ds, train_subset.indices)
    val_ds = torch.utils.data.Subset(val_ds, val_subset.indices)

    return {
        "train_ds": train_ds,
        "val_ds": val_ds,
        "class_names": class_names,
        "num_classes": num_classes,
        "imagefolder_root": root,
        "data_loading": {"mode": "random_imagefolder_split"},
    }

