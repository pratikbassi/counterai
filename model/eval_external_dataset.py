#!/usr/bin/env python3
"""
Evaluate a trained checkpoint on a **labeled** external dataset.

**ImageFolder:** pass ``--dataset-path`` (class subfolders; see ``--class-map`` if needed).

**CSV + flat images:** pass ``--labeled-csv`` and ``--csv-root`` (paths in the CSV are
resolved relative to ``csv-root``, same rules as training — e.g. ``test_data/foo.jpg``).

Example (held-out labels next to flat ``test_data/``):

  cd model
  .venv/bin/python eval_external_dataset.py \\
    --labeled-csv data/source_labels.csv \\
    --csv-root data \\
    --checkpoint artifacts/best_real_fake.pt
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets

from train.data import (
    RealFakePathsDataset,
    _collect_labeled_cells,
    _paths_from_labeled,
    find_imagefolder_root,
)
from train.loop import evaluate_metrics
from train.model import create_classifier
from train.transforms import create_transforms
from train.utils import load_checkpoint, print_confusion_matrix, resolve_device


def _parse_class_maps(items: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for raw in items:
        if ":" not in raw:
            raise ValueError(f"Invalid --class-map (expected Name:idx): {raw!r}")
        name, idx_s = raw.rsplit(":", 1)
        out[name.strip()] = int(idx_s.strip())
    return out


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def _infer_target_for_folder(name: str) -> int | None:
    """Map folder name -> 0 (human/real) or 1 (AI/fake). None if unclear."""

    n = name.lower()
    key = _norm(name)

    # ImageFolder often uses ``0`` / ``1`` as directory names for binary data.
    stripped = name.strip()
    if stripped == "0":
        return 0
    if stripped == "1":
        return 1

    human_hints = (
        "human",
        "real",
        "natural",
        "authentic",
        "photograph",
        "photo",
        "nonai",
        "notai",
        "organic",
    )
    ai_hints = (
        "ai",
        "fake",
        "synthetic",
        "generated",
        "gen",
        "gan",
        "dalle",
        "midjourney",
        "stable",
        "diffusion",
        "artificial",
        "machine",
    )

    h = any(h in n or h in key for h in human_hints)
    a = any(h in n or h in key for h in ai_hints)
    if h and not a:
        return 0
    if a and not h:
        return 1
    return None


def _resolve_folder_targets(
    class_names: list[str],
    explicit: dict[str, int],
) -> list[int]:
    """One target index (0 or 1) per ImageFolder class index."""

    targets: list[int] = []
    for cname in class_names:
        if cname in explicit:
            targets.append(explicit[cname])
            continue
        # case-insensitive / trimmed key match
        hit = None
        low = cname.strip().lower()
        for k, v in explicit.items():
            if k.strip().lower() == low:
                hit = v
                break
        if hit is not None:
            targets.append(hit)
            continue

        inf = _infer_target_for_folder(cname)
        if inf is not None:
            targets.append(inf)
            continue

        raise ValueError(
            f"Cannot map folder {cname!r} to class 0 or 1. "
            f"Known folders: {class_names!r}. "
            f"Pass e.g. --class-map '{cname}:0' or '{cname}:1'."
        )

    if len(set(targets)) < 2:
        raise ValueError(
            f"All folders mapped to the same target {targets!r}; "
            f"need one folder -> 0 and one -> 1 for binary eval."
        )
    return targets


class _RemappedImageFolder(Dataset):
    def __init__(self, root: Path, transform, folder_idx_to_target: list[int]):
        self._ds = datasets.ImageFolder(root=str(root), transform=transform)
        self._remap = folder_idx_to_target
        if len(self._remap) != len(self._ds.classes):
            raise RuntimeError("Internal error: remap length != num classes")

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, i: int):
        x, y = self._ds[i]
        return x, self._remap[y]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate a checkpoint on an external labeled dataset (ImageFolder or CSV)"
    )
    p.add_argument("--checkpoint", type=str, required=True)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--dataset-path",
        type=str,
        default="",
        help="Root directory; an ImageFolder layout is discovered under it.",
    )
    src.add_argument(
        "--labeled-csv",
        type=str,
        default="",
        help="Labeled CSV (image path + label columns). Use with --csv-root.",
    )
    p.add_argument(
        "--csv-root",
        type=str,
        default="data",
        help="Dataset root for resolving paths in --labeled-csv (default: data).",
    )
    p.add_argument(
        "--csv-images-subdir",
        type=str,
        default="test_data",
        help="Subdir of --csv-root used as flat image fallback (default: test_data).",
    )
    p.add_argument(
        "--verify-csv-paths",
        action="store_true",
        help="Stat every path in --labeled-csv at startup (slower).",
    )
    p.add_argument(
        "--find-depth",
        type=int,
        default=8,
        help="Max BFS depth when locating class subfolders under dataset root.",
    )
    p.add_argument(
        "--class-map",
        action="append",
        default=[],
        metavar="FolderName:idx",
        help="ImageFolder only: map folder name -> model class index (0 or 1). Repeatable.",
    )
    p.add_argument(
        "--csv-image-column",
        type=str,
        default="",
        help="--labeled-csv: image column name (default: infer).",
    )
    p.add_argument(
        "--csv-label-column",
        type=str,
        default="",
        help="--labeled-csv: label column name (default: infer).",
    )
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
    )
    p.add_argument(
        "--out-json",
        type=str,
        default="",
        help="Optional path to write metrics JSON (default: artifacts/eval_external_<stamp>.json).",
    )
    p.add_argument(
        "--log-every",
        type=int,
        default=50,
        help="Log val progress every N batches (0 = quiet).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ckpt_path = Path(args.checkpoint).resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    use_csv = bool(args.labeled_csv.strip())
    raw: Path
    root: Path | None = None
    ds_probe: datasets.ImageFolder | None = None
    remap: list[int] = []
    csv_items: list[tuple[Path, int]] | None = None
    labeled_csv_path: Path | None = None

    if use_csv:
        labeled_csv_path = Path(args.labeled_csv.strip()).resolve()
        if not labeled_csv_path.is_file():
            raise FileNotFoundError(f"--labeled-csv not found: {labeled_csv_path}")
        cr = args.csv_root.strip() or "data"
        raw = Path(cr).resolve() if Path(cr).is_absolute() else (Path.cwd() / cr).resolve()
        if not raw.is_dir():
            raise FileNotFoundError(f"--csv-root is not a directory: {raw}")
        images_dir = raw / args.csv_images_subdir.strip()
        if not images_dir.is_dir():
            raise FileNotFoundError(
                f"--csv-images-subdir not found under csv-root: {images_dir}"
            )
        labeled, _img_c, _lbl_c = _collect_labeled_cells(
            labeled_csv_path,
            image_column=args.csv_image_column or None,
            label_column=args.csv_label_column or None,
        )
        unique_labels = sorted({lab for _r, lab in labeled})
        label_to_idx = {name: i for i, name in enumerate(unique_labels)}
        if len(label_to_idx) != 2:
            raise ValueError(
                f"Expected exactly 2 distinct labels in CSV, got {unique_labels!r}"
            )
        csv_items = _paths_from_labeled(
            labeled,
            raw,
            images_dir,
            csv_path=labeled_csv_path,
            label_to_idx=label_to_idx,
            verify_exists=args.verify_csv_paths,
        )
        print(f"Labeled CSV: {labeled_csv_path}")
        print(f"CSV root: {raw} (images fallback dir: {images_dir})")
        print(f"Samples: {len(csv_items)}; label strings -> idx: {label_to_idx}")
    else:
        raw = Path(args.dataset_path.strip()).resolve()
        if not raw.exists():
            raise FileNotFoundError(f"--dataset-path not found: {raw}")

        root = find_imagefolder_root(raw, max_depth=args.find_depth)
        print(f"ImageFolder root: {root}")
        print(f"Folders (ImageFolder order): {datasets.ImageFolder(root=str(root)).classes}")

        explicit = _parse_class_maps(args.class_map) if args.class_map else {}
        ds_probe = datasets.ImageFolder(root=str(root))
        remap = _resolve_folder_targets(list(ds_probe.classes), explicit)

    device = resolve_device(args.device)
    print(f"Device: {device}")

    checkpoint = load_checkpoint(ckpt_path, device=device)

    ckpt_classes: list[str] = list(checkpoint.get("class_names", ["Real", "Fake"]))
    num_classes = len(ckpt_classes)
    if num_classes != 2:
        raise ValueError(f"Expected binary checkpoint (2 classes), got {ckpt_classes}")

    image_size = int(checkpoint.get("image_size", 224))
    architecture = str(checkpoint.get("architecture", "efficientnet_b0"))
    temperature = float(checkpoint.get("temperature", 1.0))

    model, _ = create_classifier(architecture, num_classes=num_classes, device=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    _, eval_tfms = create_transforms(image_size)
    if use_csv:
        assert csv_items is not None
        eval_ds = RealFakePathsDataset(csv_items, eval_tfms)
    else:
        assert root is not None and ds_probe is not None
        eval_ds = _RemappedImageFolder(root, eval_tfms, remap)
    loader = DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    if not use_csv:
        assert ds_probe is not None
        print(
            "Folder -> model target index: "
            + ", ".join(
                f"{ds_probe.classes[i]}->{remap[i]}" for i in range(len(remap))
            )
        )
    print(f"Checkpoint classes (row/col order in CM): {ckpt_classes}")

    criterion = nn.CrossEntropyLoss()

    # apply temperature inside forward by wrapping — loop uses raw model; scale logits in eval
    class _ScaledModel(nn.Module):
        def __init__(self, inner: nn.Module, t: float):
            super().__init__()
            self.inner = inner
            self.t = max(t, 1e-8)

        def forward(self, x):
            return self.inner(x) / self.t

    eval_model = _ScaledModel(model, temperature)

    metrics = evaluate_metrics(
        model=eval_model,
        loader=loader,
        criterion=criterion,
        device=device,
        num_classes=num_classes,
        class_names=ckpt_classes,
        batch_size=args.batch_size,
        amp_enabled=device.type == "cuda",
        log_every=args.log_every,
    )

    print("\n--- External eval ---")
    print(f"Samples: {len(eval_ds)}")
    print(f"Loss:           {metrics['loss']:.4f}")
    print(f"Accuracy:       {metrics['acc']:.4f}")
    print(f"Balanced acc:   {metrics['balanced_acc']:.4f}")
    print(f"Macro F1:       {metrics['macro_f1']:.4f}")
    if "per_class_accuracy" in metrics:
        print("Per-class recall (true label):")
        for k, v in metrics["per_class_accuracy"].items():
            print(f"  {k}: {v:.4f}")
    print("Confusion (rows=true, cols=pred):")
    print_confusion_matrix(metrics["confusion_matrix"], ckpt_classes)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = (
        Path(args.out_json)
        if args.out_json
        else Path("artifacts") / f"eval_external_{stamp}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict = {
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(ckpt_path),
        "dataset_path": str(raw),
        "checkpoint_class_names": ckpt_classes,
        "metrics": {k: v for k, v in metrics.items() if k != "confusion_matrix"},
        "confusion_matrix": metrics["confusion_matrix"],
    }
    if use_csv:
        payload["labeled_csv"] = str(labeled_csv_path)
        payload["csv_root"] = str(raw)
        payload["num_samples"] = len(csv_items) if csv_items else 0
    else:
        payload["imagefolder_root"] = str(root)
        assert ds_probe is not None
        payload["folder_classes"] = ds_probe.classes
        payload["folder_to_target"] = {
            ds_probe.classes[i]: remap[i] for i in range(len(remap))
        }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
