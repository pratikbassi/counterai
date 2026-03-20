from __future__ import annotations

"""
CLI entry point for training the binary Real-vs-Fake classifier.
"""

import argparse
import json
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import kagglehub
import torch
from PIL import Image
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .data import build_train_val_datasets
from .loop import evaluate, evaluate_confusion_matrix, train_one_epoch
from .model import create_resnet18_classifier
from .transforms import create_transforms


@dataclass
class TrainConfig:
    batch_size: int = 32
    epochs: int = 8
    lr: float = 1e-4
    weight_decay: float = 1e-4
    num_workers: int = 2
    val_split: float = 0.2
    image_size: int = 224
    seed: int = 42
    out_dir: str = "artifacts"
    dataset_slug: str = "antorbosuantu/asa-real-fake-dataset"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train real-vs-fake image classifier")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, default="artifacts")
    parser.add_argument(
        "--log-every",
        type=int,
        default=50,
        help="Print training/eval status every N batches (0 disables).",
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable mixed precision (auto-defaults to enabled on CUDA).",
    )
    parser.add_argument(
        "--disable-decompression-bomb-warning",
        action="store_true",
        help="Suppress PIL DecompressionBombWarning by allowing large images.",
    )
    parser.add_argument(
        "--dataset-slug",
        type=str,
        default="antorbosuantu/asa-real-fake-dataset",
        help="KaggleHub dataset slug",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = TrainConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        val_split=args.val_split,
        image_size=args.image_size,
        seed=args.seed,
        out_dir=args.out_dir,
        dataset_slug=args.dataset_slug,
    )

    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if args.disable_decompression_bomb_warning:
        Image.MAX_IMAGE_PIXELS = None
        warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)

    amp_enabled = device.type == "cuda" and (
        args.amp if args.amp is not None else True
    )

    print(f"Downloading dataset from kagglehub: {cfg.dataset_slug}")
    dataset_path = Path(kagglehub.dataset_download(cfg.dataset_slug))
    print(f"Dataset downloaded to: {dataset_path}")

    train_tfms, eval_tfms = create_transforms(cfg.image_size)
    datasets_info = build_train_val_datasets(
        dataset_path,
        train_tfms=train_tfms,
        eval_tfms=eval_tfms,
        val_split=cfg.val_split,
        seed=cfg.seed,
    )

    train_ds = datasets_info["train_ds"]
    val_ds = datasets_info["val_ds"]
    class_names = datasets_info["class_names"]
    num_classes = datasets_info["num_classes"]
    imagefolder_root = datasets_info["imagefolder_root"]
    data_loading = datasets_info["data_loading"]

    print(f"Classes: {class_names}")
    if data_loading.get("mode") == "train_test_real_fake":
        print(
            "Detected Train/Test layout with Real/Fake subfolders; "
            "using `train/` as training split and `test/` as validation split."
        )
        print(f"Train dir: {data_loading.get('train_dir')}")
        print(f"Test dir: {data_loading.get('test_dir')}")
    else:
        print(f"Using ImageFolder root: {imagefolder_root}")

    loader_kwargs = {
        "batch_size": cfg.batch_size,
        "num_workers": cfg.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": cfg.num_workers > 0,
    }
    if cfg.num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    model = create_resnet18_classifier(num_classes=num_classes, device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = out_dir / "best_real_fake_resnet18.pt"
    metrics_path = out_dir / "train_metrics.json"

    best_val_acc = -1.0
    best_epoch = 0
    history: list[dict] = []

    for epoch in range(1, cfg.epochs + 1):
        start = time.time()

        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            batch_size=cfg.batch_size,
            amp_enabled=amp_enabled,
            log_every=args.log_every,
        )
        val_loss, val_acc = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            batch_size=cfg.batch_size,
            amp_enabled=amp_enabled,
            log_every=args.log_every,
        )

        elapsed = time.time() - start
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "seconds": elapsed,
        }
        history.append(row)

        print(
            f"Epoch {epoch}/{cfg.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | {elapsed:.1f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "class_names": class_names,
                    "image_size": cfg.image_size,
                    "val_acc": best_val_acc,
                },
                best_model_path,
            )
            print(f"Saved new best model -> {best_model_path}")

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "config": cfg.__dict__,
                "dataset_path": str(dataset_path),
                "imagefolder_root": str(imagefolder_root),
                "data_loading": data_loading,
                "class_names": class_names,
                "history": history,
                "best_val_acc": best_val_acc,
                "best_epoch": best_epoch,
                "best_model_path": str(best_model_path),
            },
            f,
            indent=2,
        )
    print(f"Saved training metrics -> {metrics_path}")

    # Detailed evaluation on the validation split using the best checkpoint.
    print(
        f"Best validation: epoch={best_epoch} acc={best_val_acc:.4f} "
        f"(checkpoint={best_model_path})"
    )
    print("\nDetailed validation evaluation (best checkpoint):")

    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])

    detailed = evaluate_confusion_matrix(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=device,
        num_classes=num_classes,
        class_names=class_names,
        batch_size=cfg.batch_size,
        amp_enabled=amp_enabled,
    )

    print(f"Validation loss: {detailed['loss']:.4f}")
    print(f"Validation acc:  {detailed['acc']:.4f}")
    print("Per-class accuracy (recall by true class):")
    for cls_name, acc in detailed["per_class_accuracy"].items():
        print(f"  {cls_name}: {acc:.4f}")

    print("Confusion matrix (rows=true, cols=pred):")
    cm = detailed["confusion_matrix"]
    header = " " * 10 + "  ".join([f"pred:{n}" for n in class_names])
    print(header)
    for i, row in enumerate(cm):
        print(f"true:{class_names[i]} " + "  ".join(str(x) for x in row))


if __name__ == "__main__":
    main()

