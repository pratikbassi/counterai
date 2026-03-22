from __future__ import annotations

"""
CLI entry point for training the binary Real-vs-Fake classifier.
"""

import argparse
import json
import shutil
import time
import warnings
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import torch
from PIL import Image
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, WeightedRandomSampler

from .calibration import fit_temperature_scaling
from .data import build_train_val_datasets, iter_dataset_labels, train_dataset_label_counts
from .ema import ModelEMA, maybe_ema_scope
from .loop import evaluate_confusion_matrix, evaluate_metrics, train_one_epoch
from .model import apply_train_stage, create_classifier
from .transforms import create_transforms
from .utils import load_checkpoint, print_confusion_matrix


def _write_train_metrics(
    out_dir: Path, stamp: str, payload: dict
) -> tuple[Path, Path]:
    """Write the same payload to a timestamped archive and to train_metrics.json."""
    stamped = out_dir / f"train_metrics_{stamp}.json"
    latest = out_dir / "train_metrics.json"
    for path in (stamped, latest):
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    return stamped, latest


def _val_metric_score(metrics: dict, name: str) -> float:
    if name == "val_acc":
        return float(metrics["acc"])
    if name == "balanced_acc":
        return float(metrics["balanced_acc"])
    if name == "macro_f1":
        return float(metrics["macro_f1"])
    raise ValueError(f"Unknown early-stopping metric: {name!r}")


def _build_optimizer(
    model: nn.Module,
    architecture: str,
    *,
    stage: str,
    unfreeze_layer3: bool,
    cfg: TrainConfig,
) -> AdamW:
    arch = architecture.lower()
    if arch in ("resnet18", "resnet50"):
        if stage == "head":
            return AdamW(
                model.fc.parameters(),
                lr=cfg.lr,
                weight_decay=cfg.weight_decay,
            )
        param_groups = []
        if unfreeze_layer3:
            param_groups.append(
                {"params": model.layer3.parameters(), "lr": cfg.backbone_lr}
            )
        param_groups.append(
            {"params": model.layer4.parameters(), "lr": cfg.backbone_lr}
        )
        param_groups.append({"params": model.fc.parameters(), "lr": cfg.lr})
        return AdamW(param_groups, weight_decay=cfg.weight_decay)
    if arch == "efficientnet_b0":
        if stage == "head":
            return AdamW(
                model.classifier.parameters(),
                lr=cfg.lr,
                weight_decay=cfg.weight_decay,
            )
        return AdamW(
            [
                {"params": model.features.parameters(), "lr": cfg.backbone_lr},
                {"params": model.classifier.parameters(), "lr": cfg.lr},
            ],
            weight_decay=cfg.weight_decay,
        )
    raise ValueError(f"Unsupported architecture for optimizer: {architecture!r}")


def _make_scheduler(
    optimizer: AdamW, cfg: TrainConfig, *, epochs_this_stage: int
) -> Any:
    if cfg.scheduler == "cosine":
        return CosineAnnealingLR(optimizer, T_max=max(epochs_this_stage, 1))
    if cfg.scheduler == "plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=cfg.plateau_factor,
            patience=cfg.plateau_patience,
            threshold=1e-4,
        )
    return None


def _scheduler_step(scheduler: Any, cfg: TrainConfig, monitor_value: float) -> None:
    if scheduler is None:
        return
    if cfg.scheduler == "plateau":
        scheduler.step(monitor_value)
    else:
        scheduler.step()


def _balanced_sample_weights(train_ds: torch.utils.data.Dataset, num_classes: int) -> torch.Tensor:
    """Per-sample weights inversing class frequency (for WeightedRandomSampler)."""

    counts = train_dataset_label_counts(train_ds, num_classes).to(torch.float64)
    inv = 1.0 / counts.clamp(min=1.0)
    labels = iter_dataset_labels(train_ds)
    return torch.tensor([float(inv[y]) for y in labels], dtype=torch.double)


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
    head_epochs: int = 2
    backbone_lr: float = 1e-5
    scheduler: str = "cosine"
    early_stopping_patience: int = 3
    early_stopping_min_delta: float = 0.0
    early_stopping_metric: str = "val_acc"
    architecture: str = "resnet18"
    unfreeze_layer3: bool = False
    label_smoothing: float = 0.0
    class_weights: bool = False
    balance_sampler: bool = False
    stratified_val_split: bool = True
    augment_strength: str = "default"
    randaugment: bool = False
    randaugment_num_ops: int = 2
    randaugment_magnitude: int = 9
    jpeg_augment: bool = False
    jpeg_quality_min: int = 40
    jpeg_quality_max: int = 90
    jpeg_prob: float = 0.25
    use_ema: bool = False
    ema_decay: float = 0.999
    plateau_patience: int = 2
    plateau_factor: float = 0.5
    fit_temperature: bool = False


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
    parser.add_argument(
        "--extra-seeds",
        type=str,
        default="",
        help="Comma-separated extra seeds to run after --seed (e.g. '43,44'). Each run writes its own metrics file.",
    )
    parser.add_argument("--out-dir", type=str, default="artifacts")
    parser.add_argument(
        "--architecture",
        type=str,
        default="resnet18",
        help="Backbone: resnet18, resnet50, efficientnet_b0",
    )
    parser.add_argument(
        "--unfreeze-layer3",
        action="store_true",
        help="(ResNet only) Also unfreeze layer3 in the fine-tune stage (with layer4).",
    )
    parser.add_argument(
        "--head-epochs",
        type=int,
        default=2,
        help="Train classifier head only for N epochs, then unfreeze backbone stage.",
    )
    parser.add_argument(
        "--backbone-lr",
        type=float,
        default=1e-5,
        help="LR for unfrozen backbone (ResNet layer3/4 or EfficientNet features).",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="cosine",
        choices=["none", "cosine", "plateau"],
        help="LR scheduler: cosine per stage, or ReduceLROnPlateau on the monitored val metric.",
    )
    parser.add_argument(
        "--plateau-patience",
        type=int,
        default=2,
        help="Epochs with no val improvement before LR reduce (scheduler=plateau).",
    )
    parser.add_argument(
        "--plateau-factor",
        type=float,
        default=0.5,
        help="LR multiply factor on plateau.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=3,
        help="Stop if monitored metric does not improve for N epochs (0 disables).",
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=0.0,
        help="Minimum increase in --early-stopping-metric to count as improvement.",
    )
    parser.add_argument(
        "--early-stopping-metric",
        type=str,
        default="val_acc",
        choices=["val_acc", "balanced_acc", "macro_f1"],
        help="Metric for best checkpoint + early stopping + plateau.",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.0,
        help="Cross-entropy label smoothing (e.g. 0.05). 0 disables.",
    )
    parser.add_argument(
        "--class-weights",
        action="store_true",
        help="Inverse-frequency class weights in the loss (handles imbalance).",
    )
    parser.add_argument(
        "--balance-sampler",
        action="store_true",
        help="WeightedRandomSampler so each batch samples classes more evenly.",
    )
    parser.add_argument(
        "--no-stratified-val-split",
        action="store_true",
        help="Use a single random split instead of per-class stratified (ImageFolder fallback only).",
    )
    parser.add_argument(
        "--augment-strength",
        type=str,
        default="default",
        choices=["default", "strong"],
        help="Train-time ColorJitter strength.",
    )
    parser.add_argument(
        "--randaugment",
        action="store_true",
        help="Enable torchvision RandAugment on training images.",
    )
    parser.add_argument(
        "--randaugment-num-ops",
        type=int,
        default=2,
        help="RandAugment num_ops.",
    )
    parser.add_argument(
        "--randaugment-magnitude",
        type=int,
        default=9,
        help="RandAugment magnitude.",
    )
    parser.add_argument(
        "--jpeg-augment",
        action="store_true",
        help="Random JPEG recompression (simulates social/phone compression).",
    )
    parser.add_argument(
        "--jpeg-quality-min",
        type=int,
        default=40,
    )
    parser.add_argument(
        "--jpeg-quality-max",
        type=int,
        default=90,
    )
    parser.add_argument(
        "--jpeg-prob",
        type=float,
        default=0.25,
        help="Probability to apply JPEG augment when --jpeg-augment is set.",
    )
    parser.add_argument(
        "--strong-aug-preset",
        action="store_true",
        help="Shorthand: strong ColorJitter + RandAugment + JPEG augment.",
    )
    parser.add_argument(
        "--ema",
        action="store_true",
        help="Exponential moving average of weights for eval/saved checkpoint.",
    )
    parser.add_argument(
        "--ema-decay",
        type=float,
        default=0.999,
        help="EMA decay (higher = slower EMA, more smoothing).",
    )
    parser.add_argument(
        "--fit-temperature",
        action="store_true",
        help="After training, fit softmax temperature on val and store in checkpoint.",
    )
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
        "--local-data-dir",
        type=str,
        default="data",
        help=(
            "Directory with training data when running from `model/` "
            "(default: `data`). "
            "Expects train.csv, test.csv, train_data/, test_data/ by default "
            "(override names with --csv-*)."
        ),
    )
    parser.add_argument(
        "--csv-train-image-dir",
        type=str,
        default="train_data",
        help="Subfolder of --local-data-dir with training images (flat).",
    )
    parser.add_argument(
        "--csv-val-image-dir",
        type=str,
        default="test_data",
        help="Subfolder of --local-data-dir with validation/test images (flat).",
    )
    parser.add_argument(
        "--csv-train-file",
        type=str,
        default="train.csv",
        help="Training labels CSV filename inside --local-data-dir.",
    )
    parser.add_argument(
        "--csv-val-file",
        type=str,
        default="test.csv",
        help="Validation labels CSV filename inside --local-data-dir.",
    )
    parser.add_argument(
        "--csv-image-column",
        type=str,
        default="",
        help="CSV column for image filename (default: infer from header).",
    )
    parser.add_argument(
        "--csv-label-column",
        type=str,
        default="",
        help="CSV column for class label (default: infer from header).",
    )
    parser.add_argument(
        "--no-verify-csv-paths",
        action="store_true",
        help=(
            "Skip per-row filesystem checks when loading CSV + flat folders "
            "(much faster for large CSVs; errors surface on first missing image in DataLoader)."
        ),
    )
    return parser.parse_args()


def _config_from_args(args: argparse.Namespace) -> TrainConfig:
    augment_strength = args.augment_strength
    randaugment = args.randaugment
    jpeg_augment = args.jpeg_augment
    if args.strong_aug_preset:
        augment_strength = "strong"
        randaugment = True
        jpeg_augment = True

    return TrainConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        val_split=args.val_split,
        image_size=args.image_size,
        seed=args.seed,
        out_dir=args.out_dir,
        head_epochs=args.head_epochs,
        backbone_lr=args.backbone_lr,
        scheduler=args.scheduler,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        early_stopping_metric=args.early_stopping_metric,
        architecture=args.architecture,
        unfreeze_layer3=args.unfreeze_layer3,
        label_smoothing=args.label_smoothing,
        class_weights=args.class_weights,
        balance_sampler=args.balance_sampler,
        stratified_val_split=not args.no_stratified_val_split,
        augment_strength=augment_strength,
        randaugment=randaugment,
        randaugment_num_ops=args.randaugment_num_ops,
        randaugment_magnitude=args.randaugment_magnitude,
        jpeg_augment=jpeg_augment,
        jpeg_quality_min=args.jpeg_quality_min,
        jpeg_quality_max=args.jpeg_quality_max,
        jpeg_prob=args.jpeg_prob,
        use_ema=args.ema,
        ema_decay=args.ema_decay,
        plateau_patience=args.plateau_patience,
        plateau_factor=args.plateau_factor,
        fit_temperature=args.fit_temperature,
    )


def run_one_training_run(
    cfg: TrainConfig,
    args: argparse.Namespace,
    *,
    metrics_stamp: str,
    device: torch.device,
    amp_enabled: bool,
    dataset_path: Path,
) -> None:
    torch.manual_seed(cfg.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg.seed)

    train_tfms, eval_tfms = create_transforms(
        cfg.image_size,
        augment_strength=cfg.augment_strength,
        randaugment=cfg.randaugment,
        randaugment_num_ops=cfg.randaugment_num_ops,
        randaugment_magnitude=cfg.randaugment_magnitude,
        jpeg_augment=cfg.jpeg_augment,
        jpeg_quality_min=cfg.jpeg_quality_min,
        jpeg_quality_max=cfg.jpeg_quality_max,
        jpeg_prob=cfg.jpeg_prob,
    )
    datasets_info = build_train_val_datasets(
        dataset_path,
        train_tfms=train_tfms,
        eval_tfms=eval_tfms,
        val_split=cfg.val_split,
        seed=cfg.seed,
        stratified_val_split=cfg.stratified_val_split,
        csv_train_image_dir=args.csv_train_image_dir,
        csv_val_image_dir=args.csv_val_image_dir,
        csv_train_file=args.csv_train_file,
        csv_val_file=args.csv_val_file,
        csv_image_column=args.csv_image_column or None,
        csv_label_column=args.csv_label_column or None,
        verify_csv_paths=not args.no_verify_csv_paths,
    )

    train_ds = datasets_info["train_ds"]
    val_ds = datasets_info["val_ds"]
    class_names = datasets_info["class_names"]
    num_classes = datasets_info["num_classes"]
    imagefolder_root = datasets_info["imagefolder_root"]
    data_loading = datasets_info["data_loading"]
    train_label_counts = datasets_info["train_label_counts"]

    print(f"Classes: {class_names}")
    print(f"Train label counts: {train_label_counts.tolist()}")
    if data_loading.get("mode") == "train_test_real_fake":
        print(
            "Detected Train/Test layout with Real/Fake subfolders; "
            "using `train/` as training split and `test/` as validation split."
        )
        print(f"Train dir: {data_loading.get('train_dir')}")
        print(f"Test dir: {data_loading.get('test_dir')}")
    elif data_loading.get("mode") == "csv_flat_folders":
        vs = data_loading.get("val_strategy", "labeled_val_csv")
        print("Detected CSV + flat image folders.")
        print(f"Train CSV: {data_loading.get('train_csv')}")
        print(f"Train images: {data_loading.get('train_image_dir')}")
        if data_loading.get("val_csv"):
            print(f"Val CSV: {data_loading['val_csv']}")
        if data_loading.get("val_image_dir"):
            print(f"Val images: {data_loading['val_image_dir']}")
        print(f"Val strategy: {vs}")
        if vs == "train_holdout_stratified":
            detail = f"holdout fraction {data_loading.get('val_split')!r}"
            n_unl = data_loading.get("unlabeled_test_paths_count")
            if n_unl is not None:
                detail += f"; unlabeled test list: {n_unl} paths"
            print(f"  ({detail})")
    else:
        print(f"Using ImageFolder root: {imagefolder_root}")
        print(f"Val split mode: {data_loading.get('mode')}")

    loader_kwargs: dict = {
        "batch_size": cfg.batch_size,
        "num_workers": cfg.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": cfg.num_workers > 0,
    }
    if cfg.num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2

    if cfg.balance_sampler:
        sample_w = _balanced_sample_weights(train_ds, num_classes)
        sampler = WeightedRandomSampler(
            sample_w,
            num_samples=len(sample_w),
            replacement=True,
            generator=torch.Generator().manual_seed(cfg.seed),
        )
        train_loader = DataLoader(
            train_ds, sampler=sampler, shuffle=False, **loader_kwargs
        )
    else:
        train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)

    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    model, arch = create_classifier(
        cfg.architecture, num_classes=num_classes, device=device
    )
    apply_train_stage(
        model,
        arch,
        stage="head",
        unfreeze_layer3=False,
    )

    weight_tensor: Optional[torch.Tensor] = None
    if cfg.class_weights:
        c = train_label_counts.float().to(device)
        weight_tensor = c.sum() / (c.clamp(min=1.0) * num_classes)

    criterion = nn.CrossEntropyLoss(
        weight=weight_tensor,
        label_smoothing=cfg.label_smoothing,
    )

    optimizer = _build_optimizer(
        model, arch, stage="head", unfreeze_layer3=False, cfg=cfg
    )
    scheduler = _make_scheduler(
        optimizer, cfg, epochs_this_stage=max(cfg.head_epochs, 1)
    )

    ema: Optional[ModelEMA] = None
    if cfg.use_ema:
        ema = ModelEMA(model, decay=cfg.ema_decay)
        print(f"EMA enabled (decay={cfg.ema_decay})")

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = out_dir / "best_real_fake.pt"
    legacy_resnet_path = out_dir / "best_real_fake_resnet18.pt"

    best_metric = -1.0
    best_epoch = 0
    epochs_since_improvement = 0
    history: list[dict] = []
    early_stopped = False

    for epoch in range(1, cfg.epochs + 1):
        if epoch == (cfg.head_epochs + 1):
            apply_train_stage(
                model,
                arch,
                stage="full",
                unfreeze_layer3=cfg.unfreeze_layer3,
            )
            optimizer = _build_optimizer(
                model,
                arch,
                stage="full",
                unfreeze_layer3=cfg.unfreeze_layer3,
                cfg=cfg,
            )
            remaining_epochs = max(cfg.epochs - cfg.head_epochs, 1)
            scheduler = _make_scheduler(optimizer, cfg, epochs_this_stage=remaining_epochs)
            detail = (
                f"layer3+layer4+fc (backbone_lr={cfg.backbone_lr})"
                if cfg.unfreeze_layer3 and arch in ("resnet18", "resnet50")
                else (
                    f"layer4+fc (backbone_lr={cfg.backbone_lr})"
                    if arch in ("resnet18", "resnet50")
                    else f"features+classifier (backbone_lr={cfg.backbone_lr})"
                )
            )
            print(f"Switched to fine-tuning stage: unfroze {detail}")

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
            ema=ema,
        )

        with maybe_ema_scope(model, ema):
            val_m = evaluate_metrics(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                num_classes=num_classes,
                batch_size=cfg.batch_size,
                amp_enabled=amp_enabled,
                log_every=args.log_every,
            )
        val_loss = val_m["loss"]
        val_acc = val_m["acc"]
        val_balanced = val_m["balanced_acc"]
        val_f1 = val_m["macro_f1"]
        monitor = _val_metric_score(val_m, cfg.early_stopping_metric)

        elapsed = time.time() - start
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_balanced_acc": val_balanced,
            "val_macro_f1": val_f1,
            "seconds": elapsed,
        }
        history.append(row)

        print(
            f"Epoch {epoch}/{cfg.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            f"bal_acc={val_balanced:.4f} macro_f1={val_f1:.4f} "
            f"monitor({cfg.early_stopping_metric})={monitor:.4f} | {elapsed:.1f}s"
        )
        _scheduler_step(scheduler, cfg, monitor)

        if monitor > (best_metric + cfg.early_stopping_min_delta):
            best_metric = monitor
            best_epoch = epoch
            epochs_since_improvement = 0

            with maybe_ema_scope(model, ema):
                payload_sd = {
                    k: v.detach().cpu() for k, v in model.state_dict().items()
                }
            ckpt_payload = {
                "state_dict": payload_sd,
                "class_names": class_names,
                "image_size": cfg.image_size,
                "architecture": arch,
                "val_acc": val_m["acc"],
                "val_balanced_acc": val_m["balanced_acc"],
                "val_macro_f1": val_m["macro_f1"],
                "monitor_metric": cfg.early_stopping_metric,
                "monitor_value": best_metric,
                "temperature": 1.0,
            }
            torch.save(ckpt_payload, best_model_path)
            if arch == "resnet18":
                torch.save(ckpt_payload, legacy_resnet_path)
            print(f"Saved new best model -> {best_model_path} (monitor={best_metric:.4f})")
        else:
            epochs_since_improvement += 1
            if cfg.early_stopping_patience > 0:
                print(
                    f"No improvement on {cfg.early_stopping_metric} "
                    f"(patience {epochs_since_improvement}/{cfg.early_stopping_patience})."
                )
                if epochs_since_improvement >= cfg.early_stopping_patience:
                    print(
                        "Early stopping triggered: "
                        f"no improvement greater than {cfg.early_stopping_min_delta:.6f} "
                        f"for {cfg.early_stopping_patience} epoch(s)."
                    )
                    early_stopped = True
                    break

    payload: dict = {
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "metrics_timestamp": metrics_stamp,
        "config": cfg.__dict__,
        "dataset_path": str(dataset_path),
        "imagefolder_root": str(imagefolder_root),
        "data_loading": data_loading,
        "class_names": class_names,
        "train_label_counts": train_label_counts.tolist(),
        "history": history,
        "best_monitor_metric": cfg.early_stopping_metric,
        "best_monitor_value": best_metric,
        "best_epoch": best_epoch,
        "best_val_acc": next(
            (h["val_acc"] for h in history if h["epoch"] == best_epoch), None
        ),
        "best_model_path": str(best_model_path),
        "early_stopped": early_stopped,
        "epochs_completed": len(history),
        "detailed_validation": None,
        "temperature": 1.0,
    }
    _write_train_metrics(out_dir, metrics_stamp, payload)

    if not best_model_path.exists():
        raise RuntimeError(
            "No checkpoint was saved (no training epochs or no val improvement). "
            "Increase --epochs or check your setup."
        )

    print(
        f"Best validation: epoch={best_epoch} "
        f"{cfg.early_stopping_metric}={best_metric:.4f} "
        f"(checkpoint={best_model_path})"
    )
    print("\nDetailed validation evaluation (best checkpoint):")

    checkpoint = load_checkpoint(best_model_path, device=device)
    model.load_state_dict(checkpoint["state_dict"])

    with maybe_ema_scope(model, ema):
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
    print(f"Balanced acc:    {detailed['balanced_acc']:.4f}")
    print(f"Macro F1:        {detailed['macro_f1']:.4f}")
    print("Per-class accuracy (recall by true class):")
    for cls_name, acc in detailed["per_class_accuracy"].items():
        print(f"  {cls_name}: {acc:.4f}")

    print("Confusion matrix (rows=true, cols=pred):")
    print_confusion_matrix(detailed["confusion_matrix"], class_names)

    temperature = float(checkpoint.get("temperature", 1.0))
    if cfg.fit_temperature:
        temperature = fit_temperature_scaling(
            model,
            val_loader,
            device=device,
            amp_enabled=amp_enabled,
        )
        checkpoint["temperature"] = temperature
        torch.save(checkpoint, best_model_path)
        if arch == "resnet18":
            torch.save(checkpoint, legacy_resnet_path)
        print(f"Fitted temperature T={temperature:.4f} (saved in checkpoint)")

    archived_ckpt = out_dir / f"best_real_fake_{metrics_stamp}.pt"
    shutil.copy2(best_model_path, archived_ckpt)
    print(f"Archived checkpoint -> {archived_ckpt}")

    payload["detailed_validation"] = detailed
    payload["saved_at"] = datetime.now(timezone.utc).isoformat()
    payload["temperature"] = temperature
    stamped_path, latest_path = _write_train_metrics(out_dir, metrics_stamp, payload)
    print(
        f"Saved training metrics -> {stamped_path} (per-run archive) "
        f"and {latest_path} (latest)"
    )


def main() -> None:
    args = parse_args()
    base_cfg = _config_from_args(args)

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

    seeds = [base_cfg.seed]
    if args.extra_seeds.strip():
        seeds.extend(
            int(x.strip())
            for x in args.extra_seeds.split(",")
            if x.strip().isdigit()
        )

    base_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    raw = args.local_data_dir.strip()
    if not raw:
        raise ValueError("--local-data-dir cannot be empty")
    dataset_path = (
        Path(raw).resolve() if Path(raw).is_absolute() else (Path.cwd() / raw).resolve()
    )
    if not dataset_path.is_dir():
        raise FileNotFoundError(
            f"--local-data-dir is not a directory: {dataset_path}"
        )
    print(f"Using local dataset directory: {dataset_path}")

    for run_seed in seeds:
        cfg = replace(base_cfg, seed=run_seed)
        metrics_stamp = f"{base_stamp}_seed{run_seed}" if len(seeds) > 1 else base_stamp
        if len(seeds) > 1:
            print(f"\n========== Training with seed={run_seed} ==========\n")
        run_one_training_run(
            cfg,
            args,
            metrics_stamp=metrics_stamp,
            device=device,
            amp_enabled=amp_enabled,
            dataset_path=dataset_path,
        )


if __name__ == "__main__":
    main()
