from __future__ import annotations

import csv
import os
from collections import defaultdict
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".jpe",
    ".jfif",
    ".png",
    ".bmp",
    ".webp",
    ".gif",
    ".tif",
    ".tiff",
    ".heic",
    ".heif",
    ".avif",
    ".ppm",
    ".pgm",
    ".pbm",
}

# CSV + flat image folders (e.g. AI vs human Kaggle layout)
_DEFAULT_TRAIN_IMAGE_DIR = "train_data"
_DEFAULT_VAL_IMAGE_DIR = "test_data"
_DEFAULT_TRAIN_CSV = "train.csv"
_DEFAULT_VAL_CSV = "test.csv"
_CSV_IMAGE_HEADER_CANDIDATES = (
    "filename",
    "image",
    "image_id",
    "id",
    "file_name",
    "filepath",
    "path",
    "img",
    "file",
)
_CSV_LABEL_HEADER_CANDIDATES = (
    "label",
    "target",
    "class",
    "category",
    "type",
    "y",
)


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


def _infer_csv_columns(fieldnames: list[str]) -> tuple[str, str]:
    # Prefer longer / more specific names before short ones (e.g. "file_name" before "id").
    lower = {f.strip().lower(): f.strip() for f in fieldnames if f}
    candidates = sorted(
        _CSV_IMAGE_HEADER_CANDIDATES, key=len, reverse=True
    )
    img_col: Optional[str] = None
    for c in candidates:
        if c in lower:
            img_col = lower[c]
            break
    if img_col is None:
        for f in fieldnames:
            fl = (f or "").lower()
            if any(x in fl for x in ("jpg", "jpeg", "png", "path", "file", "image", "img")):
                img_col = f.strip()
                break
    if img_col is None and len(fieldnames) >= 2:
        img_col = fieldnames[0].strip()
    if img_col is None:
        raise ValueError("Could not infer image/filename column from CSV header.")

    lbl_col: Optional[str] = None
    for c in _CSV_LABEL_HEADER_CANDIDATES:
        if c in lower:
            lbl_col = lower[c]
            break
    if lbl_col is None:
        for f in fieldnames:
            fs = f.strip()
            if fs and fs != img_col and fs.lower() not in ("index", "idx", "unnamed: 0"):
                lbl_col = fs
                break
    if lbl_col is None:
        raise ValueError("Could not infer label column from CSV header.")
    return img_col, lbl_col


def _looks_like_image_path_cell(s: str) -> bool:
    t = s.strip()
    if not t:
        return False
    low = t.replace("\\", "/").lower()
    if "/" in low:
        return True
    return Path(t).suffix.lower() in IMAGE_EXTENSIONS


def _coerce_relative_image_path(dataset_root: Path, images_dir: Path, cell: str) -> Path:
    """
    Map a CSV cell to a path **without** ``stat()`` (for large CSVs when verify is off).

    Heuristic: multi-segment or nested paths → under ``dataset_root``; bare filename →
    ``images_dir / name``.
    """

    cell = cell.strip().strip('"').strip("'")
    if not cell:
        raise ValueError("Empty image path cell in CSV.")
    rel = Path(cell)
    if rel.is_absolute():
        return rel
    # ``foo.jpg`` → flat folder; ``train_data/foo.jpg`` → root-relative
    if rel.parent != Path("."):
        return dataset_root / rel
    return images_dir / rel.name


def _resolve_image_path(
    dataset_root: Path,
    images_dir: Path,
    cell: str,
    *,
    verify_exists: bool = True,
) -> Path:
    """
    Resolve a CSV cell to an on-disk image.

    Many competition CSVs store paths **relative to the dataset root**
    (e.g. ``train_data/uuid.jpg``). We also accept basenames inside ``images_dir``.
    """

    cell = cell.strip().strip('"').strip("'")
    if not cell:
        raise ValueError("Empty image path cell in CSV.")
    p = Path(cell)
    if p.is_absolute():
        resolved = p.resolve()
        if verify_exists and not resolved.is_file():
            raise FileNotFoundError(f"Image not found: {resolved} (CSV value was {cell!r})")
        return resolved

    rel = Path(cell)
    if not verify_exists:
        return _coerce_relative_image_path(dataset_root, images_dir, cell)

    candidates = [
        dataset_root / rel,
        images_dir / rel,
        images_dir / rel.name,
    ]
    seen: set[Path] = set()
    for cand in candidates:
        if cand in seen:
            continue
        seen.add(cand)
        if cand.is_file():
            return cand
    raise FileNotFoundError(
        f"Image not found for {cell!r}. Tried: {candidates[0]}, {candidates[1]}, {candidates[2]}"
    )


def _read_csv_rows(csv_path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"No header row in {csv_path}")
        fieldnames = [fn.strip() for fn in reader.fieldnames]
        rows: list[dict[str, str]] = []
        for raw in reader:
            row = {
                (k or "").strip(): (v if v is not None else "").strip()
                for k, v in raw.items()
            }
            rows.append(row)
        return fieldnames, rows


def _collect_labeled_cells(
    csv_path: Path,
    *,
    image_column: Optional[str],
    label_column: Optional[str],
) -> tuple[list[tuple[str, str]], str, str]:
    fieldnames, rows = _read_csv_rows(csv_path)
    img_col = image_column
    lbl_col = label_column
    if img_col is None or lbl_col is None:
        ic, lc = _infer_csv_columns(fieldnames)
        img_col = img_col or ic
        lbl_col = lbl_col or lc

    labeled: list[tuple[str, str]] = []
    for row in rows:
        if not row:
            continue
        rel = row.get(img_col, "")
        lab = row.get(lbl_col, "")
        if not rel or not lab:
            continue
        labeled.append((rel, str(lab).strip()))

    if not labeled:
        raise RuntimeError(f"No valid rows in {csv_path}")
    return labeled, img_col, lbl_col


def _read_paths_only_csv(csv_path: Path) -> list[str]:
    """
    One column per row (e.g. Kaggle-style ``test.csv`` with only file paths).

    Drops a leading row that looks like a header (``id``, ``path``, …) rather than a path.
    """

    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        raw_rows = [row for row in csv.reader(f) if row and any(c.strip() for c in row)]

    if not raw_rows:
        raise RuntimeError(f"No rows in paths-only CSV: {csv_path}")

    paths: list[str] = []
    start = 0
    if len(raw_rows[0]) == 1:
        first = raw_rows[0][0].strip()
        if first and not _looks_like_image_path_cell(first):
            start = 1
    for row in raw_rows[start:]:
        if len(row) != 1:
            raise ValueError(
                f"Expected a single column in {csv_path}, got {len(row)} cells: {row!r}"
            )
        cell = row[0].strip()
        if cell:
            paths.append(cell)
    if not paths:
        raise RuntimeError(f"No image paths parsed from {csv_path}")
    return paths


def _val_csv_row_width(val_csv: Path) -> int:
    with val_csv.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.reader(f):
            if row and any(c.strip() for c in row):
                return len(row)
    raise RuntimeError(f"Empty or unreadable val CSV: {val_csv}")


def _paths_from_labeled(
    labeled: list[tuple[str, str]],
    dataset_root: Path,
    images_dir: Path,
    *,
    csv_path: Path,
    label_to_idx: dict[str, int],
    verify_exists: bool = True,
) -> list[tuple[Path, int]]:
    items: list[tuple[Path, int]] = []
    for rel, lab in labeled:
        if lab not in label_to_idx:
            raise ValueError(
                f"Unknown label {lab!r} in {csv_path}; "
                f"expected one of {sorted(label_to_idx.keys())!r}"
            )
        path = _resolve_image_path(
            dataset_root, images_dir, rel, verify_exists=verify_exists
        )
        items.append((path, label_to_idx[lab]))
    items.sort(key=lambda x: str(x[0]))
    return items


def _label_counts(items: list[tuple[Path, int]], num_classes: int) -> torch.Tensor:
    counts = torch.zeros(num_classes, dtype=torch.long)
    for _path, y in items:
        counts[int(y)] += 1
    return counts


def discover_csv_folder_dataset(
    root: Path,
    *,
    train_image_dir: str = _DEFAULT_TRAIN_IMAGE_DIR,
    val_image_dir: str = _DEFAULT_VAL_IMAGE_DIR,
    train_csv_name: str = _DEFAULT_TRAIN_CSV,
    val_csv_name: str = _DEFAULT_VAL_CSV,
    image_column: Optional[str] = None,
    label_column: Optional[str] = None,
    verify_csv_paths: bool = True,
) -> Optional[dict]:
    """
    Discover a CSV + flat-image-folder dataset.

    Minimum requirement: ``train.csv`` + ``train_data/`` (or overrides).
    ``test.csv`` / ``test_data/`` are optional — when absent the caller gets
    ``needs_train_val_split=True`` so it can stratify a holdout from the
    training labels.

    When ``test.csv`` exists it may be:
    - labeled (same shape as train) → used as the validation split directly, or
    - paths-only (single column) → noted for sanity but not used for val metrics;
      a holdout split is done from train.csv.
    """

    root = root.resolve()
    train_csv = root / train_csv_name
    train_dir = root / train_image_dir
    if not (train_csv.is_file() and train_dir.is_dir()):
        return None

    val_csv: Optional[Path] = root / val_csv_name
    val_dir: Optional[Path] = root / val_image_dir
    has_val_csv = val_csv.is_file()
    has_val_dir = val_dir.is_dir()
    if not has_val_csv:
        val_csv = None
    if not has_val_dir:
        val_dir = None

    train_labeled, _, _ = _collect_labeled_cells(
        train_csv,
        image_column=image_column,
        label_column=label_column,
    )
    unique_labels = sorted({lab for _r, lab in train_labeled})
    label_to_idx = {name: i for i, name in enumerate(unique_labels)}
    class_names = [n for n, _ in sorted(label_to_idx.items(), key=lambda kv: kv[1])]

    train_items = _paths_from_labeled(
        train_labeled,
        root,
        train_dir,
        csv_path=train_csv,
        label_to_idx=label_to_idx,
        verify_exists=verify_csv_paths,
    )

    needs_train_val_split = False
    holdout_unlabeled_paths: list[Path] = []
    unlabeled_test_paths_count = 0
    val_items: list[tuple[Path, int]] = []

    if not has_val_csv:
        needs_train_val_split = True
    else:
        assert val_csv is not None
        val_ncols = _val_csv_row_width(val_csv)
        if val_ncols < 2:
            needs_train_val_split = True
            rels = _read_paths_only_csv(val_csv)
            if verify_csv_paths and has_val_dir:
                assert val_dir is not None
                holdout_unlabeled_paths = [
                    _resolve_image_path(root, val_dir, rel, verify_exists=True)
                    for rel in rels
                ]
                holdout_unlabeled_paths.sort(key=lambda p: str(p))
            unlabeled_test_paths_count = len(rels)
        else:
            if not has_val_dir:
                raise RuntimeError(
                    f"Val CSV {val_csv} has labels but image dir "
                    f"{root / val_image_dir} does not exist."
                )
            assert val_dir is not None
            val_labeled, _, _ = _collect_labeled_cells(
                val_csv,
                image_column=image_column,
                label_column=label_column,
            )
            val_items = _paths_from_labeled(
                val_labeled,
                root,
                val_dir,
                csv_path=val_csv,
                label_to_idx=label_to_idx,
                verify_exists=verify_csv_paths,
            )

    if not train_items:
        raise RuntimeError("CSV folder dataset produced empty training split.")

    if not needs_train_val_split:
        if not val_items:
            raise RuntimeError("CSV folder dataset produced empty validation split.")

    num_classes = len(class_names)
    train_labels = {y for _p, y in train_items}
    if len(train_labels) < num_classes:
        print(
            f"Warning: not all classes appear in train split; "
            f"saw indices {train_labels} of {num_classes}."
        )
    if not needs_train_val_split:
        val_labels = {y for _p, y in val_items}
        if not val_labels:
            raise RuntimeError("Validation split has no items.")

    return {
        "split_root": root,
        "train_dir": train_dir,
        "test_dir": val_dir,
        "train_items": train_items,
        "val_items": val_items,
        "class_names": class_names,
        "train_csv": train_csv,
        "val_csv": val_csv,
        "needs_train_val_split": needs_train_val_split,
        "holdout_unlabeled_paths": holdout_unlabeled_paths,
        "unlabeled_test_paths_count": unlabeled_test_paths_count,
    }


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


def stratified_train_val_indices(
    targets: list[int],
    *,
    val_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """
    Per-class split so each class contributes ~val_fraction to validation (ImageFolder fallback).

    Tradeoff: slightly more balanced val than a single random_split; needs >=1 train per class.
    """

    by_class: dict[int, list[int]] = defaultdict(list)
    for i, t in enumerate(targets):
        by_class[int(t)].append(i)

    g = torch.Generator().manual_seed(seed)
    train_idx: list[int] = []
    val_idx: list[int] = []

    for _c, idxs in sorted(by_class.items()):
        n = len(idxs)
        if n < 2:
            train_idx.extend(idxs)
            continue
        perm = torch.randperm(n, generator=g).tolist()
        shuffled = [idxs[j] for j in perm]
        n_val = max(1, int(round(n * val_fraction)))
        n_val = min(n_val, n - 1)
        val_idx.extend(shuffled[:n_val])
        train_idx.extend(shuffled[n_val:])

    train_idx.sort()
    val_idx.sort()
    return train_idx, val_idx


def iter_dataset_labels(ds: Dataset) -> list[int]:
    """Fast label extraction without loading images. Supports RealFakePathsDataset, Subset, and .targets."""

    if isinstance(ds, RealFakePathsDataset):
        return [int(y) for _path, y in ds.items]

    if isinstance(ds, torch.utils.data.Subset):
        targets = getattr(ds.dataset, "targets", None)
        if targets is None:
            raise RuntimeError("Subset base dataset has no .targets; cannot extract labels.")
        return [int(targets[i]) for i in ds.indices]

    targets = getattr(ds, "targets", None)
    if targets is None:
        raise RuntimeError("Dataset has no .targets; cannot extract labels.")
    return [int(t) for t in targets]


def train_dataset_label_counts(train_ds: Dataset, num_classes: int) -> torch.Tensor:
    """Per-class sample counts for the training set (for class weights / balanced sampling)."""

    counts = torch.zeros(num_classes, dtype=torch.long)
    for y in iter_dataset_labels(train_ds):
        counts[y] += 1
    return counts


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
    stratified_val_split: bool = True,
    csv_train_image_dir: str = _DEFAULT_TRAIN_IMAGE_DIR,
    csv_val_image_dir: str = _DEFAULT_VAL_IMAGE_DIR,
    csv_train_file: str = _DEFAULT_TRAIN_CSV,
    csv_val_file: str = _DEFAULT_VAL_CSV,
    csv_image_column: Optional[str] = None,
    csv_label_column: Optional[str] = None,
    verify_csv_paths: bool = True,
) -> dict:
    """
    Returns:
      - train_ds, val_ds, class_names, num_classes
      - imagefolder_root + data_loading metadata for train_metrics.json
    """

    csv_split = discover_csv_folder_dataset(
        dataset_path,
        train_image_dir=csv_train_image_dir,
        val_image_dir=csv_val_image_dir,
        train_csv_name=csv_train_file,
        val_csv_name=csv_val_file,
        image_column=csv_image_column,
        label_column=csv_label_column,
        verify_csv_paths=verify_csv_paths,
    )
    if csv_split is not None:
        class_names = csv_split["class_names"]
        num_classes = len(class_names)

        if csv_split["needs_train_val_split"]:
            if not stratified_val_split:
                raise ValueError(
                    "No labeled validation CSV available. Stratified val split from train "
                    "is required; remove --no-stratified-val-split."
                )
            full_items: list[tuple[Path, int]] = csv_split["train_items"]
            targets = [y for _p, y in full_items]
            tr_idx, va_idx = stratified_train_val_indices(
                targets, val_fraction=val_split, seed=seed
            )
            train_items = [full_items[i] for i in tr_idx]
            val_items = [full_items[i] for i in va_idx]
            if len(train_items) < 1 or len(val_items) < 1:
                raise RuntimeError(
                    "Stratified train/val split from train.csv produced an empty split; "
                    "adjust --val-split or check labels per class."
                )
            val_strategy = "train_holdout_stratified"
            if csv_split["val_csv"] is None:
                print(
                    f"No val/test CSV found. "
                    f"Using stratified {val_split:.0%} holdout from train.csv "
                    f"for validation."
                )
            else:
                n_unl = csv_split["unlabeled_test_paths_count"]
                test_dir_name = (
                    csv_split["test_dir"].name
                    if csv_split["test_dir"] is not None
                    else csv_val_image_dir
                )
                print(
                    f"Note: {csv_split['val_csv'].name} is paths-only (no labels). "
                    f"Using stratified {val_split:.0%} holdout from train.csv "
                    f"for validation; {n_unl} unlabeled test image(s) listed for "
                    f"{test_dir_name} (not used for val metrics)."
                )
        else:
            train_items = csv_split["train_items"]
            val_items = csv_split["val_items"]
            val_strategy = "labeled_val_csv"

        train_ds = RealFakePathsDataset(train_items, transform=train_tfms)
        val_ds = RealFakePathsDataset(val_items, transform=eval_tfms)
        if len(train_ds) < 1 or len(val_ds) < 1:
            raise RuntimeError("CSV folder dataset produced empty train or val DataLoader.")

        t_counts = _label_counts(train_items, num_classes)

        data_loading: dict = {
            "mode": "csv_flat_folders",
            "val_strategy": val_strategy,
            "val_split": val_split if csv_split["needs_train_val_split"] else None,
            "train_csv": str(csv_split["train_csv"]),
            "val_csv": str(csv_split["val_csv"]) if csv_split["val_csv"] else None,
            "train_image_dir": str(csv_split["train_dir"]),
            "val_image_dir": str(csv_split["test_dir"]) if csv_split["test_dir"] else None,
        }
        if csv_split["needs_train_val_split"] and csv_split["val_csv"] is not None:
            data_loading["unlabeled_test_paths_count"] = csv_split[
                "unlabeled_test_paths_count"
            ]
        data_loading["verify_csv_paths"] = verify_csv_paths

        return {
            "train_ds": train_ds,
            "val_ds": val_ds,
            "class_names": class_names,
            "num_classes": num_classes,
            "imagefolder_root": csv_split["split_root"],
            "train_label_counts": t_counts,
            "data_loading": data_loading,
        }

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

        t_counts = _label_counts(split_data["train_items"], num_classes)

        return {
            "train_ds": train_ds,
            "val_ds": val_ds,
            "class_names": class_names,
            "num_classes": num_classes,
            "imagefolder_root": split_data["split_root"],
            "train_label_counts": t_counts,
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

    targets_list = list(base_ds.targets)

    if stratified_val_split:
        train_indices, val_indices = stratified_train_val_indices(
            targets_list, val_fraction=val_split, seed=seed
        )
        if len(train_indices) < 1 or len(val_indices) < 1:
            raise RuntimeError("Stratified split produced an empty split. Adjust --val-split.")
        data_loading_meta = {"mode": "stratified_imagefolder_split", "val_split": val_split}
    else:
        val_size = max(1, int(num_samples * val_split))
        train_size = num_samples - val_size
        if train_size < 1:
            raise RuntimeError("Train split too small. Reduce --val-split.")
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(num_samples, generator=g).tolist()
        val_indices = perm[:val_size]
        train_indices = perm[val_size:]
        data_loading_meta = {"mode": "random_imagefolder_split", "val_split": val_split}

    train_ds = datasets.ImageFolder(root=str(root), transform=train_tfms)
    val_ds = datasets.ImageFolder(root=str(root), transform=eval_tfms)
    train_ds = torch.utils.data.Subset(train_ds, train_indices)
    val_ds = torch.utils.data.Subset(val_ds, val_indices)

    t_counts = train_dataset_label_counts(train_ds, num_classes)

    return {
        "train_ds": train_ds,
        "val_ds": val_ds,
        "class_names": class_names,
        "num_classes": num_classes,
        "imagefolder_root": root,
        "train_label_counts": t_counts,
        "data_loading": data_loading_meta,
    }

