# Local `data/` layout (flat)

Everything under `model/data/` is **gitignored**. Use a **single flat directory** (default `--local-data-dir data` when run from `model/`):

```
model/data/
  train.csv           # labeled training rows (required)
  train_data/         # flat training images (required)
  test.csv            # optional: paths-only or labeled val CSV
  test_data/          # optional: flat images referenced by test.csv / eval CSV
  source_labels.csv   # optional: labeled held-out list for eval (see RESTORE_TEST_LABELS.md)
```

**Minimum requirement:** `train.csv` + `train_data/`. When `test.csv` / `test_data/` are absent, the trainer automatically does a stratified train/val split from `train.csv` (controlled by `--val-split`, default 20%).

No symlinks, no `training/` / `testing/` subfolders.

## Held-out eval (CSV)

Place **`source_labels.csv`** here (e.g. Kaggle `test_v2_labels.csv` renamed). Columns: image id/path + label (same conventions as training CSVs).

```bash
cd model
make eval-external EVAL_ARGS="--labeled-csv data/source_labels.csv --csv-root data --checkpoint artifacts/best_real_fake.pt"
```

Add `--verify-csv-paths` to stat every row at startup (slower).

## ImageFolder eval (other benchmarks)

Unzip a dataset somewhere and:

```bash
make eval-external EVAL_ARGS="--dataset-path path/to/imagefolder_root --checkpoint artifacts/best_real_fake.pt"
```
