# Restore `data/source_labels.csv`

The labeled test manifest is **not** in git (everything under `data/` is ignored).

If you need it again, copy from the Kaggle bundle: **`test_v2_labels.csv`** (columns `id`, `label`). Kaggle paths use the prefix **`test_data_v2/`**; this repo expects **`test_data/`**. After copying to `model/data/source_labels.csv`, rewrite prefixes:

```bash
sed -i 's|test_data_v2/|test_data/|g' model/data/source_labels.csv
```

Then run eval:

```bash
cd model
make eval-external EVAL_ARGS="--labeled-csv data/source_labels.csv --csv-root data --checkpoint artifacts/best_real_fake.pt"
```
