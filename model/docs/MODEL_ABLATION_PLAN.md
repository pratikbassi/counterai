# Model ablation plan (Real vs Fake)

Living document: update the **Results log** tables after each run. Keep the **fixed protocol** the same unless you are explicitly studying that variable.

## Context

| Item | Notes |
|------|--------|
| Dataset | Phases A–E: ~4k train (ASA Real/Fake; local copy). **Phase F:** large **AI vs human** set under flat `model/data/` (CSV + flat folders; see Phase F). **Held-out eval:** `data/source_labels.csv` + `test_data/` (`docs/DATA_LAYOUT.md`, `docs/RESTORE_TEST_LABELS.md`). |
| Hardware | ~12 GB VRAM; **throughput defaults** in this doc assume **many CPU cores** + spare RAM (`--num-workers 16`). If you OOM, lower `--batch-size` first. |
| Trainer | `model/` → `make train ARGS="..."` |
| Architectures in code today | `--architecture`: `resnet18`, `resnet50`, `efficientnet_b0`, `convnext_tiny`, `efficientnet_v2_s`, `vit_b_16` (`train/model.py`). Newer backbones pin `IMAGENET1K_V1` weights to keep ImageNet mean/std + 224-crop preprocessing (avoids ViT-B/16's `DEFAULT` SWAG variant which uses mean=0.5/std=0.5 + crop=384). |

## Goals

1. Compare backbones **fairly** (same data, same epoch budget unless OOM).
2. Record **val_acc**, **balanced_acc**, **macro_f1**, and confusion matrix (from console + `artifacts/train_metrics_*.json`).
3. Note **OOM**, **slow steps**, or **instability** (NaN, collapse).

---

## Fixed protocol (Phase A — default “fair” run)

Use for every row in **Phase A** unless the row explicitly overrides a cell.

| Setting | Suggested value | Notes |
|---------|-----------------|--------|
| AMP | On (CUDA default) | `--amp` or omit (defaults on for CUDA) |
| `--image-size` | `224` | First pass; Phase B scales up for finalists |
| `--batch-size` | **High-throughput defaults** in command blocks (tuned to fill ~12 GB when you were ~2.5 GB @ batch 30) | If OOM → reduce (e.g. 128→96→64) and **record** actual batch in the results table |
| `--num-workers` | `16` in all commands below | Raise/lower if CPU is oversubscribed or data stalls the GPU |
| `--epochs` | `12` | Same for all Phase A rows |
| `--head-epochs` | `2` | Staged head warmup |
| `--backbone-lr` | `1e-5` | |
| `--lr` | `1e-4` | Head / classifier LR |
| `--scheduler` | `cosine` | |
| `--early-stopping-patience` | `4` | Use `0` to always run full `--epochs` |
| `--early-stopping-min-delta` | `0.001` | |
| `--early-stopping-metric` | **Pick one:** `macro_f1` *or* `balanced_acc` | Use **one** metric for all Phase A; prefer `balanced_acc` if classes are imbalanced |
| Augmentation | Either **default** *or* `--strong-aug-preset` | Pick one policy for all Phase A |
| Seed | `42` | Use `--extra-seeds 43,44` only for **finalists** (Phase E) |

**Artifacts produced each run (automatic)**

- `artifacts/train_metrics_<stamp>.json` — full history + `detailed_validation`
- `artifacts/best_real_fake_<stamp>.pt` — archived best weights for that run
- `artifacts/best_real_fake.pt` — latest best (overwritten)

**Per-run checklist**

- [ ] Note peak VRAM (`nvidia-smi`) if training is tight.
- [ ] If OOM: lower `--batch-size`, re-run, document batch in the table.
- [ ] Copy **best epoch**, **monitor metric**, and **confusion matrix** into the results row (or link to the JSON path).

---

## Phase A — Implemented backbones (single-model sweep)

Run in order; complete one row per run.

| # | Architecture | `make train` architecture flag | Batch | Image | Val macro_f1 | Val bal_acc | Val acc | OOM? | Notes |
|---|--------------|--------------------------------|-------|-------|--------------|-------------|---------|------|-------|
| A1 | ResNet-18 | `--architecture resnet18` | 32 | 224 | 0.7845 | 0.7845 | 0.7847 | No | Baseline; val loss 0.4602; recall Real 0.774 / Fake 0.795; CM [[774,226],[205,795]]; `artifacts/train_metrics_20260320_223027.json` |
| A2 | ResNet-50 | `--architecture resnet50` | 32 | 224 | 0.7845 | 0.7845 | 0.7847 | No | val loss 0.4601; recall Real 0.774 / Fake 0.795; CM [[774,226],[205,795]]; `artifacts/train_metrics_20260320_231936.json` |
| A3 | EfficientNet-B0 | `--architecture efficientnet_b0` | 32 | 224 | 0.7974 | 0.7975 | 0.7981 | No | Phase A best macro_f1; val loss 0.4378; recall Real 0.816 / Fake 0.779; CM [[816,184],[221,779]]; `artifacts/train_metrics_20260321_014823.json` |

**Phase A — copy-paste `make` commands** (run from the **`model/`** directory)

Shared fixed protocol: 12 epochs, head 2, cosine, early stop on **macro_f1** (patience 4, min_delta 0.001), image 224, seed 42 (default).

**Throughput:** `--num-workers 16` and **larger batches** per row (ResNet-50 is capped below ResNet-18). If you OOM, halve batch for that row only.

**A1 — ResNet-18 (baseline)**

```bash
cd model
make train ARGS="--architecture resnet18 --epochs 12 --batch-size 128 --num-workers 16 --head-epochs 2 --backbone-lr 1e-5 --lr 1e-4 --scheduler cosine --early-stopping-metric macro_f1 --early-stopping-patience 4 --early-stopping-min-delta 0.001"
```

**A2 — ResNet-50**

```bash
cd model
make train ARGS="--architecture resnet50 --epochs 12 --batch-size 96 --num-workers 16 --head-epochs 2 --backbone-lr 1e-5 --lr 1e-4 --scheduler cosine --early-stopping-metric macro_f1 --early-stopping-patience 4 --early-stopping-min-delta 0.001"
```

**A3 — EfficientNet-B0** (if OOM, try `--batch-size 96` then `64`)

```bash
cd model
make train ARGS="--architecture efficientnet_b0 --epochs 12 --batch-size 112 --num-workers 16 --head-epochs 2 --backbone-lr 1e-5 --lr 1e-4 --scheduler cosine --early-stopping-metric macro_f1 --early-stopping-patience 4 --early-stopping-min-delta 0.001"
```

After each run, note the new `artifacts/train_metrics_*.json` path (and optionally paste key metrics into the table above).

---

## Phase B — Resolution sweep (top 1–2 from Phase A only)

Repeat only the best one or two architectures. Increase image size until VRAM or batch size becomes impractical.

| # | Architecture | `--image-size` | Batch | Val macro_f1 | Val bal_acc | Notes |
|---|--------------|------------------|-------|--------------|-------------|-------|
| B1 | efficientnet_b0 | 288 | 16 | 0.8213 | 0.8215 | val loss 0.4049; recall Real 0.855 / Fake 0.788; CM [[855,145],[212,788]]; `artifacts/train_metrics_20260321_023755.json` |
| B2 | efficientnet_b0 | 320 | 64 | 0.7974 | 0.7985 | val loss 0.4435; recall Real 0.872 / Fake 0.725; CM [[872,128],[275,725]]; `artifacts/train_metrics_20260321_033557.json` |

**Stop if** batch must drop below ~4 for stability; prefer a slightly smaller model over an unstable tiny batch.

**B1 — EfficientNet-B0 @ 288** (same throughput stance: fill GPU; reduce batch if OOM)

```bash
cd model
make train ARGS="--architecture efficientnet_b0 --image-size 288 --epochs 12 --batch-size 80 --num-workers 16 --head-epochs 2 --backbone-lr 1e-5 --lr 1e-4 --scheduler cosine --early-stopping-metric macro_f1 --early-stopping-patience 4 --early-stopping-min-delta 0.001"
```

**B2 — EfficientNet-B0 @ 320** (batch 64; you measured ~2.5 GB @ 30 — room to scale; try 80 if stable)

```bash
cd model
make train ARGS="--architecture efficientnet_b0 --image-size 320 --epochs 12 --batch-size 64 --num-workers 16 --head-epochs 2 --backbone-lr 1e-5 --lr 1e-4 --scheduler cosine --early-stopping-metric macro_f1 --early-stopping-patience 4 --early-stopping-min-delta 0.001"
```

---

## Phase C — Training extras (fixed winning architecture)

Run only after Phase A champion is clear. One variable at a time is ideal.

| # | Variant | Extra ARGS | Val macro_f1 | Val bal_acc | Notes |
|---|---------|------------|--------------|-------------|-------|
| C1 | EMA | `--ema` | 0.7285 | 0.7295 | val loss 0.5547; recall Real 0.669 / Fake 0.790; CM [[669,331],[210,790]]; `artifacts/train_metrics_20260321_112316.json` |
| C2 | Imbalance (if needed) | `--class-weights` and/or `--balance-sampler` | 0.7964 | 0.7970 | val loss 0.4464; best epoch 12; recall Real 0.851 / Fake 0.743; CM [[851,149],[257,743]]; `artifacts/train_metrics_20260321_115756.json` |
| C3 | Label smoothing | `--label-smoothing 0.05` | 0.7960 | 0.7965 | val loss 0.4782; recall Real 0.848 / Fake 0.745; CM [[848,152],[255,745]]; `artifacts/train_metrics_20260321_122309.json` |
| C4 | LR on plateau | `--scheduler plateau --plateau-patience 2` | 0.8224 | 0.8225 | val loss 0.4009; best epoch 11; recall Real 0.844 / Fake 0.801; CM [[844,156],[199,801]]; `artifacts/train_metrics_20260321_125908.json` |
| C5 | Temperature (calibration) | `--fit-temperature` | 0.7974 | 0.7980 | val loss 0.4464; T≈0.9799 in ckpt; recall Real 0.852 / Fake 0.744; CM [[852,148],[256,744]]; below C4 on macro_f1; `artifacts/train_metrics_20260321_133634.json` |

**Phase C commands (one variable at a time) — based on current winner B1 (`efficientnet_b0`, `image-size 288`)**

Baseline for C-runs:

```bash
cd model
make train ARGS="--architecture efficientnet_b0 --image-size 288 --epochs 12 --batch-size 80 --num-workers 16 --head-epochs 2 --backbone-lr 1e-5 --lr 1e-4 --scheduler cosine --early-stopping-metric macro_f1 --early-stopping-patience 4 --early-stopping-min-delta 0.001"
```

C1 — EMA:

```bash
cd model
make train ARGS="--architecture efficientnet_b0 --image-size 288 --epochs 12 --batch-size 80 --num-workers 16 --head-epochs 2 --backbone-lr 1e-5 --lr 1e-4 --scheduler cosine --early-stopping-metric macro_f1 --early-stopping-patience 4 --early-stopping-min-delta 0.001 --ema"
```

C2 — class weights:

```bash
cd model
make train ARGS="--architecture efficientnet_b0 --image-size 288 --epochs 12 --batch-size 80 --num-workers 16 --head-epochs 2 --backbone-lr 1e-5 --lr 1e-4 --scheduler cosine --early-stopping-metric macro_f1 --early-stopping-patience 4 --early-stopping-min-delta 0.001 --class-weights"
```

C3 — label smoothing:

```bash
cd model
make train ARGS="--architecture efficientnet_b0 --image-size 288 --epochs 12 --batch-size 80 --num-workers 16 --head-epochs 2 --backbone-lr 1e-5 --lr 1e-4 --scheduler cosine --early-stopping-metric macro_f1 --early-stopping-patience 4 --early-stopping-min-delta 0.001 --label-smoothing 0.05"
```

C4 — plateau scheduler:

```bash
cd model
make train ARGS="--architecture efficientnet_b0 --image-size 288 --epochs 12 --batch-size 80 --num-workers 16 --head-epochs 2 --backbone-lr 1e-5 --lr 1e-4 --scheduler plateau --plateau-patience 2 --early-stopping-metric macro_f1 --early-stopping-patience 4 --early-stopping-min-delta 0.001"
```

C5 — temperature fit:

```bash
cd model
make train ARGS="--architecture efficientnet_b0 --image-size 288 --epochs 12 --batch-size 80 --num-workers 16 --head-epochs 2 --backbone-lr 1e-5 --lr 1e-4 --scheduler cosine --early-stopping-metric macro_f1 --early-stopping-patience 4 --early-stopping-min-delta 0.001 --fit-temperature"
```

---

## Phase D — Future backbones (extend `train/model.py`)

When new models are added to the trainer, add rows here **before** running.

| # | Architecture | Status | Notes |
|---|--------------|--------|--------|
| D1 | `convnext_tiny` | **In repo** (~28M params); see Phase G | Strong CNN baseline for fine-tuning |
| D2 | `efficientnet_v2_s` | **In repo** (~20M params); see Phase G | Good accuracy/size tradeoff |
| D3 | e.g. `vit_b_16` | Not in repo | Watch batch size on 12 GB at 224+ |

---

## Phase E — Final verification (best config)

| # | Config summary | Seeds | Mean macro_f1 | Std / spread | Notes |
|---|----------------|-------|---------------|--------------|-------|
| E1 | `efficientnet_b0` @ 288, batch 80, workers 16, `scheduler=plateau`, ASA Real/Fake (~4k) | `42,43,44` | **0.8213** | range **0.0021** (0.8224→0.8203) | seed42 macro_f1=0.8224 `train_metrics_20260321_140305_seed42.json`; seed43=0.8213 `..._seed43.json`; seed44=0.8203 `..._seed44.json`. Phase F (local large data) F1+F2 logged — see Phase F. |

---

## Phase F — Richer data (AI vs human, large scale)

**Status:** **F1 + F2 complete** (local flat `model/data/`). Next steps → see **After Phase F** below.

**Goal:** Re-run the **same hyperparameter recipe as E1** (EfficientNet-B0 @ 288, plateau scheduler, macro_f1 early stopping) on the **larger AI vs human** distribution.

### Data + validation (read this before comparing metrics)

| Item | Phase F (local flat `model/data/`) |
|------|-------------------------------|
| Train | All labeled rows in `train.csv` → images under `train_data/` (paths root-relative, e.g. `train_data/….jpg`) |
| Validation | **Stratified holdout from train** (default `--val-split 0.2`, seed per run). Used because `test.csv` is **paths-only** (no labels). |
| `test.csv` / `test_data/` | Paths-only test listing (no labels); **not** used for val metrics. |
| Startup | Large CSV parse still costs **~20s+**; **`--no-verify-csv-paths`** skips per-row `stat` (recommended). |

**CLI gotcha:** the scheduler flag is **`--scheduler plateau`** (two tokens). **`--schedulerplateau`** is invalid and may fall back to a default scheduler.

### F1 — single seed (42), local data, E1 recipe

Tune **`--batch-size`** down (e.g. 64 → 48) if you OOM at 288² on ~12 GB. Epochs will be **much longer** than Phases A–E because of dataset size.

```bash
cd model
make train ARGS="--no-verify-csv-paths --architecture efficientnet_b0 --image-size 288 --epochs 12 --batch-size 80 --num-workers 16 --head-epochs 2 --backbone-lr 1e-5 --lr 1e-4 --scheduler plateau --plateau-patience 2 --early-stopping-metric macro_f1 --early-stopping-patience 4 --early-stopping-min-delta 0.001"
```

### F2 — multi-seed (42, 43, 44), same as F1

```bash
cd model
make train ARGS="--no-verify-csv-paths --architecture efficientnet_b0 --image-size 288 --epochs 12 --batch-size 80 --num-workers 16 --head-epochs 2 --backbone-lr 1e-5 --lr 1e-4 --scheduler plateau --plateau-patience 2 --early-stopping-metric macro_f1 --early-stopping-patience 4 --early-stopping-min-delta 0.001 --extra-seeds 43,44"
```

| # | Data source | Val protocol | Notes | Val macro_f1 | Val bal_acc | `train_metrics_*.json` |
|---|-------------|--------------|-------|--------------|-------------|-------------------------|
| F1 | Local `data/` | 20% stratified holdout from `train.csv` | Standalone run **stamp 181050**; seed **42**; best ep **10**; val_loss **0.0339**; recall 0/1 **0.9940** / **0.9819**; CM [[7947,48],[145,7850]]; `best_real_fake_20260321_181050.pt` | **0.9879** | **0.9879** | `artifacts/train_metrics_20260321_181050.json` |
| F2 | Local `data/` | same (each seed → its own stratified split) | One job **stamp 184439**; seed **42** ep10 macro_f1 **0.9878**; seed **43** ep11 **0.9906**; seed **44** ep8 **0.9885** (early stop); mean macro_f1 **0.9890**; range **0.0028** (0.9878→0.9906); `best_real_fake_20260321_184439_seed{42,43,44}.pt` | **0.9890** | **0.9890** | `train_metrics_20260321_184439_seed42.json`, `…_seed43.json`, `…_seed44.json` |

### Phase F — multi-seed summary (F2 only)

| # | Config summary | Seeds | Mean macro_f1 | Spread | Notes |
|---|----------------|-------|-----------------|--------|-------|
| F2 | Same as F1 + `--extra-seeds 43,44` | `42,43,44` | **0.9890** | **0.0028** (0.9878→0.9906) | Best single seed: **43** (0.9906). Variance similar in scale to Phase E1 on ASA. |

**Tradeoff:** Phase F numbers are **not comparable** to Phases A–E: **different data**, **different validation** (holdout from train vs fixed test split on ASA). Treat F as a **new benchmark line**. **F1 vs F2** (standalone seed 42 vs batch seed 42) can differ slightly because the multi-seed job uses one **shared** `metrics_stamp` and any codepath differences; treat **F2 mean** as the official multi-seed line for this dataset.

### After Phase F — suggested next steps

1. **Choose a shipping checkpoint** — e.g. `best_real_fake_20260321_184439_seed43.pt` (best val macro_f1 in F2) or prefer seed 42 for parity with earlier protocol.
2. **External / held-out labeled benchmark** — `make eval-external` with `--labeled-csv data/source_labels.csv --csv-root data` (see `docs/RESTORE_TEST_LABELS.md` if the CSV is missing) or ImageFolder `--dataset-path` for other bundles (`model/README.md`, `docs/DATA_LAYOUT.md`). For ImageFolder, verify folder→class mapping vs checkpoint semantics; use `--class-map` if heuristics miss. Expect **lower** macro-F1 than Phase F holdout; treat as **domain-shift** signal, not a replacement for val.
3. **Calibration** — optional `--fit-temperature` (Phase C5 style) on the Phase F checkpoint if you need calibrated probabilities.
4. **Harder reality check** — spot-label a few hundred `test_data/` images (or mix in external images) and measure **off-protocol** error; expect **lower** scores than ~0.99 holdout.

---

## Phase G — New backbones on Phase F dataset (large AI vs human)

**Status:** Backbones added in code (`convnext_tiny`, `efficientnet_v2_s`, `vit_b_16`); rows below are **screening templates** — fill in `Val macro_f1` / `Val bal_acc` / metrics path after each run. The training data has grown to **~140k labeled rows** (balanced ~70k / ~70k) since Phase F2.

### Goal

Re-anchor the EfficientNet-B0 baseline on the larger dataset, screen each new backbone with a **single seed**, then promote the winner to a 3-seed run for variance estimation. Mirror Phase F's recipe (plateau scheduler + macro_f1 early stopping) so numbers can be stacked against F1/F2.

### Fixed protocol (apply to every Phase G row unless the row overrides)

| Setting | Value | Notes |
|---------|-------|-------|
| `--no-verify-csv-paths` | yes | Skip per-row `stat` on the ~140k CSV (~20s+ savings at startup) |
| `--epochs` | 12 | |
| `--head-epochs` | 2 | Staged head warmup |
| `--lr` / `--backbone-lr` | `1e-4` / `1e-5` | |
| `--scheduler plateau --plateau-patience 2` | yes | Same as F1/F2 |
| `--early-stopping-metric macro_f1 --early-stopping-patience 4 --early-stopping-min-delta 0.001` | yes | |
| `--num-workers` | 16 | Tune to CPU |
| Seed | 42 | (`--extra-seeds 43,44` only for the G6 winner row) |
| Augmentation | default | Keep parity with F |
| AMP | on (CUDA default) | |

**OOM playbook:** halve `--batch-size` for that row only and **record the actual batch** you used in the row.

### Per-architecture screening (single seed = 42)

| # | Architecture | Image | Batch | Notes / VRAM caveat |
|---|--------------|-------|-------|---------------------|
| G1 | `efficientnet_b0` | 288 | 80 | **Re-anchor** of Phase F2 winner on the new ~140k data; reference row for the others. |
| G2 | `resnet50` | 224 | 96 | Sanity check on a CNN baseline at the new scale. |
| G3 | `convnext_tiny` | 224 | 64 | ~28M params; halve batch if OOM. |
| G4 | `efficientnet_v2_s` | 288 | 48 | Heavier than B0; halve to 32 if OOM. |
| G5 | `vit_b_16` | **224 (fixed)** | 64 | Positional embeddings tied to 14x14 patches @ patch-size 16; do **not** sweep image size. ~86M params; halve to 32 (or smaller) if OOM. |

### Results table (fill after each run)

| # | Arch | Image | Batch | Val macro_f1 | Val bal_acc | Best epoch | val_loss | Per-class recall (0/1) | CM | Notes / `train_metrics_*.json` |
|---|------|-------|-------|--------------|-------------|------------|----------|------------------------|----|--------------------------------|
| G1 | efficientnet_b0 | 288 | 80 | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| G2 | resnet50 | 224 | 96 | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| G3 | convnext_tiny | 224 | 64 | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| G4 | efficientnet_v2_s | 288 | 48 | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| G5 | vit_b_16 | 224 | 64 | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

### Phase G — copy-paste `make` commands (run from `model/`)

**G1 — EfficientNet-B0 @ 288 (re-anchor)**

```bash
cd model
make train ARGS="--no-verify-csv-paths --architecture efficientnet_b0 --image-size 288 --epochs 12 --batch-size 80 --num-workers 16 --head-epochs 2 --backbone-lr 1e-5 --lr 1e-4 --scheduler plateau --plateau-patience 2 --early-stopping-metric macro_f1 --early-stopping-patience 4 --early-stopping-min-delta 0.001"
```

**G2 — ResNet-50 @ 224**

```bash
cd model
make train ARGS="--no-verify-csv-paths --architecture resnet50 --image-size 224 --epochs 12 --batch-size 96 --num-workers 16 --head-epochs 2 --backbone-lr 1e-5 --lr 1e-4 --scheduler plateau --plateau-patience 2 --early-stopping-metric macro_f1 --early-stopping-patience 4 --early-stopping-min-delta 0.001"
```

**G3 — ConvNeXt-Tiny @ 224**

```bash
cd model
make train ARGS="--no-verify-csv-paths --architecture convnext_tiny --image-size 224 --epochs 12 --batch-size 64 --num-workers 16 --head-epochs 2 --backbone-lr 1e-5 --lr 1e-4 --scheduler plateau --plateau-patience 2 --early-stopping-metric macro_f1 --early-stopping-patience 4 --early-stopping-min-delta 0.001"
```

**G4 — EfficientNetV2-S @ 288**

```bash
cd model
make train ARGS="--no-verify-csv-paths --architecture efficientnet_v2_s --image-size 288 --epochs 12 --batch-size 48 --num-workers 16 --head-epochs 2 --backbone-lr 1e-5 --lr 1e-4 --scheduler plateau --plateau-patience 2 --early-stopping-metric macro_f1 --early-stopping-patience 4 --early-stopping-min-delta 0.001"
```

**G5 — ViT-B/16 @ 224 (fixed)**

```bash
cd model
make train ARGS="--no-verify-csv-paths --architecture vit_b_16 --image-size 224 --epochs 12 --batch-size 64 --num-workers 16 --head-epochs 2 --backbone-lr 1e-5 --lr 1e-4 --scheduler plateau --plateau-patience 2 --early-stopping-metric macro_f1 --early-stopping-patience 4 --early-stopping-min-delta 0.001"
```

### G6 — Multi-seed run on the G1–G5 winner

After picking the row with the best `Val macro_f1` from G1–G5, append `--extra-seeds 43,44` to that row's command (and substitute the winning architecture / image / batch). Example assuming G3 wins:

```bash
cd model
make train ARGS="--no-verify-csv-paths --architecture convnext_tiny --image-size 224 --epochs 12 --batch-size 64 --num-workers 16 --head-epochs 2 --backbone-lr 1e-5 --lr 1e-4 --scheduler plateau --plateau-patience 2 --early-stopping-metric macro_f1 --early-stopping-patience 4 --early-stopping-min-delta 0.001 --extra-seeds 43,44"
```

| # | Config summary | Seeds | Mean macro_f1 | Spread | Notes |
|---|----------------|-------|---------------|--------|-------|
| G6 | _winner from G1–G5_ | `42,43,44` | _TBD_ | _TBD_ | Compare against Phase F2 mean (0.9890) on the same data layout. |

### After Phase G — suggested next steps

1. **Promote the winning checkpoint** — copy/keep `artifacts/best_real_fake_<stamp>_seed<best>.pt` and update `DETECTOR_JOB_TODO.md` (item 1: configurable checkpoint path) to point production to it.
2. **External / held-out labeled benchmark** — `make eval-external EVAL_ARGS="--labeled-csv data/source_labels.csv --csv-root data --checkpoint artifacts/best_real_fake_<stamp>_seed<best>.pt"`. Expect lower macro-F1 than the holdout — that gap is the domain-shift signal, not a regression.
3. **Calibration (optional)** — re-run the winner with `--fit-temperature` (Phase C5 style) if downstream code wants calibrated probabilities.
4. **Image-size sweep on the winner** — only if the winner is **not** ViT (ViT is locked at 224); try `--image-size 320` if VRAM permits.

### Performance and scaling tradeoffs (Phase G)

- **Single-seed screening (G1–G5) before multi-seed (G6)** is roughly 5x faster than naive "3 seeds for everyone"; only the winner pays the multi-seed cost.
- **Pinned `IMAGENET1K_V1` weights** for the new backbones (rather than `DEFAULT`) keep the existing transform pipeline (ImageNet mean/std @ 224 crop) usable. Tradeoff: leaves SWAG-pretrained ViT performance on the table; the alternative (a second 384-crop transform path) would have inflated training time on ~140k images.
- **Fixed image size per arch** (no per-arch resolution sweep) avoids combinatorial blowup. Resolution sweeps are reserved for the Phase G winner.
- **Subprocess-per-image inference is unchanged** — `classify.py` reads the `architecture` field from the checkpoint, so any G winner drops in via `artifacts/best_real_fake.pt` and `DetectorJob` keeps working without code changes.

---

## Decision rules

1. **Primary:** best **macro_f1** or **balanced_acc** (whichever you fixed in Phase A) on the **same validation protocol** — not raw accuracy alone if classes skew.
2. **Tie-break:** simpler / faster model, or better **recall on Fake** (inspect confusion matrix) if that matches product risk.
3. **Document:** path to `train_metrics_*.json` and `best_real_fake_*.pt` for the chosen run.

---

## Changelog

| Date | Author | Change |
|------|--------|--------|
| 2026-03-19 | — | Initial plan |
| 2026-03-20 | — | Phase A1 (ResNet-18) results logged |
| 2026-03-21 | — | Phase A3 (EfficientNet-B0) results logged; Phase B1 queued |
| 2026-03-21 | — | Phase B1 (EfficientNet-B0 @ 288) results logged; Phase B2 next |
| 2026-03-21 | — | Phase B2 (EfficientNet-B0 @ 320) results logged; B1 remains best |
| 2026-03-21 | — | Raised default `--batch-size` / `--num-workers` in all command blocks for high-throughput hardware |
| 2026-03-21 | — | Phase C1 (EMA) results logged; significant drop vs B1 baseline |
| 2026-03-21 | — | Phase C2 (class-weights) results logged; still below B1 baseline |
| 2026-03-21 | — | Phase C3 (label smoothing) results logged; still below B1 baseline |
| 2026-03-21 | — | Phase C4 (plateau scheduler) results logged; new best so far |
| 2026-03-21 | — | Phase C5 (fit-temperature on cosine B1-style run) logged; next: Phase E on C4 |
| 2026-03-21 | — | Added Phase F (large AI vs human dataset; local CSV layout) |
| 2026-03-21 | — | Phase E1 (3 seeds, plateau @ 288) results logged; Phase F ready to run |
| 2026-03-21 | — | Phase F: documented `--local-data-dir` + flat `model/data/` CSV layout |
| 2026-03-21 | — | Phase F: primary = local large data; val = stratified train holdout; copy-paste F1/F2 + `--scheduler plateau` note |
| 2026-03-21 | — | Phase F1 results logged (macro_f1=0.9879, local data, seed 42) |
| 2026-03-21 | — | Phase F2 multi-seed logged (mean macro_f1≈0.9890, stamp `184439`); “After Phase F” next steps added |
| 2026-03-21 | — | Removed `predict_csv.py` / `make predict-test` (no test labels / no submission flow) |
| 2026-03-21 | — | `eval_external_dataset.py` + `make eval-external`; README + “After Phase F” Shutterstock benchmark step |
| 2026-03-21 | — | Removed Kaggle download tooling (`kagglehub`, wget script); training default `data/`; manual drops under `data/` |
| 2026-03-22 | — | Flat `data/` (no `training/`/`testing/` subdirs, no symlinks); eval via `--labeled-csv` + `--csv-root`; removed `build_v2_labeled_imagefolder.py` / `make rebuild-testing-v2` |
| 2026-03-22 | — | Renamed default val image dir `test_data_v2` → `test_data` (CLI defaults + docs; rewrite CSV paths + folder on disk) |
| 2026-04-19 | — | Added `convnext_tiny`, `efficientnet_v2_s`, `vit_b_16` to `train/model.py` (pinned `IMAGENET1K_V1` weights). `train.csv` grew from ~60k → ~140k labeled rows (balanced). Added Phase G (re-anchor + 3 new backbones screened single-seed, winner promoted to 3 seeds). Updated Phase D status table. |
