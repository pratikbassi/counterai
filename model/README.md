# Image Detection Model

A PyTorch-based image detection project that uses pre-trained object detection models to detect objects in images.

## Features

- **Pre-trained Models**: Uses Faster R-CNN with ResNet-50 backbone (COCO dataset)
- **Flexible Input**: Accepts images as file paths, PIL Images, numpy arrays, or bytes
- **GPU Support**: Automatically uses CUDA if available
- **Configurable Confidence**: Adjustable confidence threshold for detections
- **Clean API**: Easy-to-use Python API and CLI interface

## Installation

1. Create and activate a virtual environment (recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

2. Install Python dependencies:

**For CPU-only installation:**
```bash
pip install -r requirements.txt
```

**For GPU support (CUDA 12.8):**
```bash
pip install torch>=2.9.1 torchvision>=0.24.0 torchaudio>=2.9.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt --no-deps
pip install Pillow>=11.0.0 opencv-python>=4.10.0 numpy>=2.1.0 requests>=2.32.0
```

**Note:** Ensure you have NVIDIA Driver release 570 or later and CUDA 12.8.1+ for GPU support.

3. The model weights will be automatically downloaded on first use.

## Usage

### Train a real-vs-fake classifier

Training always reads from a **local directory** (no automatic Kaggle downloads). Default is **`data/`** when you run from `model/` (`--local-data-dir`).

**Layout:** put `train.csv`, `test.csv`, and flat folders `train_data/` and `test_data/` under **`model/data/`** (defaults; override with `--csv-*` flags). Paths in CSVs are usually **relative to that root** (e.g. `train_data/uuid.jpg`). If `test.csv` is **paths-only** (no label column), the trainer builds a **stratified validation holdout from `train.csv`** and only checks that unlabeled test paths exist for sanity; use **`--no-verify-csv-paths`** on large sets to skip per-row `stat` during startup (failures show up when an image is loaded). Optional **`source_labels.csv`** in the same folder for held-out eval (see **[docs/DATA_LAYOUT.md](docs/DATA_LAYOUT.md)** and **[docs/RESTORE_TEST_LABELS.md](docs/RESTORE_TEST_LABELS.md)**). The whole `data/` tree is gitignored.

#### Ingest `upload/AI` and `upload/nonAI` into `train.csv`

Small helper **[ingest_upload_to_train.py](ingest_upload_to_train.py)** (stdlib only): in one run it **lists all pending** images under `data/upload/AI` then `data/upload/nonAI` (sorted), then processes them in **batches** (default 256): copy each file in the batch, append that many rows to `train.csv`, then rename each source to `uploaded_<name>`. Filenames already starting with `uploaded` (any case) are skipped. Progress goes to **stderr** at INFO; use **`-v`** for per-batch timings. **`--batch-size`** changes batch size. CSV shape: ``, `file_name`, `label` (index, `train_data/...`, **`0`** = not AI, **`1`** = AI).

```bash
cd model
python3 ingest_upload_to_train.py
python3 ingest_upload_to_train.py --batch-size 512
python3 ingest_upload_to_train.py -v
python3 ingest_upload_to_train.py --dry-run
```

The trainer still reads `label` as text (`"0"` / `"1"`); that matches your existing `train.csv`.

```bash
cd model
make train
# equivalent:
make train ARGS="--local-data-dir data ..."
```

To compare backbones systematically (`resnet18`, `resnet50`, `efficientnet_b0`, …), follow **[docs/MODEL_ABLATION_PLAN.md](docs/MODEL_ABLATION_PLAN.md)**.

Optional arguments:

```bash
make train ARGS="--epochs 12 --batch-size 64 --lr 0.0002 --out-dir artifacts"
```

Improved fine-tuning run (recommended):

```bash
make train ARGS="--epochs 12 --batch-size 64 --num-workers 12 --head-epochs 2 --backbone-lr 1e-5 --scheduler cosine"
```

Outputs:
- `artifacts/best_real_fake.pt` (rolling best checkpoint; includes `architecture`, optional `temperature`)
- `artifacts/best_real_fake_resnet18.pt` (duplicate mirror when training **ResNet-18** only, for older scripts)
- `artifacts/best_real_fake_<metrics_stamp>.pt` (archived copy of that run’s best weights)
- `artifacts/train_metrics.json` (always overwritten: latest run’s full metrics)
- `artifacts/train_metrics_YYYYMMDD_HHMMSS.json` (and `..._seedN` when using `--extra-seeds`)
- After training: per-class recall, balanced accuracy, macro-F1, and confusion matrix on val.

Notes:
- The script auto-discovers an `ImageFolder`-compatible dataset root (`class_a/`, `class_b/`, ...), or **Train/Test + Real/Fake** folders.
- Default backbone is **ResNet-18**; use `--architecture resnet50` or `efficientnet_b0` for more capacity (slower / larger checkpoints).
- Staged fine-tuning: classifier head warmup, then backbone + head (ResNet: `layer4` [+ optional `layer3`]; EfficientNet: `features` + classifier).
- Transforms use ImageNet normalization; optional **RandAugment**, **JPEG recompression**, and **strong** ColorJitter.
- **Stratified** train/val split for ImageFolder fallback (disable with `--no-stratified-val-split`).
- **Class imbalance:** `--class-weights` (loss) and/or `--balance-sampler` (WeightedRandomSampler).
- **Label smoothing:** `--label-smoothing 0.05` (CrossEntropyLoss).
- **EMA:** `--ema` smooths weights for eval and the saved checkpoint.
- **Scheduler:** `--scheduler cosine` (default) or `plateau` (ReduceLROnPlateau on the monitored metric).
- **Early stopping / best checkpoint** follow `--early-stopping-metric`: `val_acc`, `balanced_acc`, or `macro_f1`.
- **Temperature scaling:** `--fit-temperature` refits a scalar T on val; `classify.py` divides logits by T.
- **Multi-seed runs:** `--extra-seeds 43,44` (sequential runs, separate metrics files).
- For throughput, tune `--num-workers` and `--batch-size`.
- If you see `PIL.Image.DecompressionBombWarning`, use `--disable-decompression-bomb-warning`.

Example (stronger aug + macro-F1 early stopping + EMA):

```bash
make train ARGS="--strong-aug-preset --early-stopping-metric macro_f1 --ema --epochs 20 --early-stopping-patience 5"
```

Early stopping usage example:

```bash
make train ARGS="--epochs 20 --early-stopping-patience 4 --early-stopping-min-delta 0.001 --early-stopping-metric balanced_acc"
```

Patience in this project means:
- After each epoch, we compare the chosen `--early-stopping-metric` against the best so far.
- An epoch only counts as an improvement if `metric > best_metric + min_delta`.
- `patience=N` allows up to `N` consecutive non-improving epochs before stopping.
- `--early-stopping-patience 0` disables early stopping.

### Classify (inference)

Default checkpoint path is `artifacts/best_real_fake.pt`. Older ResNet-18-only runs may use `artifacts/best_real_fake_resnet18.pt`.

### Evaluate on an external labeled dataset

**CSV + flat folder** (same path rules as training): default eval for `source_labels.csv` next to `test_data/`:

```bash
cd model
make eval-external EVAL_ARGS="--labeled-csv data/source_labels.csv --csv-root data --checkpoint artifacts/best_real_fake_20260321_184439_seed43.pt"
```

**ImageFolder:** unzip a benchmark and pass `--dataset-path`; folder names are mapped to class `0`/`1` (see `--class-map` if needed).

```bash
make eval-external EVAL_ARGS="--dataset-path data/other_benchmark --checkpoint artifacts/best_real_fake_20260321_184439_seed43.pt"
```

Metrics JSON is written under `artifacts/eval_external_*.json` by default.

If folder names are ambiguous, pass explicit maps (repeat `--class-map`):

```bash
make eval-external EVAL_ARGS="--dataset-path /path/to/extracted --checkpoint artifacts/best_real_fake_20260321_184439_seed43.pt --class-map Human:0 --class-map AI:1"
```

### Command Line Interface

Run detection on an image file:

```bash
python detect.py image.jpg
```

With custom confidence threshold:

```bash
python detect.py image.jpg --confidence 0.7
```

Save results to JSON file:

```bash
python detect.py image.jpg --output results.json
```

### Python API

```python
from model_handler import ImageDetectionModel

# Initialize the model
detector = ImageDetectionModel()

# Run detection on an image
detections = detector.detect('path/to/image.jpg', confidence_threshold=0.5)

# Process results
for det in detections:
    print(f"Found {det['label']} with confidence {det['score']:.2f}")
    print(f"Bounding box: {det['bbox']}")
```

### Supported Input Types

The `detect()` method accepts:
- **File path** (string): `detector.detect('image.jpg')`
- **PIL Image**: `detector.detect(pil_image)`
- **Numpy array**: `detector.detect(numpy_array)`
- **Bytes**: `detector.detect(image_bytes)`

## Output Format

Each detection returns a dictionary with:
- `bbox`: Bounding box coordinates `[x1, y1, x2, y2]`
- `score`: Confidence score (0.0-1.0)
- `label`: Class label name (e.g., "person", "car", "bicycle")
- `label_id`: Numeric class ID

## Model Details

- **Model**: Faster R-CNN with ResNet-50 FPN backbone
- **Dataset**: Pre-trained on COCO dataset (80 object classes)
- **Device**: Automatically uses GPU (CUDA) if available, otherwise CPU

## Performance Considerations

- **First Run**: Model weights are downloaded (~170MB) on first use
- **GPU Acceleration**: Significantly faster inference on GPU
- **Batch Processing**: Currently processes one image at a time
- **Memory**: Model requires ~2-3GB RAM/VRAM

## Next Steps With the Venv

### Makefile (Linux / macOS / WSL)

From `model/`:

```bash
make help          # list targets
make install       # creates .venv, upgrades pip, installs requirements.txt
make train ARGS="--epochs 8 --batch-size 32 --num-workers 2 --out-dir artifacts"
make detect IMG=path/to/image.jpg
```

GPU install mirrors the README CUDA 12.8 flow: `make install-gpu`.

### Manual workflow

Common day-to-day workflow:

```bash
cd model
source .venv/bin/activate
python -m pip install -r requirements.txt
make train ARGS="--epochs 8 --batch-size 32 --num-workers 2 --out-dir artifacts"
```

When done:

```bash
deactivate
```

Suggested iteration path:
- Start with a short sanity run (`--epochs 1`) to validate environment + dataset access.
- Tune throughput first (`--num-workers`, `--batch-size`) before model changes.
- Keep outputs under `artifacts/` and compare archived `train_metrics_*.json` files across runs (`train_metrics.json` is only the most recent run).

## Extending the Project

To use a different model:

1. Add the model loading logic in `model_handler.py`
2. Update the `_load_model()` method
3. Ensure the model output format matches the expected structure

## Requirements

- Python 3.10+ (Python 3.14 recommended for latest PyTorch)
- PyTorch 2.9.1+ (latest as of December 2025)
- torchvision 0.24.0+
- torchaudio 2.9.1+
- Pillow 11.0+
- numpy 2.1.0+
- opencv-python 4.10.0+
- requests 2.32.0+

## License

This project is part of the CounterAI monorepo.

