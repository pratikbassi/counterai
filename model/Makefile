# CounterAI model — local dev workflow (POSIX: Linux, macOS, WSL).
# Windows CMD: use README PowerShell steps, or run `wsl` and use this Makefile.

.DEFAULT_GOAL := help

PYTHON ?= python3
VENV   ?= .venv
PY     := $(VENV)/bin/python
PIP    := $(VENV)/bin/pip

# Override training args: `make train ARGS="--epochs 1 --batch-size 16"`
ARGS ?=

# External eval: `make eval-external EVAL_ARGS="--dataset-path ... --checkpoint ..."`
EVAL_ARGS ?=

# Detection: `make detect IMG=path/to/image.jpg`
IMG ?=

.PHONY: help venv pip-upgrade install install-gpu train classify eval-external detect clean clean-artifacts clean-venv

help:
	@echo "Targets:"
	@echo "  make venv            Create $(VENV) if missing"
	@echo "  make pip-upgrade     Upgrade pip inside $(VENV)"
	@echo "  make install         CPU: pip install -r requirements.txt"
	@echo "  make install-gpu     CUDA 12.8 wheels + rest of requirements (see README)"
	@echo "  make train           Run train_entry.py (pass ARGS=\"...\" for CLI flags)"
	@echo "  make classify IMG=... Classify an image (Real/Fake)"
	@echo "  make eval-external   Eval checkpoint (EVAL_ARGS=...; ImageFolder or --labeled-csv)"
	@echo "  make detect IMG=...  Run detect.py on an image"
	@echo "  make clean-artifacts Remove artifacts/, __pycache__, *.py[cod]"
	@echo "  make clean-venv      Remove $(VENV)/"
	@echo "  make clean           clean-artifacts + clean-venv"

venv:
	@test -d "$(VENV)" || $(PYTHON) -m venv "$(VENV)"

pip-upgrade: venv
	"$(PY)" -m pip install --upgrade pip

install: venv pip-upgrade
	"$(PIP)" install -r requirements.txt

# GPU stack per model/README.md — requires NVIDIA driver + CUDA 12.8 toolchain.
install-gpu: venv pip-upgrade
	"$(PIP)" install "torch>=2.9.1" "torchvision>=0.24.0" "torchaudio>=2.9.1" \
		--index-url https://download.pytorch.org/whl/cu128
	"$(PIP)" install -r requirements.txt --no-deps
	"$(PIP)" install "Pillow>=11.0.0" "opencv-python>=4.10.0" "numpy>=2.1.0" "requests>=2.32.0"

train: venv
	"$(PY)" train_entry.py $(ARGS)

detect: venv
	@test -n "$(IMG)" || (echo 'Usage: make detect IMG=path/to/image.jpg' >&2; exit 1)
	"$(PY)" detect.py "$(IMG)"

classify: venv
	@test -n "$(IMG)" || (echo 'Usage: make classify IMG=path/to/image.jpg' >&2; exit 1)
	"$(PY)" classify.py "$(IMG)"

eval-external: venv
	@test -n "$(EVAL_ARGS)" || (echo 'Usage: make eval-external EVAL_ARGS="--labeled-csv data/source_labels.csv --csv-root data --checkpoint artifacts/....pt"' >&2; exit 1)
	"$(PY)" eval_external_dataset.py $(EVAL_ARGS)

clean-artifacts:
	rm -rf artifacts __pycache__ .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ ! -path "./$(VENV)/*" -exec rm -rf {} + 2>/dev/null || true
	find . -type f \( -name '*.pyc' -o -name '*.pyo' \) ! -path "./$(VENV)/*" -delete 2>/dev/null || true

clean-venv:
	rm -rf "$(VENV)"

clean: clean-artifacts clean-venv
