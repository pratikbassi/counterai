# CounterAI downloader — URL → image fetch (placeholder). POSIX: Linux, macOS, WSL.

.DEFAULT_GOAL := help

PYTHON ?= python3
VENV   ?= .venv
PY     := $(VENV)/bin/python
PIP    := $(VENV)/bin/pip

# Override: `make run URL="https://example.com/a.jpg"`
URL ?=

.PHONY: help venv pip-upgrade install run clean clean-venv

help:
	@echo "Targets:"
	@echo "  make venv         Create $(VENV) if missing"
	@echo "  make pip-upgrade  Upgrade pip inside $(VENV)"
	@echo "  make install      pip install -r requirements.txt"
	@echo "  make run URL=...  Run download_url.py (pass ARGS=\"...\" for extra flags)"
	@echo "  make clean        Remove __pycache__, caches"
	@echo "  make clean-venv   Remove $(VENV)/"

venv:
	@test -d "$(VENV)" || $(PYTHON) -m venv "$(VENV)"

pip-upgrade: venv
	"$(PY)" -m pip install --upgrade pip

install: venv pip-upgrade
	"$(PIP)" install -r requirements.txt

ARGS ?=

run: venv
	@test -n "$(URL)" || (echo 'Usage: make run URL=https://example.com/image.jpg' >&2; exit 1)
	"$(PY)" download_url.py "$(URL)" $(ARGS)

clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache downloads
	find . -type d -name __pycache__ ! -path "./$(VENV)/*" -exec rm -rf {} + 2>/dev/null || true
	find . -type f \( -name '*.pyc' -o -name '*.pyo' \) ! -path "./$(VENV)/*" -delete 2>/dev/null || true

clean-venv:
	rm -rf "$(VENV)"
