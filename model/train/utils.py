from __future__ import annotations

from typing import Any

import torch


def load_checkpoint(
    path: Any,
    *,
    device: torch.device,
) -> dict:
    """Load a torch checkpoint, handling older PyTorch versions without weights_only."""
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "cuda":
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print_confusion_matrix(
    cm: list[list[int]],
    class_names: list[str],
) -> None:
    header = " " * 12 + "  ".join(f"pred:{n}" for n in class_names)
    print(header)
    for i, row in enumerate(cm):
        print(f"true:{class_names[i]:>6s} " + "  ".join(str(x) for x in row))
