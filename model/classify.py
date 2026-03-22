"""
Classify a single image as Real vs Fake using the trained checkpoint.

CLI usage:
    python classify.py image.jpg
    python classify.py image.jpg --json

Programmatic usage:
    from classify import classify_image
    result = classify_image("/path/to/image.jpg")
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image

from train.model import create_classifier
from train.transforms import create_transforms
from train.utils import load_checkpoint, resolve_device

_DEFAULT_CHECKPOINT = Path(__file__).parent / "artifacts" / "best_real_fake.pt"


def classify_image(
    image_path: str | Path,
    *,
    checkpoint: str | Path = _DEFAULT_CHECKPOINT,
    device: str = "auto",
) -> dict:
    """
    Classify a single image as Real vs Fake.

    Returns a dict with:
        label:         predicted class name (e.g. "Real" or "Fake")
        confidence:    probability of the predicted class
        probabilities: {class_name: probability, ...}
        architecture:  model architecture used
        temperature:   softmax temperature applied
    """

    image_path = Path(image_path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    ckpt_path = Path(checkpoint)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    dev = resolve_device(device)
    ckpt = load_checkpoint(ckpt_path, device=dev)
    class_names = ckpt.get("class_names", ["Real", "Fake"])
    num_classes = len(class_names)
    image_size = int(ckpt.get("image_size", 224))
    architecture = str(ckpt.get("architecture", "resnet18"))
    temperature = float(ckpt.get("temperature", 1.0))

    model, _ = create_classifier(architecture, num_classes=num_classes, device=dev)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    _, eval_tfms = create_transforms(image_size)

    img = Image.open(image_path).convert("RGB")
    x = eval_tfms(img).unsqueeze(0).to(dev)

    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=(dev.type == "cuda")):
            logits = model(x) / temperature
            probs = torch.softmax(logits, dim=1)[0]

    pred_idx = int(probs.argmax().item())
    pred_label = class_names[pred_idx] if pred_idx < len(class_names) else str(pred_idx)

    return {
        "label": pred_label,
        "confidence": float(probs[pred_idx]),
        "probabilities": {name: float(probs[i]) for i, name in enumerate(class_names)},
        "architecture": architecture,
        "temperature": temperature,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real vs Fake image classifier")
    parser.add_argument("image", type=str, help="Path to an image file")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(_DEFAULT_CHECKPOINT),
        help="Path to a trained checkpoint (.pt).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to run inference on.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output result as a single JSON object (for programmatic callers).",
    )
    parser.add_argument(
        "--disable-decompression-bomb-warning",
        action="store_true",
        help="Suppress PIL DecompressionBombWarning during image load.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.disable_decompression_bomb_warning:
        Image.MAX_IMAGE_PIXELS = None

    try:
        result = classify_image(
            args.image,
            checkpoint=args.checkpoint,
            device=args.device,
        )
    except (FileNotFoundError, RuntimeError) as exc:
        if args.json_output:
            json.dump({"error": str(exc)}, sys.stdout)
            sys.stdout.write("\n")
            sys.exit(1)
        raise

    if args.json_output:
        json.dump(result, sys.stdout)
        sys.stdout.write("\n")
        return

    print(f"Image: {args.image}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Architecture: {result['architecture']} | temperature T={result['temperature']:.4f}")
    print(f"Predicted: {result['label']} (p={result['confidence']:.4f})")
    print("Probabilities:")
    for name, prob in result["probabilities"].items():
        print(f"  {name}: {prob:.4f}")


if __name__ == "__main__":
    main()
