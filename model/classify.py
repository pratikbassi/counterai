"""
Classify a single image as Real vs Fake using the trained checkpoint.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from train_model import create_classifier
from train_transforms import create_transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real vs Fake image classifier")
    parser.add_argument("image", type=str, help="Path to an image file")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(Path(__file__).parent / "artifacts" / "best_real_fake.pt"),
        help="Path to a trained checkpoint (.pt). Legacy resnet18 runs also mirror best_real_fake_resnet18.pt.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to run inference on.",
    )
    parser.add_argument(
        "--disable-decompression-bomb-warning",
        action="store_true",
        help="Suppress PIL DecompressionBombWarning during image load.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    if args.disable_decompression_bomb_warning:
        Image.MAX_IMAGE_PIXELS = None

    device: torch.device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    try:
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(ckpt_path, map_location=device)
    class_names = checkpoint.get("class_names", ["Real", "Fake"])
    num_classes = len(class_names)
    image_size = int(checkpoint.get("image_size", 224))
    architecture = checkpoint.get("architecture", "resnet18")
    temperature = float(checkpoint.get("temperature", 1.0))

    model, _ = create_classifier(
        architecture, num_classes=num_classes, device=device
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    _train_tfms, eval_tfms = create_transforms(image_size)

    img = Image.open(image_path).convert("RGB")
    x = eval_tfms(img).unsqueeze(0).to(device)

    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            logits = model(x) / temperature
            probs = torch.softmax(logits, dim=1)[0]

    pred_idx = int(probs.argmax().item())
    pred_label = class_names[pred_idx] if pred_idx < len(class_names) else str(pred_idx)

    print(f"Image: {image_path}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Architecture: {architecture} | temperature T={temperature:.4f}")
    print(f"Predicted: {pred_label} (p={float(probs[pred_idx]):.4f})")
    print("Probabilities:")
    for i, name in enumerate(class_names):
        print(f"  {name}: {float(probs[i]):.4f}")


if __name__ == "__main__":
    main()

