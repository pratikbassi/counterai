"""
Main entry point for image detection.
Can be used as a CLI tool or imported as a module.
"""

import argparse
import json
import sys
from pathlib import Path
from model_handler import ImageDetectionModel


def detect_image(image_path: str, confidence: float = 0.5, output: str = None):
    """
    Run object detection on an image file.
    
    Args:
        image_path: Path to the image file
        confidence: Confidence threshold (0.0-1.0)
        output: Optional path to save JSON results
    """
    # Validate image path
    if not Path(image_path).exists():
        print(f"Error: Image file not found: {image_path}", file=sys.stderr)
        sys.exit(1)
    
    # Initialize model
    print("Loading detection model...")
    detector = ImageDetectionModel()
    
    # Run detection
    print(f"Running detection on: {image_path}")
    detections = detector.detect(image_path, confidence_threshold=confidence)
    
    # Prepare results
    results = {
        'image_path': image_path,
        'num_detections': len(detections),
        'detections': detections
    }
    
    # Print results
    print(f"\nFound {len(detections)} detection(s):")
    for i, det in enumerate(detections, 1):
        print(f"  {i}. {det['label']} (confidence: {det['score']:.2f})")
        print(f"     BBox: {det['bbox']}")
    
    # Save to file if requested
    if output:
        with open(output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output}")
    
    return results


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Run object detection on an image using PyTorch',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python detect.py image.jpg
  python detect.py image.jpg --confidence 0.7
  python detect.py image.jpg --output results.json
        """
    )
    
    parser.add_argument(
        'image',
        type=str,
        help='Path to the image file'
    )
    
    parser.add_argument(
        '--confidence',
        type=float,
        default=0.5,
        help='Confidence threshold (0.0-1.0, default: 0.5)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Optional path to save JSON results'
    )
    
    args = parser.parse_args()
    
    # Validate confidence threshold
    if not 0.0 <= args.confidence <= 1.0:
        print("Error: Confidence threshold must be between 0.0 and 1.0", file=sys.stderr)
        sys.exit(1)
    
    # Run detection
    try:
        detect_image(args.image, args.confidence, args.output)
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()


