"""
Model handler for image detection using PyTorch.
Supports pre-trained object detection models from torchvision.
"""

import torch
import torchvision.transforms as transforms
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights
)
from PIL import Image
import numpy as np
from typing import List, Dict, Tuple, Union
import io


class ImageDetectionModel:
    """
    A wrapper class for PyTorch object detection models.
    Uses Faster R-CNN with ResNet-50 backbone by default.
    """
    
    def __init__(self, model_name: str = "fasterrcnn_resnet50_fpn", device: str = None):
        """
        Initialize the detection model.
        
        Args:
            model_name: Name of the model to use (default: fasterrcnn_resnet50_fpn)
            device: Device to run inference on ('cuda', 'cpu', or None for auto-detection)
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        self.model = None
        self.transform = None
        self._load_model()
    
    def _load_model(self):
        """Load the pre-trained detection model."""
        if self.model_name == "fasterrcnn_resnet50_fpn":
            weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
            self.model = fasterrcnn_resnet50_fpn(weights=weights)
            self.transform = weights.transforms()
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
        
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded on device: {self.device}")
    
    def preprocess_image(self, image: Union[str, Image.Image, np.ndarray, bytes]) -> torch.Tensor:
        """
        Preprocess an image for model input.
        
        Args:
            image: Image input as file path, PIL Image, numpy array, or bytes
            
        Returns:
            Preprocessed image tensor
        """
        # Convert various input types to PIL Image
        if isinstance(image, str):
            # File path
            img = Image.open(image).convert('RGB')
        elif isinstance(image, bytes):
            # Bytes data
            img = Image.open(io.BytesIO(image)).convert('RGB')
        elif isinstance(image, np.ndarray):
            # Numpy array
            img = Image.fromarray(image).convert('RGB')
        elif isinstance(image, Image.Image):
            # Already a PIL Image
            img = image.convert('RGB')
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        # Apply model-specific transforms
        if self.transform:
            img_tensor = self.transform(img)
        else:
            # Fallback transform
            transform = transforms.Compose([
                transforms.ToTensor()
            ])
            img_tensor = transform(img)
        
        return img_tensor
    
    def detect(self, image: Union[str, Image.Image, np.ndarray, bytes], 
               confidence_threshold: float = 0.5) -> List[Dict]:
        """
        Run object detection on an image.
        
        Args:
            image: Image input as file path, PIL Image, numpy array, or bytes
            confidence_threshold: Minimum confidence score for detections (0.0-1.0)
            
        Returns:
            List of detection dictionaries, each containing:
            - 'bbox': [x1, y1, x2, y2] bounding box coordinates
            - 'score': Confidence score
            - 'label': Class label name
            - 'label_id': Class label ID
        """
        # Preprocess image
        img_tensor = self.preprocess_image(image)
        
        # Add batch dimension and move to device
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        # Run inference
        with torch.no_grad():
            predictions = self.model(img_tensor)
        
        # Process predictions
        detections = []
        pred = predictions[0]  # Get first (and only) batch item
        
        # Get class names
        if hasattr(self.model, 'get_class_names'):
            class_names = self.model.get_class_names()
        else:
            # COCO class names (default for Faster R-CNN)
            class_names = self._get_coco_class_names()
        
        boxes = pred['boxes'].cpu().numpy()
        scores = pred['scores'].cpu().numpy()
        labels = pred['labels'].cpu().numpy()
        
        for box, score, label in zip(boxes, scores, labels):
            if score >= confidence_threshold:
                detections.append({
                    'bbox': box.tolist(),
                    'score': float(score),
                    'label': class_names[label],
                    'label_id': int(label)
                })
        
        return detections
    
    def _get_coco_class_names(self) -> List[str]:
        """Get COCO dataset class names."""
        return [
            '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
            'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
            'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
            'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]


