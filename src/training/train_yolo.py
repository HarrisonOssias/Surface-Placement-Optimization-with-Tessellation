"""
YOLOv8 Training Script for Box Detection
=========================================

Trains a YOLOv8 model on the UW RGB-D Object Dataset for detecting boxes,
packages, and containers on desk surfaces.

This script will:
1. Convert RGB-D dataset to YOLO format
2. Generate training/validation splits
3. Train YOLOv8 model
4. Export trained model for use with box_detector.py

Dataset: https://rgbd-dataset.cs.washington.edu/index.html

Usage:
    # Install dependencies first
    pip install ultralytics pillow pyyaml

    # Then run training
    python train_yolo.py --dataset_path "path/to/rgbd-scenes" --epochs 100
"""

import os
import sys
import argparse
import shutil
from pathlib import Path
import cv2
import numpy as np
import yaml
from typing import List, Tuple, Dict
import json

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Box/package categories from the RGB-D dataset
# These are objects commonly found on desks that we want to detect
BOX_CATEGORIES = [
    'box', 'food_box', 'kleenex', 'cereal_box', 'food_can', 'food_bag',
    'soda_can', 'wine_bottle', 'cap', 'flashlight', 'ball', 'garlic',
    'mushroom', 'orange', 'apple', 'lemon', 'peach', 'pear', 'plum', 
    'lightbulb', 'plate', 'bowl', 'cup', 'sponge', 'marker', 'scissors'
]

# Image augmentation settings for training
AUGMENTATION_CONFIG = {
    'hsv_h': 0.015,  # HSV hue augmentation
    'hsv_s': 0.7,    # HSV saturation
    'hsv_v': 0.4,    # HSV value
    'degrees': 10.0,  # Rotation degrees
    'translate': 0.1, # Translation
    'scale': 0.5,     # Scale augmentation
    'shear': 0.0,     # Shear
    'perspective': 0.0, # Perspective
    'flipud': 0.1,    # Flip upside-down probability
    'fliplr': 0.5,    # Flip left-right probability
    'mosaic': 1.0,    # Mosaic augmentation
    'mixup': 0.1,     # MixUp augmentation
}

# ==============================================================================
# DATASET PREPARATION
# ==============================================================================

class RGBDDatasetConverter:
    """Converts UW RGB-D Object Dataset to YOLO format."""
    
    def __init__(self, dataset_path: str, output_path: str):
        """
        Initialize converter.
        
        Args:
            dataset_path: Path to rgbd-scenes-v2 directory
            output_path: Where to save YOLO-formatted dataset
        """
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_path)
        self.categories = BOX_CATEGORIES
        self.category_to_id = {cat: idx for idx, cat in enumerate(self.categories)}
        
        # Create output directories
        self.train_images = self.output_path / 'images' / 'train'
        self.train_labels = self.output_path / 'labels' / 'train'
        self.val_images = self.output_path / 'images' / 'val'
        self.val_labels = self.output_path / 'labels' / 'val'
        
        for dir_path in [self.train_images, self.train_labels, 
                         self.val_images, self.val_labels]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def find_objects_in_scene(self, scene_dir: Path) -> List[Dict]:
        """
        Parse scene directory to find objects and their locations.
        
        For the RGB-D dataset, we'll use a simple approach:
        - Use depth discontinuities to segment objects
        - Classify based on size, shape, and appearance
        
        Args:
            scene_dir: Path to scene directory (e.g., scene_01)
        
        Returns:
            List of detected objects with bounding boxes
        """
        objects = []
        
        # Get all color images in scene
        color_files = sorted(scene_dir.glob('*-color.png'))
        
        for color_file in color_files:
            # Load corresponding depth image
            depth_file = color_file.parent / color_file.name.replace('-color.png', '-depth.png')
            if not depth_file.exists():
                continue
            
            color = cv2.imread(str(color_file))
            depth = cv2.imread(str(depth_file), cv2.IMREAD_UNCHANGED)
            
            if color is None or depth is None:
                continue
            
            # Simple object detection using depth segmentation
            detected = self._detect_objects_simple(color, depth)
            
            if detected:
                objects.append({
                    'image_path': color_file,
                    'depth_path': depth_file,
                    'objects': detected
                })
        
        return objects
    
    def _detect_objects_simple(self, color: np.ndarray, depth: np.ndarray) -> List[Dict]:
        """
        Simple object detection using depth and color.
        
        This is a basic approach for auto-labeling. For better results,
        you should manually annotate or use the dataset's provided labels.
        
        Args:
            color: RGB image
            depth: Depth map
        
        Returns:
            List of detected objects with bounding boxes
        """
        if depth.dtype == np.uint16:
            depth = depth.astype(np.float32) / 1000.0  # Convert to meters
        
        # Normalize depth for processing
        valid_depth = depth > 0
        if not valid_depth.any():
            return []
        
        # Simple foreground extraction: objects closer than median depth
        median_depth = np.median(depth[valid_depth])
        foreground = (depth > 0) & (depth < median_depth - 0.1)
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        foreground = cv2.morphologyEx(foreground.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detected = []
        h, w = color.shape[:2]
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area
            if area < 500 or area > 50000:
                continue
            
            # Get bounding box
            x, y, bw, bh = cv2.boundingRect(contour)
            
            # Filter by aspect ratio
            aspect = bw / bh if bh > 0 else 0
            if aspect < 0.3 or aspect > 3.0:
                continue
            
            # Normalize coordinates to YOLO format [0-1]
            x_center = (x + bw / 2) / w
            y_center = (y + bh / 2) / h
            width = bw / w
            height = bh / h
            
            # Classify as generic 'box' for now
            # In practice, you'd use more sophisticated classification
            detected.append({
                'class_id': 0,  # 'box' is first in our category list
                'bbox': [x_center, y_center, width, height]
            })
        
        return detected
    
    def convert_to_yolo(self, train_split: float = 0.8, use_auto_label: bool = True):
        """
        Convert RGB-D dataset to YOLO format.
        
        Args:
            train_split: Fraction of data to use for training (vs validation)
            use_auto_label: If True, auto-generate labels. If False, skip scenes without labels.
        """
        print("=" * 70)
        print("Converting RGB-D Dataset to YOLO Format")
        print("=" * 70)
        
        # Find all scene directories
        scenes_dir = self.dataset_path / 'imgs'
        if not scenes_dir.exists():
            print(f"Error: Could not find {scenes_dir}")
            print("Please ensure dataset path points to rgbd-scenes-v2 directory")
            return False
        
        scene_dirs = sorted([d for d in scenes_dir.iterdir() if d.is_dir()])
        print(f"Found {len(scene_dirs)} scenes")
        
        if not scene_dirs:
            print("Error: No scene directories found")
            return False
        
        # Process each scene
        all_samples = []
        for scene_dir in scene_dirs:
            print(f"\nProcessing {scene_dir.name}...")
            samples = self.find_objects_in_scene(scene_dir)
            all_samples.extend(samples)
            print(f"  Found {len(samples)} labeled frames")
        
        if not all_samples:
            print("\nError: No samples generated. Check dataset structure.")
            return False
        
        print(f"\nTotal samples: {len(all_samples)}")
        
        # Split into train/val
        np.random.shuffle(all_samples)
        split_idx = int(len(all_samples) * train_split)
        train_samples = all_samples[:split_idx]
        val_samples = all_samples[split_idx:]
        
        print(f"Train samples: {len(train_samples)}")
        print(f"Val samples: {len(val_samples)}")
        
        # Save samples
        self._save_samples(train_samples, self.train_images, self.train_labels, 'train')
        self._save_samples(val_samples, self.val_images, self.val_labels, 'val')
        
        # Create dataset YAML
        self._create_dataset_yaml()
        
        print("\n" + "=" * 70)
        print("✓ Dataset conversion complete!")
        print(f"✓ Output saved to: {self.output_path}")
        print("=" * 70)
        
        return True
    
    def _save_samples(self, samples: List[Dict], image_dir: Path, 
                      label_dir: Path, split: str):
        """Save samples to YOLO format."""
        for idx, sample in enumerate(samples):
            # Copy image
            src_image = sample['image_path']
            dst_image = image_dir / f"{split}_{idx:06d}.jpg"
            shutil.copy(src_image, dst_image)
            
            # Save label
            label_file = label_dir / f"{split}_{idx:06d}.txt"
            with open(label_file, 'w') as f:
                for obj in sample['objects']:
                    class_id = obj['class_id']
                    bbox = obj['bbox']
                    f.write(f"{class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")
    
    def _create_dataset_yaml(self):
        """Create YOLO dataset configuration file."""
        yaml_content = {
            'path': str(self.output_path.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'nc': len(self.categories),
            'names': self.categories
        }
        
        yaml_file = self.output_path / 'dataset.yaml'
        with open(yaml_file, 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False)
        
        print(f"\n✓ Created dataset config: {yaml_file}")

# ==============================================================================
# TRAINING
# ==============================================================================

def train_yolo_model(dataset_path: str, epochs: int = 100, batch_size: int = 16,
                     image_size: int = 640, model_size: str = 'n'):
    """
    Train YOLOv8 model on the prepared dataset.
    
    Args:
        dataset_path: Path to YOLO-formatted dataset directory
        epochs: Number of training epochs
        batch_size: Batch size for training
        image_size: Input image size (640 recommended)
        model_size: YOLOv8 model size ('n', 's', 'm', 'l', 'x')
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: ultralytics package not installed")
        print("Install with: pip install ultralytics")
        return None
    
    print("\n" + "=" * 70)
    print("Starting YOLOv8 Training")
    print("=" * 70)
    
    # Load base model
    model_name = f'yolov8{model_size}.pt'
    print(f"\nLoading base model: {model_name}")
    model = YOLO(model_name)
    
    # Train
    dataset_yaml = Path(dataset_path) / 'dataset.yaml'
    print(f"Dataset config: {dataset_yaml}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Image size: {image_size}")
    print("\nTraining started...\n")
    
    results = model.train(
        data=str(dataset_yaml),
        epochs=epochs,
        imgsz=image_size,
        batch=batch_size,
        name='box_detection',
        patience=20,  # Early stopping patience
        save=True,
        device=0 if check_cuda() else 'cpu',
        **AUGMENTATION_CONFIG
    )
    
    print("\n" + "=" * 70)
    print("✓ Training complete!")
    print("=" * 70)
    
    # Get best model path
    best_model = Path('runs/detect/box_detection/weights/best.pt')
    print(f"\n✓ Best model saved to: {best_model}")
    print(f"\nTo use this model with box_detector.py:")
    print(f"  detector = BoxDetector(use_ml_model=True, model_path='{best_model}')")
    
    return best_model

def check_cuda():
    """Check if CUDA is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train YOLOv8 for box detection')
    
    # Dataset arguments
    parser.add_argument('--dataset_path', type=str, 
                        default='rgbd-scenes-v2',
                        help='Path to RGB-D dataset (rgbd-scenes-v2 directory)')
    parser.add_argument('--output_path', type=str,
                        default='yolo_dataset',
                        help='Where to save YOLO-formatted dataset')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--image_size', type=int, default=640,
                        help='Input image size')
    parser.add_argument('--model_size', type=str, default='n',
                        choices=['n', 's', 'm', 'l', 'x'],
                        help='YOLOv8 model size (n=nano, s=small, m=medium, l=large, x=xlarge)')
    
    # Options
    parser.add_argument('--skip_conversion', action='store_true',
                        help='Skip dataset conversion (use existing YOLO dataset)')
    parser.add_argument('--auto_label', action='store_true', default=True,
                        help='Auto-generate labels from depth data')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("YOLOv8 Box Detection Training Pipeline")
    print("=" * 70)
    print(f"Dataset: {args.dataset_path}")
    print(f"Output: {args.output_path}")
    print(f"Model size: YOLOv8{args.model_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"CUDA available: {check_cuda()}")
    print("=" * 70)
    
    # Step 1: Convert dataset
    if not args.skip_conversion:
        converter = RGBDDatasetConverter(args.dataset_path, args.output_path)
        success = converter.convert_to_yolo(use_auto_label=args.auto_label)
        if not success:
            print("\nDataset conversion failed. Exiting.")
            return 1
    else:
        print("\nSkipping dataset conversion (using existing dataset)")
    
    # Step 2: Train model
    best_model = train_yolo_model(
        dataset_path=args.output_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=args.image_size,
        model_size=args.model_size
    )
    
    if best_model and best_model.exists():
        print("\n" + "=" * 70)
        print("✓ Training pipeline completed successfully!")
        print("=" * 70)
        return 0
    else:
        print("\nTraining failed. Check errors above.")
        return 1

if __name__ == '__main__':
    sys.exit(main())

