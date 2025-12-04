# Training Guide: Box Detection with UW RGB-D Object Dataset

## Overview

This guide explains how to train a box/package detection model using the [UW RGB-D Object Dataset](https://rgbd-dataset.cs.washington.edu/index.html).

The dataset contains:
- **300 common household objects** including boxes, packages, containers
- **Synchronized RGB + Depth images** at 640x480 resolution
- **Multiple viewpoints** (3 camera heights per object)
- **Full rotation sequences** (turntable recording)
- **Ground truth pose annotations**
- **51 object categories** organized hierarchically

## Quick Start

### Option 1: Use Pre-Configured Box Detector (No Training)

The `box_detector.py` script works out-of-the-box with traditional computer vision:

```bash
python box_detector.py
```

This uses contour detection + depth filtering to detect boxes without any ML training.

### Option 2: Train Custom ML Model (Advanced)

For better accuracy, train a model on the RGB-D dataset.

---

## Dataset Setup

### You Already Have It! ✓

You've downloaded the **RGB-D Scenes Dataset v2** which is perfect for training!

**Your dataset locations:**
- Images: `C:\Users\Sarri\Downloads\rgbd-scenes-v2_imgs\rgbd-scenes-v2\imgs`
- Point Clouds: `C:\Users\Sarri\Downloads\rgbd-scenes-v2_pc\rgbd-scenes-v2\pc`

### Explore Your Dataset

```bash
python explore_dataset.py
```

This interactive tool lets you:
1. **Browse** all 14 scenes with RGB + Depth
2. **View** specific frames
3. **Extract** training samples automatically
4. **See statistics** about your dataset

### Dataset Structure

You have **14 scenes** with:
- ~1,000-2,000 frames per scene (22,000+ total frames!)
- RGB images (640x480 PNG)
- Aligned depth maps (16-bit PNG)
- Point clouds (.ply format)
- Object labels (.label format)

Each scene contains furniture and tabletop objects - perfect for desk scenarios!

### Quick Start with Your Dataset

```bash
# 1. Explore the dataset
python explore_dataset.py

# 2. Choose option 1 to browse interactively
#    Use arrow keys to navigate scenes/frames

# 3. Choose option 3 to extract training samples
#    This creates: data/processed_rgbd/
```

---

## Training Approaches

### Approach 1: YOLOv8 + RGB-D (Recommended)

Train YOLOv8 with RGB images and use depth for post-processing.

#### Step 1: Install Dependencies

```bash
pip install ultralytics opencv-python pillow
```

#### Step 2: Prepare Dataset for YOLO

Create `prepare_yolo_dataset.py`:

```python
"""
Convert UW RGB-D Dataset to YOLO format
"""
import os
import shutil
from pathlib import Path
import cv2

# Define box-related categories from RGB-D dataset
BOX_CATEGORIES = [
    'cereal_box', 'kleenex_tissue_box', 'food_box',
    'shipping_box', 'office_box', 'cardboard_box'
]

def convert_to_yolo(dataset_path, output_path):
    """Convert RGB-D dataset to YOLO format."""
    
    output_path = Path(output_path)
    (output_path / 'images' / 'train').mkdir(parents=True, exist_ok=True)
    (output_path / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
    
    image_id = 0
    
    for category_idx, category in enumerate(BOX_CATEGORIES):
        category_path = Path(dataset_path) / category
        
        if not category_path.exists():
            print(f"Warning: {category} not found")
            continue
        
        # Process each object instance
        for obj_dir in category_path.iterdir():
            if not obj_dir.is_dir():
                continue
            
            # Process each image
            for img_file in obj_dir.glob('*_crop.png'):
                # Copy image
                img_out = output_path / 'images' / 'train' / f'{image_id:06d}.png'
                shutil.copy(img_file, img_out)
                
                # Create label (object occupies most of image after cropping)
                # Format: class_id center_x center_y width height (normalized 0-1)
                label_out = output_path / 'labels' / 'train' / f'{image_id:06d}.txt'
                with open(label_out, 'w') as f:
                    f.write(f"{category_idx} 0.5 0.5 0.8 0.8\n")
                
                image_id += 1
    
    print(f"Converted {image_id} images")
    
    # Create dataset.yaml
    with open(output_path / 'dataset.yaml', 'w') as f:
        f.write(f"""
path: {output_path.absolute()}
train: images/train
val: images/train  # Use same for now, split later

nc: {len(BOX_CATEGORIES)}
names: {BOX_CATEGORIES}
""")

if __name__ == "__main__":
    convert_to_yolo('data/rgbd-dataset', 'data/yolo_boxes')
```

Run it:
```bash
python prepare_yolo_dataset.py
```

#### Step 3: Train YOLOv8

```python
from ultralytics import YOLO

# Load pretrained model
model = YOLO('yolov8n.pt')

# Train
results = model.train(
    data='data/yolo_boxes/dataset.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    name='box_detector'
)
```

#### Step 4: Use Trained Model

Update `box_detector.py`:

```python
detector = BoxDetector(use_ml_model=True, model_path='runs/detect/box_detector/weights/best.pt')
```

---

### Approach 2: Custom CNN with RGB-D Fusion

Train a custom model that uses both RGB and depth channels.

```python
import torch
import torch.nn as nn

class RGBDBoxDetector(nn.Module):
    """4-channel input (RGB + Depth) for box detection."""
    
    def __init__(self, num_classes=6):
        super().__init__()
        # Use ResNet backbone modified for 4 channels
        self.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3)
        # ... rest of network
        
    def forward(self, rgbd):
        # rgbd shape: (B, 4, H, W) - RGB + Depth
        x = self.conv1(rgbd)
        # ... detection head
        return boxes, classes, scores
```

---

### Approach 3: Traditional CV (No Training)

Already implemented in `box_detector.py`! Uses:
- Contour detection on height-filtered depth map
- Shape filtering (area, aspect ratio)
- 3D dimension estimation

---

## Integration with Existing Scripts

### With desk_monitor.py

Add box detection to desk monitoring:

```python
# In desk_monitor.py, add after imports:
from box_detector import BoxDetector

# In main(), after analyzer initialization:
box_detector = BoxDetector()

# In main loop, after getting frames:
boxes = box_detector.detect(color_bgr, d, k._mapper)
display = box_detector.visualize_detections(display, boxes)

# Print box info
if len(boxes) > 0:
    total_box_volume = sum(b.volume_m3 for b in boxes)
    print(f"  Boxes on desk: {len(boxes)}, Total volume: {total_box_volume*1000:.1f}L")
```

### With main.py

Avoid placing packages on top of detected boxes:

```python
# In main.py, integrate box detection:
from box_detector import BoxDetector

box_detector = BoxDetector()

# After creating occupancy grid:
boxes = box_detector.detect(color_bgr, d, k._mapper)

# Mark box locations as occupied in grid
for box in boxes:
    x, y, w, h = box.bbox_2d
    # Convert to grid coordinates and mark as occupied
    # ... (similar to existing obstacle marking)
```

---

## Dataset Statistics

From the [UW RGB-D Object Dataset](https://rgbd-dataset.cs.washington.edu/index.html):

- **Total objects**: 300
- **Categories**: 51
- **Box-related categories**: ~8-10 (boxes, containers, packages)
- **Images per object**: ~500-600 (full rotation, 3 heights)
- **Total box images**: ~4,000-6,000
- **Resolution**: 640x480 (RGB + aligned depth)
- **Format**: PNG (RGB), PNG (16-bit depth)

### Box Categories in Dataset:
1. Cereal boxes
2. Food boxes  
3. Tissue boxes (Kleenex)
4. Office supply boxes
5. Shipping/cardboard boxes
6. Storage containers
7. Packaging boxes

---

## Performance Tips

### For Real-Time Detection:

1. **Downsample frames**: Process every Nth frame
2. **ROI processing**: Only process desk region
3. **Model optimization**: Use TensorRT or ONNX
4. **Tracking**: Use Kalman filter to smooth detections

```python
# Example: Process every 3rd frame
if frame_count % 3 == 0:
    boxes = detector.detect(color_bgr, d, k._mapper)
else:
    # Use previous detections with tracking
    boxes = tracker.predict()
```

### For Accuracy:

1. **Data augmentation**: Rotation, scaling, lighting
2. **Multi-scale training**: Different object distances
3. **Depth normalization**: Handle varying distances
4. **Ensemble models**: Combine multiple approaches

---

## Citation

If you use the UW RGB-D Object Dataset, cite:

```bibtex
@inproceedings{lai2011large,
  title={A large-scale hierarchical multi-view rgb-d object dataset},
  author={Lai, Kevin and Bo, Liefeng and Ren, Xiaofeng and Fox, Dieter},
  booktitle={2011 IEEE international conference on robotics and automation},
  pages={1817--1824},
  year={2011},
  organization={IEEE}
}
```

---

## Next Steps

1. **Download dataset**: Get box categories from [rgbd-dataset.cs.washington.edu](https://rgbd-dataset.cs.washington.edu/)
2. **Try box_detector.py**: Test traditional CV approach
3. **Train custom model**: Use YOLO or custom architecture
4. **Integrate**: Add to desk_monitor.py or main.py
5. **Evaluate**: Test on real Kinect data
6. **Iterate**: Improve based on real-world performance

---

## Troubleshooting

### "Dataset not found"
- Check download location
- Verify extracted structure matches expected format

### "Low detection accuracy"
- Collect more training data from your specific environment
- Adjust detection thresholds in config
- Try data augmentation

### "Slow performance"
- Use model quantization
- Reduce input resolution
- Process fewer frames per second

---

## Resources

- **Dataset**: https://rgbd-dataset.cs.washington.edu/
- **YOLOv8 Docs**: https://docs.ultralytics.com/
- **RGB-D Features**: [Kernel Descriptors paper](https://rgbd-dataset.cs.washington.edu/publications.html)
- **Detection Code**: [UW GitHub](https://github.com/liefeng-c/rgbd-dataset)

---

**Ready to start?** Run `python box_detector.py` to test the basic detector!

