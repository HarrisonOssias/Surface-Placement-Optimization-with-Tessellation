# Your RGB-D Dataset

## ✓ Dataset Successfully Located!

You have the **UW RGB-D Scenes Dataset v2** with excellent coverage:

- **14 scenes** with real indoor environments
- **11,427 total frames** (888-1,128 frames per scene)
- **RGB + Depth images** (640x480 aligned)
- **Point clouds** (.ply format) for all scenes
- **Object labels** available

## Locations

- **Images**: `C:\Users\Sarri\Downloads\rgbd-scenes-v2_imgs\rgbd-scenes-v2\imgs`
- **Point Clouds**: `C:\Users\Sarri\Downloads\rgbd-scenes-v2_pc\rgbd-scenes-v2\pc`

## Quick Commands

### Check what you have
```bash
python check_dataset.py
```

### Browse interactively
```bash
python explore_dataset.py
# Choose option 1: Browse dataset interactively
# Use arrow keys: ←→ for frames, ↑↓ for scenes
```

### Test box detection (no training needed)
```bash
python box_detector.py
```

## Scenes Overview

| Scene | Frames | Description |
|-------|--------|-------------|
| scene_01 | 888 | Office/desk environment |
| scene_02 | 834 | Meeting room setup |
| scene_03 | 861 | Kitchen/dining area |
| scene_04 | 868 | Workspace with objects |
| scene_05 | 1,128 | Large desk scene |
| scene_06 | 1,048 | Office with furniture |
| scene_07 | 943 | Tabletop scene |
| scene_08 | 925 | Desk with items |
| scene_09 | 732 | Kitchen counter |
| scene_10 | 716 | Office desk |
| scene_11 | 640 | Meeting table |
| scene_12 | 723 | Workspace |
| scene_13 | 462 | Small desk |
| scene_14 | 659 | Office setup |

## What's in Each Frame

Each frame consists of:
1. **Color image** (`XXXXX-color.png`): RGB image at 640x480
2. **Depth image** (`XXXXX-depth.png`): 16-bit depth map, aligned to color
3. **Point cloud** (`.ply` file): Full 3D reconstruction of scene

## Training Workflow

### Option 1: Use Pre-built Detector (No Training)

```bash
python box_detector.py
```

This uses traditional computer vision - works immediately!

### Option 2: Train Custom Model

1. **Explore dataset**:
   ```bash
   python explore_dataset.py
   ```

2. **Extract training samples**:
   - Choose option 3 in explorer
   - Creates `data/processed_rgbd/` with samples

3. **Train YOLOv8**:
   ```python
   from ultralytics import YOLO
   model = YOLO('yolov8n.pt')
   # ... training code (see TRAINING_GUIDE.md)
   ```

4. **Use trained model**:
   ```python
   from box_detector import BoxDetector
   detector = BoxDetector(use_ml_model=True, model_path='path/to/weights.pt')
   ```

## Integration with Your Applications

### With desk_monitor.py

Add box detection to see what objects are on desk:

```python
from box_detector import BoxDetector

# In main():
box_detector = BoxDetector()

# In loop:
boxes = box_detector.detect(color_bgr, d, k._mapper)
print(f"Boxes on desk: {len(boxes)}")
```

### With main.py

Avoid placing packages where boxes already exist:

```python
from box_detector import BoxDetector

box_detector = BoxDetector()

# Detect existing boxes first
boxes = box_detector.detect(color_bgr, d, k._mapper)

# Mark box locations as occupied in grid
# ... (integrate with existing occupancy grid)
```

## File Formats

### Color Images
- Format: PNG
- Size: 640x480 pixels
- Channels: RGB (3 channels)

### Depth Images
- Format: PNG (16-bit)
- Size: 640x480 pixels
- Units: Millimeters
- Range: ~500-4500mm

### Point Clouds
- Format: PLY (Stanford format)
- Contains: XYZ coordinates + RGB colors
- Use: Open in MeshLab or CloudCompare

### Labels
- Format: Text file
- Contains: Object bounding boxes and classes
- Use: Ground truth for training/evaluation

## Tips

1. **Performance**: Process every Nth frame for faster training data prep
2. **Quality**: Focus on scenes 1, 5, 6, 7 for desk-like environments
3. **Augmentation**: Use different frames from same scene for variety
4. **Testing**: Hold out 2-3 scenes for validation/testing

## Resources

- **Dataset Paper**: [UW RGB-D Object Dataset (ICRA 2011)](https://rgbd-dataset.cs.washington.edu/)
- **Your Training Guide**: `TRAINING_GUIDE.md`
- **Feature Comparison**: `FEATURES.md`
- **Quick Start**: `QUICKSTART.md`

## Citation

If you use this dataset in research:

```bibtex
@inproceedings{lai2014unsupervised,
  title={Unsupervised feature learning for 3d scene labeling},
  author={Lai, Kevin and Bo, Liefeng and Fox, Dieter},
  booktitle={2014 IEEE International Conference on Robotics and Automation (ICRA)},
  pages={3050--3057},
  year={2014},
  organization={IEEE}
}
```

## Next Steps

1. ✅ **Dataset verified** - You have 14 scenes, 11,427 frames
2. ⏭️ **Explore**: Run `python explore_dataset.py`
3. ⏭️ **Test detector**: Run `python box_detector.py`
4. ⏭️ **Train model** (optional): See `TRAINING_GUIDE.md`
5. ⏭️ **Integrate**: Add to `desk_monitor.py` or `main.py`

---

**You're all set!** 🎉 The dataset is ready to use.

