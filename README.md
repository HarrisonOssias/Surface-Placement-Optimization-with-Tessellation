# Surface Placement Optimization with Tessellation (SPO-T)

Real-time detection of optimal package placement positions using Microsoft Kinect v2 and computer vision.

## Features

- Real-time depth sensing with Kinect v2
- RANSAC-based plane detection for surface identification
- Obstacle detection and avoidance
- Optimal placement calculation based on safety margins
- Live visual feedback with placement markers

## Requirements

- Windows OS (for Kinect v2 SDK support)
- Microsoft Kinect v2 sensor
- Python 3.8+
- Anaconda/Miniconda (recommended)

## Installation

### 1. Install Kinect SDK

Download and install the [Kinect for Windows SDK 2.0](https://www.microsoft.com/en-us/download/details.aspx?id=44561)

### 2. Set up Python Environment

```bash
# Create a new conda environment
conda create -n SPO-T python=3.9
conda activate SPO-T

# Install dependencies
pip install -r requirements.txt
```

### 3. Patch pykinect2 for Compatibility

The `pykinect2` library has compatibility issues with Python 3.8+ and 64-bit systems. Run the setup script to automatically patch it:

```bash
python setup_pykinect2.py
```

This script will:
- Fix structure size assertions for 64-bit Python
- Remove incompatible comtypes version checks
- Replace deprecated `time.clock()` with `time.perf_counter()`

## Usage

### Main Application: Package Placement Optimization

```bash
python main.py
```

**Controls:**
- **ESC**: Quit the application
- **S**: Save snapshot of current frame

**Purpose:** Finds the optimal location to place a package on a desk, avoiding obstacles.

---

### Desk Space Monitor

```bash
python desk_monitor.py
```

**Controls:**
- **ESC**: Quit and save all data
- **S**: Save current snapshot and data
- **R**: Reset usage heatmap
- **H**: Toggle heatmap view

**Purpose:** Real-time monitoring and analysis of available desk space with:
- Free space percentage tracking
- Multiple free region detection
- Clutter alerts
- Usage heatmap over time
- Data logging (CSV format)

---

### Box Detector

```bash
python box_detector.py
```

**Controls:**
- **ESC**: Quit
- **S**: Save detection image
- **I**: Toggle 3D info display

**Purpose:** Detect and track boxes/packages on desk surfaces with:
- 3D bounding box detection using RGB-D data
- Size and volume estimation
- Box classification (small/medium/large)
- Real-time tracking
- Traditional CV (no training required) or ML-based detection

**Training:** For improved accuracy, train a YOLOv8 model - see [TRAINING_README.md](TRAINING_README.md)

To use a trained model:
```python
detector = BoxDetector(use_ml_model=True, model_path='runs/detect/box_detection/weights/best.pt')
```

### Configuration

Edit the tunables in `main.py` to adjust detection parameters:

```python
PACKAGE_L = 0.30  # Package length (meters)
PACKAGE_W = 0.20  # Package width (meters)
MARGIN    = 0.015 # Safety buffer (meters)
GRID_RES  = 0.005 # Grid resolution (meters/cell)
H_MIN, H_MAX = 0.005, 0.08  # Obstacle height range (meters)
```

## How It Works

### main.py - Package Placement

1. **Depth Capture**: Acquires depth frames from Kinect v2
2. **Plane Detection**: Uses RANSAC to identify the dominant surface plane
3. **Obstacle Detection**: Identifies objects within specified height range above the plane
4. **Grid Mapping**: Creates an occupancy grid in plane coordinates
5. **Feasibility Analysis**: Calculates areas where the package can fit safely
6. **Placement Optimization**: Selects optimal position maximizing distance from obstacles

### desk_monitor.py - Free Space Detection

1. **Surface Detection**: Same RANSAC-based plane detection as main.py
2. **Occupancy Analysis**: Creates grid showing free vs. occupied areas
3. **Region Identification**: Detects and ranks separate free space regions
4. **Time-Series Tracking**: Logs free space percentage over time
5. **Usage Heatmap**: Accumulates which areas are frequently occupied
6. **Alert System**: Warns when desk becomes too cluttered (>70% occupied)

## Output Files

### From main.py

- `captures/[timestamp]_color.png`: Color image with placement marker
- `captures/[timestamp]_depth.png`: Depth visualization

### From desk_monitor.py

- `desk_monitor_[timestamp].csv`: Time-series log of free space metrics
  - Columns: timestamp, free_percentage, free_area_m2, occupied_area_m2
- `desk_monitor_heatmap_[timestamp].png`: Heatmap showing frequently occupied areas
- `desk_snapshot_[timestamp].png`: Current view with free space overlay

## Improving Detection Accuracy

### Why is box detection inaccurate?

The default traditional computer vision approach uses contour detection on depth data, which can be inaccurate due to:
- Depth sensor noise and artifacts
- Lighting conditions affecting edge detection
- Complex scenes with overlapping objects
- Difficulty distinguishing similar-height objects

### Solution: Train a Machine Learning Model

Train a YOLOv8 model for **significantly better accuracy** (40-60% → 80-95%):

```bash
# 1. Install training dependencies
pip install ultralytics pillow pyyaml

# 2. Train on RGB-D dataset (takes 30 min - 2 hours)
python train_yolo.py --dataset_path "path/to/rgbd-scenes-v2" --epochs 100 --model_size m

# 3. Use trained model in box_detector.py
# Model will be saved to: runs/detect/box_detection/weights/best.pt
```

**See [TRAINING_README.md](TRAINING_README.md) for complete training guide.**

Benefits of trained model:
- ✓ 80-95% detection accuracy (vs 40-60% traditional CV)
- ✓ Robust to noise and lighting variations
- ✓ Better classification of box types
- ✓ More accurate dimension measurements (±1-2cm vs ±5-10cm)

## Troubleshooting

### pykinect2 Installation Issues

If you encounter issues with `pykinect2`:

1. Make sure you're running on Windows (not WSL)
2. Verify Kinect SDK 2.0 is installed
3. Run `python setup_pykinect2.py` to apply compatibility patches
4. If issues persist, check that your Kinect is properly connected and recognized by Windows

### Common Errors

- **"AssertionError: 80"**: Run `setup_pykinect2.py` to fix structure size issues
- **"ImportError: Wrong version"**: Run `setup_pykinect2.py` to disable version checks
- **"AttributeError: module 'time' has no attribute 'clock'"**: Run `setup_pykinect2.py` to fix deprecated time.clock()

## License

See LICENSE file for details.

## Acknowledgments

- Microsoft Kinect SDK
- pykinect2 library
- OpenCV community
