# SPO-T: Surface Placement Optimization with Tessellation

**Adaptive Package Placement System for Indoor Delivery Robots**

Real-time detection and feasibility analysis for placing packages on desk surfaces using Microsoft Kinect v2, combining computer vision with machine learning.

![Project Status](https://img.shields.io/badge/status-active-success.svg)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

---

## 🎯 Project Overview

This system enables indoor delivery robots to adaptively place packages on desks and tables without predefined docking points. It combines:

- **Box Detection**: Measures package dimensions using RGB-D data
- **Desk Analysis**: Detects free space regions on surfaces
- **Placement Feasibility**: Determines if and where a package can safely fit
- **Interactive GUI**: Step-by-step workflow for intuitive operation

**Key Innovation:** Unlike fixed-shelf delivery systems, SPO-T adapts to dynamic desk layouts and variable package sizes.

---

## ✨ Features

### 🖥️ Interactive GUI Application
- **4-Step Workflow**: Box measurement → Desk scanning → Analysis → Results
- **Real-time Feedback**: Live visualization with animated effects
- **Dual Detection Modes**: Traditional CV or YOLOv8 (toggle with 'M' key)
- **3D Visualization**: Animated box placement with clearance indicators

### 📦 Box Detection
- RGB-D based 3D bounding box detection
- Dimension estimation (length × width × height)
- Traditional CV or trained YOLOv8 model
- ~80-95% accuracy with YOLO (vs 40-60% traditional)

### 📊 Desk Space Analysis
- RANSAC plane detection for surface identification
- Occupancy grid mapping at 5mm resolution
- Multiple free region detection and ranking
- Real-time free space percentage tracking

### 🎯 Placement Feasibility
- Checks if box fits in detected free space
- Considers multiple orientations (0°, 90°)
- Calculates safety clearances
- Ranks placement candidates by score

---

## 📋 Requirements

- **OS**: Windows (for Kinect v2 SDK support)
- **Hardware**: Microsoft Kinect v2 sensor (Xbox One version)
- **Python**: 3.8 or higher
- **Optional**: NVIDIA GPU for YOLO training

---

## 🚀 Quick Start

### 1. Install Kinect SDK

Download: [Kinect for Windows SDK 2.0](https://www.microsoft.com/en-us/download/details.aspx?id=44561)

### 2. Set up Python Environment

```bash
# Create conda environment
conda create -n SPO-T python=3.9
conda activate SPO-T

# Install dependencies
pip install -r requirements.txt

# Patch pykinect2 for compatibility
python utils/setup_pykinect2.py
```

### 3. Run the Interactive GUI

```bash
python run_placement_system.py
```

**Controls:**
- `SPACE` - Capture/Next step
- `B` - Go back
- `R` - Reset to beginning
- `S` - Save screenshot
- `M` - Toggle detection method (CV/YOLO)
- `ESC` - Quit

---

## 🎬 Usage Workflow

### Step 1: Box Measurement
1. Point camera at the box you want to place
2. System detects box and shows dimensions
3. Press `SPACE` to capture

### Step 2: Desk Scanning
1. Point camera at the desk surface
2. Green overlay shows detected free space
3. Angle indicator ensures camera is horizontal
4. Press `SPACE` to scan

### Step 3: Analysis (Automated)
- System analyzes placement feasibility
- Animated progress with visual feedback
- Takes ~2 seconds

### Step 4: Results
- **✓ Feasible**: Shows 3D box projection at optimal position with clearance arrows
- **✗ Not Feasible**: Displays reason (e.g., "Box too large" or "Desk 85% occupied")

---

## 📁 Project Structure

```
SPO-T/
├── src/
│   ├── core/                    # Core detection modules
│   │   ├── box_detector.py      # Box detection (CV + YOLO)
│   │   ├── desk_monitor.py      # Desk space analysis
│   │   ├── placement_feasibility.py  # Placement logic
│   │   └── visualization_effects.py  # Animated effects
│   ├── gui/                     # GUI application
│   │   └── placement_gui.py     # Interactive 4-step GUI
│   ├── training/                # YOLO training scripts
│   │   └── train_yolo.py        # YOLOv8 training
│   └── evaluation/              # Evaluation tools
│       ├── evaluate_system.py   # Metrics & evaluation
│       └── compare_methods.py   # CV vs YOLO comparison
├── utils/                       # Utility scripts
│   ├── setup_pykinect2.py       # Kinect setup
│   └── test_setup.py            # Test installation
├── docs/                        # Documentation
│   ├── FEATURES.md              # Feature comparison
│   ├── TRAINING_README.md       # Training guide
│   └── flowcharts/              # Algorithm flowcharts
├── legacy/                      # Original demo scripts
│   └── main.py                  # Original placement optimizer
├── data/                        # Data storage
│   ├── test_scenarios/          # Test cases
│   └── captures/                # Screenshots
├── models/                      # Trained models
│   └── runs/                    # YOLO model outputs
├── run_placement_system.py      # 🎯 Main launcher
└── README.md
```

---

## 🤖 Training YOLO Model (Optional)

For improved box detection accuracy, train a YOLOv8 model:

```bash
# 1. Install training dependencies
pip install ultralytics pillow pyyaml

# 2. Download RGB-D dataset (Washington RGB-D Object Dataset)
# Place in: C:\Users\[User]\Downloads\rgbd-scenes-v2_imgs\rgbd-scenes-v2

# 3. Train model (2-6 hours)
python src/training/train_yolo.py --dataset_path "path/to/rgbd-scenes-v2" --epochs 50 --model_size n

# 4. Model saved to: models/runs/detect/box_detection/weights/best.pt
```

**Performance Improvement:**
- Traditional CV: ~40-60% accuracy, ±5-10cm error
- YOLOv8: ~80-95% accuracy, ±1-2cm error

See [`docs/TRAINING_README.md`](docs/TRAINING_README.md) for detailed training guide.

---

## 📊 Evaluation & Testing

### Capture Test Scenarios

Use the GUI to capture test cases:
1. Run `python run_placement_system.py`
2. Complete workflow for different scenarios (empty desk, small box, cluttered desk, etc.)
3. Press `S` to save screenshots
4. Scenarios saved to `data/test_scenarios/`

### Run Evaluation

```bash
# Evaluate system accuracy
python src/evaluation/evaluate_system.py

# Compare Traditional CV vs YOLO
python src/evaluation/compare_methods.py
```

Results saved to `evaluation_results.md`.

---

## 🔬 How It Works

### 1. Plane Detection (RANSAC)
- Samples 3 random points from 3D point cloud
- Calculates plane equation: **n·x + d = 0**
- Counts inlier points within ±10mm threshold
- Refines with PCA on all inliers

### 2. Occupancy Grid
- Projects 3D points onto plane coordinate system (u, v, h)
- Creates 5mm × 5mm grid cells
- Marks cells occupied if height ∈ [5mm, 80mm]
- Applies morphological filtering to remove noise

### 3. Free Region Detection
- Finds connected components in free space
- Filters regions by minimum area (>0.001 m²)
- Ranks by size and accessibility

### 4. Feasibility Analysis
- Tests box placement at each region centroid
- Tries multiple orientations (0°, 90°)
- Checks if box footprint fits without obstacles
- Calculates clearance distances (front, back, left, right)
- Scores candidates (70% clearance + 30% balance)

---

## 📈 Results (Preliminary)

### Detection Accuracy
| Metric | Traditional CV | YOLOv8 |
|--------|---------------|---------|
| Dimension MAE | 2.5 cm | 1.2 cm |
| Volume Error | 15% | 8% |
| Detection Time | 50 ms | 30 ms |

### Placement Decisions
- **Accuracy**: ~85-90% correct feasibility decisions
- **Processing Time**: ~150ms per analysis
- **Success Rate**: 92% when desk >40% free

---

## 🛠️ Legacy Applications

Individual components available as standalone scripts:

```bash
# Original placement optimizer (fixed package size)
python legacy/main.py

# Desk space monitor (continuous tracking)
python src/core/desk_monitor.py

# Box detector (standalone detection)
python src/core/box_detector.py
```

See [`docs/FEATURES.md`](docs/FEATURES.md) for feature comparison.

---

## 🐛 Troubleshooting

### Kinect Not Detected
1. Verify Kinect SDK 2.0 is installed
2. Check USB 3.0 connection
3. Ensure Kinect appears in Device Manager

### pykinect2 Errors
```bash
# Fix structure size issues
python utils/setup_pykinect2.py

# Common errors:
# - "AssertionError: 80" → Run setup script
# - "Wrong version" → Run setup script
# - "time.clock()" → Run setup script
```

### YOLO Model Not Found
- Train model first: `python src/training/train_yolo.py`
- Or use Traditional CV mode (no training required)

---

## 📚 Documentation

- [`docs/QUICKSTART.md`](docs/QUICKSTART.md) - Quick installation guide
- [`docs/TRAINING_README.md`](docs/TRAINING_README.md) - YOLOv8 training guide
- [`docs/FEATURES.md`](docs/FEATURES.md) - Feature comparison
- [`docs/flowcharts/`](docs/flowcharts/) - Algorithm flowcharts

---

## 🎓 Course Context

**AER1515 Project - University of Toronto**

This project extends previous work in MIE1075 (robot design) by adding:
- Adaptive surface detection (vs. fixed April tag shelves)
- Package dimension estimation
- Dynamic placement feasibility analysis

**Key Course Topics Applied:**
- Camera models & coordinate transformations
- 2D/3D segmentation (plane fitting, region detection)
- Point cloud analysis (RANSAC, grid mapping)
- RGB-D integration (depth + color fusion)
- Machine learning (YOLOv8 training)

---

## 🙏 Acknowledgments

- **Microsoft Kinect SDK** - Depth sensing
- **pykinect2** - Python Kinect interface
- **OpenCV** - Computer vision algorithms
- **Ultralytics YOLOv8** - Object detection
- **Washington RGB-D Object Dataset** - Training data

---

## 📄 License

MIT License - See [LICENSE](LICENSE) file for details.

---

## 🚀 Future Enhancements

- [ ] Multi-box placement optimization
- [ ] Robotic arm integration (MoveIt)
- [ ] Web interface for remote monitoring
- [ ] Pure camera-based model (no depth sensor)
- [ ] Kinect 360 comparison (structured light vs ToF)
- [ ] Mobile app for status monitoring

---

## 📞 Contact

For questions or issues, please open an issue on GitHub or contact the project maintainer.

**Project Repository**: [GitHub Link]

---

**Made with ❤️ for AER1515 - Perception for Robotics**
