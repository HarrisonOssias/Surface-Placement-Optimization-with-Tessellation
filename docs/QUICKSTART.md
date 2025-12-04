# Quick Start Guide

## 🚀 Get Started in 5 Minutes

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Patch pykinect2

```bash
python setup_pykinect2.py
```

### Step 3: Test Your Setup

```bash
python test_setup.py
```

### Step 4: Run the Application

**Option A - Package Placement:**
```bash
python main.py
```

**Option B - Desk Space Monitor:**
```bash
python desk_monitor.py
```

---

## 📋 What Each Script Does

### `main.py` - Package Placement
- **Goal:** Find where to place a package
- **Output:** Green crosshair marker at optimal location
- **Use:** Press 'S' to save snapshot, ESC to quit

### `desk_monitor.py` - Free Space Detection  
- **Goal:** Monitor available desk space
- **Output:** Free space %, multiple views, data logs
- **Use:** Press 'H' for heatmap, 'S' to save, ESC to quit

---

## 🎯 Typical Workflow

1. **First Time Setup** (once):
   ```bash
   pip install -r requirements.txt
   python setup_pykinect2.py
   python test_setup.py
   ```

2. **Daily Use** (every time):
   ```bash
   # For placement optimization:
   python main.py
   
   # For space monitoring:
   python desk_monitor.py
   ```

---

## ⚙️ Configuration

Edit parameters at the top of each script:

### main.py
```python
PACKAGE_L = 0.30  # Package length (meters)
PACKAGE_W = 0.20  # Package width (meters)
MARGIN = 0.015    # Safety buffer (meters)
```

### desk_monitor.py
```python
H_MIN = 0.005            # Min obstacle height
H_MAX = 0.08             # Max obstacle height  
CLUTTER_THRESHOLD = 70   # Alert when >70% full
```

---

## 📊 Output Files

### main.py outputs:
- `captures/[timestamp]_color.png`
- `captures/[timestamp]_depth.png`

### desk_monitor.py outputs:
- `desk_monitor_[timestamp].csv` - Time-series data
- `desk_monitor_heatmap_[timestamp].png` - Usage heatmap
- `desk_snapshot_[timestamp].png` - Current view

---

## 🐛 Troubleshooting

### "AssertionError: 80"
```bash
python setup_pykinect2.py
```

### "ImportError: Wrong version"  
```bash
python setup_pykinect2.py
```

### "AttributeError: module 'time' has no attribute 'clock'"
```bash
python setup_pykinect2.py
```

### No Kinect frames
- Check USB 3.0 connection
- Install Kinect SDK 2.0
- Close other programs using Kinect

---

## 📖 Learn More

- **README.md** - Full documentation
- **FEATURES.md** - Detailed feature comparison
- **main.py** - Well-commented code with explanations

---

## 💡 Tips

1. **Place Kinect 0.5-1m above desk** for best view
2. **Point slightly downward** to see full surface  
3. **Avoid direct sunlight** (interferes with infrared)
4. **Clear warm-up period** - first few frames may be noisy
5. **Press 'S' frequently** to save interesting results

---

## 🎓 Understanding the Output

### main.py Display:
- **Green Cross** = Optimal placement location
- **"PLACEMENT"** text = Confirmation of valid location
- **Depth view** = Grayscale distance visualization
- **Feasibility heatmap** = Blue=bad, Red=good locations

### desk_monitor.py Display:
- **Green dots** = Free space overlay
- **Info panel** = Real-time metrics (top-left)
- **Grid view** = Occupied (colored) | Free (white)
- **Heatmap (H key)** = Hot colors = frequently occupied

---

## 🔗 Quick Commands Reference

```bash
# Setup (once)
pip install -r requirements.txt
python setup_pykinect2.py

# Test
python test_setup.py

# Run package placement
python main.py

# Run desk monitor
python desk_monitor.py

# Check if Kinect is working (from test_setup.py result)
# Look for "✓ Received depth frame"
```

---

## Next Steps

1. ✅ Got it working? Read **FEATURES.md** for advanced usage
2. 🔧 Want to customize? Edit configuration parameters in scripts
3. 📊 Need data analysis? Check CSV output from desk_monitor.py
4. 🤖 Building automation? Use placement coordinates from main.py
5. 📚 Want to understand? Read detailed comments in source code

---

**Need Help?** Open an issue on GitHub with:
- Output from `python test_setup.py`
- Error messages
- Python version
- Operating system

Happy monitoring! 🎉

