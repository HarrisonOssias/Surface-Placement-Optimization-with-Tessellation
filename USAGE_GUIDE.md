# SPO-T Usage Guide

Complete guide for using the Surface Placement Optimization with Tessellation system.

---

## 🎯 Quick Start

### 1. Run the Interactive GUI

```bash
python run_placement_system.py
```

This launches the main application with the 4-step workflow.

---

## 📖 Step-by-Step Usage

### Step 1: Box Measurement

**Goal**: Measure the dimensions of the box you want to place

1. Hold or place the box in view of the Kinect camera
2. Ensure the entire box is visible
3. The system will automatically detect the box and show dimensions overlaid on the video
4. When satisfied with detection, press `SPACE` to capture

**Tips:**
- Keep box at least 0.5m from camera
- Ensure good lighting
- Avoid reflective or transparent surfaces
- If multiple boxes detected, largest one is selected

**Example output:**
```
✓ Box captured: 25.3 x 18.1 x 12.4 cm
```

### Step 2: Desk Scanning

**Goal**: Scan the desk surface to detect free space

1. Point camera at the desk where you want to place the box
2. Keep camera roughly horizontal (±15° angle tolerance)
   - Green angle indicator = good
   - Red indicator = adjust camera angle
3. Free space regions will be shown as green overlay
4. Press `SPACE` when ready to analyze

**Tips:**
- Camera should be 0.5-1.5m from desk
- Desk should be relatively horizontal
- Remove hands/arms from view
- Wait for stable free-space overlay

**What you'll see:**
- Green dots on free space regions
- Angle indicator (top-right)
- Free space percentage

### Step 3: Analysis (Automatic)

**Goal**: System analyzes placement feasibility

- Automatic processing (~2 seconds)
- Progress bar shows analysis status
- Spinning animation indicates processing

**What happens:**
1. Identifies distinct free regions
2. Tests box placement at each region
3. Tries multiple orientations (0°, 90°)
4. Calculates clearance distances
5. Ranks candidates by safety score

### Step 4: Results

**Goal**: View placement decision and visualization

#### If Placement is Feasible ✓

You'll see:
- **3D box projection** at optimal position (green outline with depth effect)
- **Clearance arrows** showing distances to obstacles
- **Info panel** with:
  - Placement score (0-100%)
  - Orientation angle
  - Clearance distances (front, back, left, right)
  - Desk free space percentage

**Example:**
```
✓ PLACEMENT FEASIBLE
Score: 87%
Orientation: 0°

Clearances:
  Front: 8.2cm
  Back:  12.5cm
  Left:  6.1cm
  Right: 10.3cm

Desk Free: 65.3%
```

#### If Placement is Not Feasible ✗

You'll see:
- **Red X** over the desk
- **Reason** for infeasibility
- **Suggestions** (if applicable)

**Common reasons:**
- "Desk 85% occupied" → Clear more space
- "Box too large" → Box doesn't fit in largest free region
- "No space available" → Desk completely cluttered

---

## ⌨️ Keyboard Controls

### Navigation
- `SPACE` - Proceed to next step / Capture current state
- `B` - Go back to previous step
- `R` - Reset to Step 1 (start over)
- `ESC` - Quit application

### Features
- `S` - Save screenshot of current view (saves to `data/captures/`)
- `M` - Toggle detection method (Traditional CV ↔ YOLO)

### Detection Method Toggle

Press `M` to switch between:
- **Traditional CV**: Fast, no training needed, ~40-60% accuracy
- **YOLO**: Slower, requires trained model, ~80-95% accuracy

**Note**: YOLO requires a trained model at `models/runs/detect/box_detection/weights/best.pt`

---

## 🔧 Advanced Usage

### Training YOLO Model

For better detection accuracy:

```bash
# 1. Verify dataset
python prepare_dataset.py

# 2. Train model (2-6 hours depending on GPU)
python src/training/train_yolo.py \
  --dataset_path "C:\Path\To\rgbd-scenes-v2" \
  --epochs 50 \
  --model_size n

# 3. Model auto-loads in GUI (press M to enable)
```

### Capturing Test Scenarios

To create test cases for evaluation:

1. Run through complete workflow (Steps 1-4)
2. Press `S` in Step 4 to save scenario
3. Manually create `ground_truth.json` in scenario folder:

```json
{
  "scenario_id": "scenario_01",
  "box_dimensions": [0.25, 0.18, 0.12],
  "should_fit": true,
  "desk_free_percentage": 65.0,
  "notes": "Small box on moderately cluttered desk"
}
```

### Running Evaluation

```bash
# Evaluate all test scenarios
python src/evaluation/evaluate_system.py

# Compare CV vs YOLO
python src/evaluation/compare_methods.py
```

---

## 📊 Understanding the Results

### Placement Score

Score is calculated based on:
- **70%** - Average clearance distance
- **30%** - Clearance balance (similarity on all sides)

**Score interpretation:**
- 90-100%: Excellent, large clearances
- 70-89%: Good, safe placement
- 50-69%: Acceptable, tight fit
- <50%: Marginal, minimal clearance

### Clearance Distances

Distances from box edges to nearest obstacles:
- **Front**: Obstacle distance in negative v direction
- **Back**: Obstacle distance in positive v direction
- **Left**: Obstacle distance in negative u direction
- **Right**: Obstacle distance in positive u direction

### Desk Free Percentage

Percentage of desk surface that is unoccupied:
- **>70%**: Plenty of space, multiple placement options
- **40-70%**: Moderate space, limited options
- **20-40%**: Cluttered, difficult placement
- **<20%**: Very cluttered, likely infeasible

---

## 🐛 Troubleshooting

### "No box detected"

**Solutions:**
1. Ensure box is fully visible
2. Move camera closer (0.5-1.5m optimal)
3. Improve lighting
4. Try different box material (avoid shiny/reflective)
5. Use larger box (>10cm minimum)

### "No desk surface detected"

**Solutions:**
1. Adjust camera angle (horizontal ±15°)
2. Ensure desk is in frame
3. Remove obstructions from view
4. Move camera to better vantage point

### "Placement not feasible" (but you think it should fit)

**Reasons:**
1. Safety margins are conservative (15mm default)
2. Desk surface may have small undetected obstacles
3. Box dimensions may be overestimated

**Solutions:**
- Clear more space around target area
- Retry desk scanning from different angle
- Manually verify box dimensions

### Camera lag / slow performance

**Solutions:**
1. Close other applications
2. Ensure Kinect has USB 3.0 connection
3. Use Traditional CV mode (faster than YOLO)
4. Reduce window size (edit `display_width/height` in code)

---

## 💡 Tips for Best Results

### Box Detection
- Use boxes with matte finish (not glossy)
- Ensure uniform color/texture
- Avoid boxes that blend with background
- Keep box fully visible (no occlusions)

### Desk Scanning
- Remove temporary clutter first
- Ensure consistent lighting
- Keep camera steady during scan
- Multiple angles give better confidence

### Placement Decision
- Larger clearances = more reliable placement
- Consider rotation if placement fails
- Free space >40% usually allows placement
- Test with different box sizes

---

## 📝 Output Files

### Screenshots (`data/captures/`)
- `screenshot_YYYYMMDD_HHMMSS.png` - Saved views

### Test Scenarios (`data/test_scenarios/scenario_XX/`)
- `box_color.png` - Box image
- `box_depth.png` - Box depth map
- `desk_color.png` - Desk image
- `desk_depth.png` - Desk depth map
- `ground_truth.json` - Manual annotations

### Evaluation Results
- `evaluation_results.md` - System performance metrics
- Model outputs in `models/runs/`

---

## 🎥 Creating a Demo Video

### Recommended Sequence

1. **Introduction** (30 sec)
   - Show empty desk
   - Introduce box

2. **Box Measurement** (20 sec)
   - Point at box
   - Show detection overlay
   - Capture dimensions

3. **Desk Scanning** (30 sec)
   - Pan to desk
   - Show angle adjustment
   - Show free space overlay
   - Capture scan

4. **Analysis** (15 sec)
   - Show progress animation
   - Let it complete

5. **Results - Success** (30 sec)
   - Show 3D visualization
   - Highlight clearances
   - Read info panel

6. **Results - Failure** (20 sec)
   - Add clutter to desk
   - Rescan
   - Show "Not Feasible" result

7. **Method Comparison** (30 sec)
   - Press 'M' to switch to YOLO
   - Rerun detection
   - Show improved accuracy

**Total: ~3 minutes**

### Recording Tips
- Use screen recording software (OBS Studio)
- Add voiceover narration
- Highlight key features with annotations
- Show real robot placement if available

---

## 📞 Getting Help

If you encounter issues:

1. Check this guide's troubleshooting section
2. Review `README.md` for setup issues
3. Check console output for error messages
4. Verify Kinect is connected and recognized by Windows

---

**Need more information?** See additional documentation:
- [`README.md`](README.md) - Project overview
- [`docs/TRAINING_README.md`](docs/TRAINING_README.md) - YOLO training details
- [`docs/FEATURES.md`](docs/FEATURES.md) - Feature comparison

---

**Happy Placing! 📦→🎯**

