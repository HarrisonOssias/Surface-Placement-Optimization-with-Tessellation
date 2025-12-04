# SPO-T Quick Reference Card

**Fast reference for common tasks**

---

## 🚀 Quick Start

```bash
# 1. Activate environment
conda activate SPO-T

# 2. Run the system
python run_placement_system.py
```

---

## ⌨️ Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `SPACE` | Capture / Next step |
| `B` | Back to previous step |
| `R` | Reset to beginning |
| `S` | Save screenshot |
| `M` | Toggle CV ↔ YOLO |
| `ESC` | Quit |

---

## 📋 Workflow Steps

1. **Box Measurement**
   - Point at box → `SPACE` to capture

2. **Desk Scanning**
   - Point at desk → Check angle → `SPACE` to scan

3. **Analysis**
   - Automatic (2 seconds)

4. **Results**
   - View placement decision
   - `S` to save

---

## 🔧 Common Commands

### Testing
```bash
# Test Kinect connection
python utils/test_setup.py

# Run system
python run_placement_system.py

# Check dataset
python prepare_dataset.py
```

### Training (Optional)
```bash
# Train YOLO model (2-6 hours)
python src/training/train_yolo.py \
  --dataset_path "C:\Path\To\rgbd-scenes-v2" \
  --epochs 50 \
  --model_size n
```

### Evaluation
```bash
# Run evaluation
python src/evaluation/evaluate_system.py

# Compare methods
python src/evaluation/compare_methods.py
```

---

## 📁 Important Files

| File | Purpose |
|------|---------|
| `run_placement_system.py` | Main launcher |
| `README.md` | Project overview |
| `USAGE_GUIDE.md` | Detailed usage |
| `TESTING_INSTRUCTIONS.md` | Testing guide |
| `PROJECT_STATUS.md` | Implementation status |

---

## 🎯 Workflow Visual

```
Step 1: BOX          Step 2: DESK         Step 3: ANALYZE      Step 4: RESULTS
┌──────────┐         ┌──────────┐         ┌──────────┐         ┌──────────┐
│  📦      │         │  🏢      │         │ ⚙️⚙️⚙️  │         │ ✅ or ❌  │
│ Point at │  ───>   │ Point at │  ───>   │ Auto     │  ───>   │ Show     │
│ box      │         │ desk     │         │ process  │         │ result   │
│ SPACE    │         │ SPACE    │         │ ~2 sec   │         │ S=save   │
└──────────┘         └──────────┘         └──────────┘         └──────────┘
```

---

## 🐛 Quick Troubleshooting

| Problem | Solution |
|---------|----------|
| No box detected | Better lighting, move closer |
| No desk detected | Adjust camera angle (±15°) |
| System slow | Close other apps, use CV mode |
| Kinect not found | Check USB 3.0, restart Kinect |
| Module not found | `conda activate SPO-T` |

---

## 📊 Expected Accuracy

| Method | Dimension Error | Speed |
|--------|----------------|-------|
| Traditional CV | ±5-10 cm | Fast (30ms) |
| YOLO | ±1-2 cm | Medium (50ms) |

---

## 🎥 Demo Video Checklist

- [ ] Empty desk → Box fits ✓
- [ ] Cluttered desk → Box doesn't fit ✗
- [ ] Show all 4 workflow steps
- [ ] Demonstrate clearance visualization
- [ ] Toggle CV ↔ YOLO (if trained)
- [ ] Length: 3-5 minutes

---

## 📞 Need Help?

1. **Usage questions** → `USAGE_GUIDE.md`
2. **Testing help** → `TESTING_INSTRUCTIONS.md`
3. **Training help** → `docs/TRAINING_README.md`
4. **Technical details** → `README.md`

---

## 🎓 Video Due: Nov 28 | Report Due: Dec 13

**Priority Actions:**
1. ✅ Test system (1 hour)
2. ✅ Record demo video (2 hours)
3. ⏳ Train YOLO (optional, for better results)
4. ⏳ Capture test scenarios (for evaluation)

---

**Quick tips:**
- Start with empty desk (easy success case)
- Use medium-sized box (15-30cm)
- Ensure good lighting
- Keep camera 0.5-1.5m from objects
- Practice workflow before recording

**Good luck! 🚀**

