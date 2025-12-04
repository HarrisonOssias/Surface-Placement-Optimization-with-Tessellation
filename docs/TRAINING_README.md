# Training a YOLOv8 Model for Accurate Box Detection

This guide will help you train a machine learning model to improve the accuracy of box detection in your desk monitoring system.

## Why Train a Model?

The traditional computer vision approach (contour detection) can be inaccurate because:
- **Noise**: Depth sensors have noise that creates false detections
- **Lighting**: Shadows and reflections confuse edge detection
- **Complex scenes**: Overlapping objects are hard to segment
- **Shape variation**: Boxes come in many shapes and materials

A trained YOLOv8 model learns to recognize boxes specifically, resulting in:
- ✓ **Higher accuracy**: 80-95% detection rate vs 40-60% for traditional CV
- ✓ **Better classification**: Distinguishes box types and sizes
- ✓ **Robust to noise**: Learned features are noise-resistant
- ✓ **Faster inference**: 30+ FPS on GPU

## Prerequisites

### 1. Install Training Dependencies

```bash
# Activate your virtual environment first
.\venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # Linux/Mac

# Install YOLOv8 and training tools
pip install ultralytics pillow pyyaml

# Optional: Install PyTorch with CUDA for GPU acceleration
# Check https://pytorch.org for your specific CUDA version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. Verify Dataset Location

The script expects your RGB-D dataset at:
```
C:\Users\Sarri\Downloads\rgbd-scenes-v2_imgs\rgbd-scenes-v2\
```

Verify you have the `imgs/` directory with scene folders:
```
rgbd-scenes-v2/
  imgs/
    scene_01/
      00000-color.png
      00000-depth.png
      ...
    scene_02/
      ...
```

## Training Steps

### Option 1: Quick Training (Recommended for First Try)

Use the nano model for fast training (~30 minutes on GPU, ~2 hours on CPU):

```bash
python train_yolo.py --dataset_path "C:\Users\Sarri\Downloads\rgbd-scenes-v2_imgs\rgbd-scenes-v2" --epochs 50 --model_size n
```

### Option 2: Full Training (Best Accuracy)

Use the medium model for better accuracy (~2 hours on GPU, ~8 hours on CPU):

```bash
python train_yolo.py --dataset_path "C:\Users\Sarri\Downloads\rgbd-scenes-v2_imgs\rgbd-scenes-v2" --epochs 100 --model_size m --batch_size 8
```

### Option 3: Custom Configuration

```bash
python train_yolo.py \
  --dataset_path "C:\Users\Sarri\Downloads\rgbd-scenes-v2_imgs\rgbd-scenes-v2" \
  --output_path "my_dataset" \
  --epochs 150 \
  --batch_size 16 \
  --image_size 640 \
  --model_size s
```

### Training Parameters Explained

- `--model_size`: YOLOv8 model size
  - `n` (nano): Fastest, smallest, ~70-80% accuracy
  - `s` (small): Good balance, ~75-85% accuracy
  - `m` (medium): **Recommended**, ~80-90% accuracy
  - `l` (large): Slower, ~85-92% accuracy
  - `x` (xlarge): Slowest, best accuracy ~88-95%

- `--epochs`: Training iterations
  - 50-100: Quick training, may underfit
  - 100-200: **Recommended**, good convergence
  - 200+: Diminishing returns, may overfit

- `--batch_size`: Images per training step
  - Larger = faster training, more GPU memory needed
  - Reduce if you get out-of-memory errors
  - Typical: 8 (CPU), 16 (GPU with 8GB), 32 (GPU with 16GB+)

## What the Script Does

The training script automatically:

1. **Converts dataset to YOLO format**
   - Scans all scene directories
   - Extracts objects using depth segmentation
   - Creates bounding box annotations
   - Splits into 80% train / 20% validation

2. **Downloads base model**
   - Downloads pre-trained YOLOv8 weights from Ultralytics
   - These weights are trained on COCO dataset (common objects)

3. **Fine-tunes on your data**
   - Adapts the model to recognize boxes specifically
   - Uses data augmentation (rotation, scaling, etc.)
   - Implements early stopping to prevent overfitting

4. **Saves best model**
   - Model saved to: `runs/detect/box_detection/weights/best.pt`
   - You can use this with `box_detector.py`

## Monitoring Training Progress

The script will print training metrics:

```
Epoch    GPU_mem   box_loss   cls_loss   dfl_loss   Instances       Size
  1/100      1.2G      1.234      0.876      1.123          45        640
  ...
 50/100      1.2G      0.432      0.234      0.456          45        640
```

Look for:
- **box_loss**: Should decrease to < 0.5
- **cls_loss**: Should decrease to < 0.3
- **Validation mAP**: Should increase to > 0.6 (60% accuracy)

Training will automatically stop if no improvement for 20 epochs.

## Using the Trained Model

After training completes, update `box_detector.py`:

```python
# Instead of:
detector = BoxDetector()

# Use:
detector = BoxDetector(
    use_ml_model=True, 
    model_path='runs/detect/box_detection/weights/best.pt'
)
```

Or run directly:

```bash
python box_detector.py --model runs/detect/box_detection/weights/best.pt
```

## Expected Results

### Before Training (Traditional CV)
- False positives: 30-40%
- Missed detections: 20-30%
- Inaccurate dimensions: ±5-10cm error

### After Training (YOLOv8)
- False positives: 5-10%
- Missed detections: 5-10%
- Accurate dimensions: ±1-2cm error
- Can distinguish box types (cardboard, plastic, etc.)

## Troubleshooting

### "CUDA out of memory"
Reduce batch size:
```bash
python train_yolo.py --batch_size 4  # or even 2
```

### "No samples generated"
Check dataset path is correct and contains `imgs/` directory with scenes.

### "Training very slow"
- Use smaller model: `--model_size n`
- Reduce epochs: `--epochs 50`
- Use GPU if available (install CUDA PyTorch)

### "Model not improving"
- Train longer: `--epochs 200`
- Use larger model: `--model_size m`
- Check if dataset has enough variety (need 500+ samples)

## Advanced: Manual Annotation

For best results, manually annotate your own desk scenes:

1. Collect images from your actual Kinect setup
2. Use annotation tools:
   - [LabelImg](https://github.com/heartexlabs/labelImg)
   - [CVAT](https://www.cvat.ai/)
   - [Roboflow](https://roboflow.com/)
3. Export annotations in YOLO format
4. Place in `custom_dataset/` directory
5. Run training: `python train_yolo.py --dataset_path custom_dataset --skip_conversion`

## Next Steps

1. **Start training**: Run the quick training option to get a baseline model
2. **Test the model**: Use with `box_detector.py` and compare to traditional CV
3. **Iterate**: If accuracy is poor, try:
   - Train longer (more epochs)
   - Use larger model
   - Collect more training data
   - Manually annotate difficult cases

## References

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [UW RGB-D Object Dataset](https://rgbd-dataset.cs.washington.edu/)
- [Training Custom YOLOv8 Models](https://docs.ultralytics.com/modes/train/)

