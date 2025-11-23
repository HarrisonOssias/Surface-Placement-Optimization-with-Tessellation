#!/bin/bash
# Quick Training Script for YOLOv8 Box Detection
# This script makes it easy to start training

echo "========================================================================"
echo "YOLOv8 Box Detection - Quick Training"
echo "========================================================================"
echo ""

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "[ERROR] Python not found. Please activate your virtual environment:"
    echo "  source venv/bin/activate"
    exit 1
fi

# Check if ultralytics is installed
if ! pip show ultralytics &> /dev/null; then
    echo "[INFO] Installing training dependencies..."
    pip install ultralytics pillow pyyaml
    echo ""
fi

# Set default dataset path (adjust as needed)
DATASET_PATH="${1:-./rgbd-scenes-v2}"

# Check if dataset exists
if [ ! -d "$DATASET_PATH/imgs" ]; then
    echo "[ERROR] Dataset not found at: $DATASET_PATH"
    echo "Usage: ./quick_train.sh <path_to_rgbd-scenes-v2>"
    echo "Or extract rgbd-scenes-v2_imgs.zip to current directory"
    exit 1
fi

echo "[INFO] Dataset found: $DATASET_PATH"
echo ""
echo "Choose training mode:"
echo "  1. Quick Training (nano model, 50 epochs, ~30 min)"
echo "  2. Balanced Training (small model, 100 epochs, ~1 hour)  [RECOMMENDED]"
echo "  3. Best Accuracy (medium model, 150 epochs, ~2-3 hours)"
echo ""

read -p "Enter choice (1-3): " choice

case $choice in
    1)
        echo ""
        echo "Starting QUICK training..."
        python train_yolo.py --dataset_path "$DATASET_PATH" --epochs 50 --model_size n --batch_size 16
        ;;
    2)
        echo ""
        echo "Starting BALANCED training..."
        python train_yolo.py --dataset_path "$DATASET_PATH" --epochs 100 --model_size s --batch_size 12
        ;;
    3)
        echo ""
        echo "Starting BEST ACCURACY training..."
        python train_yolo.py --dataset_path "$DATASET_PATH" --epochs 150 --model_size m --batch_size 8
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

echo ""
echo "========================================================================"
echo "Training complete!"
echo ""
echo "Your trained model is saved at:"
echo "  runs/detect/box_detection/weights/best.pt"
echo ""
echo "To use it with box_detector.py, update the code or run:"
echo "  python box_detector.py --model runs/detect/box_detection/weights/best.pt"
echo "========================================================================"

