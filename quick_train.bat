@echo off
REM Quick Training Script for YOLOv8 Box Detection
REM This batch file makes it easy to start training

echo ========================================================================
echo YOLOv8 Box Detection - Quick Training
echo ========================================================================
echo.

REM Check if virtual environment is activated
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please activate your virtual environment:
    echo   .\venv\Scripts\activate
    exit /b 1
)

REM Check if ultralytics is installed
pip show ultralytics >nul 2>&1
if errorlevel 1 (
    echo [INFO] Installing training dependencies...
    pip install ultralytics pillow pyyaml
    echo.
)

REM Set default dataset path
set DATASET_PATH=C:\Users\Sarri\Downloads\rgbd-scenes-v2_imgs\rgbd-scenes-v2

REM Check if dataset exists
if not exist "%DATASET_PATH%\imgs" (
    echo [ERROR] Dataset not found at: %DATASET_PATH%
    echo Please update DATASET_PATH in this script or extract rgbd-scenes-v2_imgs.zip
    exit /b 1
)

echo [INFO] Dataset found: %DATASET_PATH%
echo.
echo Choose training mode:
echo   1. Quick Training (nano model, 50 epochs, ~30 min)
echo   2. Balanced Training (small model, 100 epochs, ~1 hour)  [RECOMMENDED]
echo   3. Best Accuracy (medium model, 150 epochs, ~2-3 hours)
echo.

set /p choice="Enter choice (1-3): "

if "%choice%"=="1" (
    echo.
    echo Starting QUICK training...
    python train_yolo.py --dataset_path "%DATASET_PATH%" --epochs 50 --model_size n --batch_size 16
) else if "%choice%"=="2" (
    echo.
    echo Starting BALANCED training...
    python train_yolo.py --dataset_path "%DATASET_PATH%" --epochs 100 --model_size s --batch_size 12
) else if "%choice%"=="3" (
    echo.
    echo Starting BEST ACCURACY training...
    python train_yolo.py --dataset_path "%DATASET_PATH%" --epochs 150 --model_size m --batch_size 8
) else (
    echo Invalid choice. Exiting.
    exit /b 1
)

echo.
echo ========================================================================
echo Training complete!
echo.
echo Your trained model is saved at:
echo   runs\detect\box_detection\weights\best.pt
echo.
echo To use it with box_detector.py, update the code or run:
echo   python box_detector.py --model runs\detect\box_detection\weights\best.pt
echo ========================================================================
pause

