"""
Dataset Preparation Helper
==========================

Checks and prepares the Washington RGB-D dataset for training.
"""

from pathlib import Path
import sys

def check_dataset(dataset_path: str):
    """Check if dataset exists and is properly structured."""
    print("=" * 70)
    print("Washington RGB-D Dataset Verification")
    print("=" * 70)
    
    path = Path(dataset_path)
    
    if not path.exists():
        print(f"\n✗ Dataset not found at: {path}")
        print("\nPlease download the dataset from:")
        print("  https://rgbd-dataset.cs.washington.edu/dataset/rgbd-scenes-v2/")
        print("\nExpected location:")
        print(f"  {path}")
        return False
    
    print(f"\n✓ Dataset directory found: {path}")
    
    # Check for imgs directory
    imgs_dir = path / "imgs"
    if not imgs_dir.exists():
        print(f"\n✗ 'imgs' directory not found")
        print(f"   Expected: {imgs_dir}")
        return False
    
    print(f"✓ Images directory found: {imgs_dir}")
    
    # Count scene directories
    scenes = list(imgs_dir.glob("scene_*"))
    if not scenes:
        print(f"\n✗ No scene directories found in {imgs_dir}")
        return False
    
    print(f"✓ Found {len(scenes)} scene directories")
    
    # Check first scene for structure
    first_scene = sorted(scenes)[0]
    color_files = list(first_scene.glob("*-color.png"))
    depth_files = list(first_scene.glob("*-depth.png"))
    
    print(f"\n{first_scene.name}:")
    print(f"  Color images: {len(color_files)}")
    print(f"  Depth images: {len(depth_files)}")
    
    if not color_files or not depth_files:
        print("\n✗ Missing color or depth images")
        return False
    
    print("\n" + "=" * 70)
    print("✓ DATASET READY FOR TRAINING")
    print("=" * 70)
    print("\nTo train YOLOv8 model:")
    print(f'  python src/training/train_yolo.py --dataset_path "{dataset_path}" --epochs 50 --model_size n')
    print("\nFor faster training (nano model, ~2 hours on GPU):")
    print(f'  python src/training/train_yolo.py --dataset_path "{dataset_path}" --epochs 50 --model_size n')
    print("\nFor better accuracy (medium model, ~6 hours on GPU):")
    print(f'  python src/training/train_yolo.py --dataset_path "{dataset_path}" --epochs 100 --model_size m')
    
    return True


def main():
    """Main entry point."""
    # Default path (adjust based on user's system)
    default_path = r"C:\Users\Sarri\Downloads\rgbd-scenes-v2_imgs\rgbd-scenes-v2"
    
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
    else:
        dataset_path = default_path
    
    print(f"\nChecking dataset at: {dataset_path}\n")
    
    if check_dataset(dataset_path):
        print("\n✓ Ready to proceed with training!")
    else:
        print("\n✗ Please fix dataset issues before training")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

