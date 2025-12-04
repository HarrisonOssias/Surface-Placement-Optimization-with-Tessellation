"""Quick dataset checker - shows what you have"""

import os
from pathlib import Path

IMGS_PATH = r"C:\Users\Sarri\Downloads\rgbd-scenes-v2_imgs\rgbd-scenes-v2\imgs"
PC_PATH = r"C:\Users\Sarri\Downloads\rgbd-scenes-v2_pc\rgbd-scenes-v2\pc"

print("=" * 70)
print("RGB-D SCENES DATASET v2 - QUICK CHECK")
print("=" * 70)

imgs_path = Path(IMGS_PATH)
pc_path = Path(PC_PATH)

if not imgs_path.exists():
    print(f"ERROR: Images path not found: {imgs_path}")
    exit(1)

print(f"\nImages: {imgs_path}")
print(f"Point Clouds: {pc_path}")
print()

total_frames = 0
scenes = []

for scene_dir in sorted(imgs_path.iterdir()):
    if scene_dir.is_dir() and scene_dir.name.startswith('scene_'):
        num = scene_dir.name.split('_')[1]
        color_imgs = list(scene_dir.glob('*-color.png'))
        scenes.append((scene_dir.name, len(color_imgs), num))
        total_frames += len(color_imgs)

print(f"Found {len(scenes)} scenes with {total_frames} total frames\n")
print(f"{'Scene':<15} {'Frames':<10} {'Has PC':<10}")
print("-" * 40)

for name, count, num in scenes:
    pc_file = pc_path / f"{num}.ply"
    has_pc = "Yes" if pc_file.exists() else "No"
    print(f"{name:<15} {count:<10} {has_pc:<10}")

print("\n" + "=" * 70)
print("\nNext steps:")
print("  1. Run: python explore_dataset.py")
print("     - Browse the dataset interactively")
print("  2. Run: python box_detector.py")
print("     - Test box detection on live Kinect")
print("=" * 70)

