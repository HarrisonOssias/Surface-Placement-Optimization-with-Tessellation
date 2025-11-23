"""
RGB-D Dataset Explorer
======================

Explore and prepare the UW RGB-D Scenes Dataset v2 for training.

This dataset contains:
- 14 scenes with furniture and tabletop objects
- RGB images (color)
- Depth images (aligned depth maps)
- Point clouds (.ply files)
- Object labels (.label files)

Usage:
    python explore_dataset.py
"""

import os
import cv2
import numpy as np
from pathlib import Path
import json

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Dataset paths (modify if needed)
IMGS_PATH = r"C:\Users\Sarri\Downloads\rgbd-scenes-v2_imgs\rgbd-scenes-v2\imgs"
PC_PATH = r"C:\Users\Sarri\Downloads\rgbd-scenes-v2_pc\rgbd-scenes-v2\pc"

# Output directory for processed data
OUTPUT_DIR = "data/processed_rgbd"

# ==============================================================================
# DATASET EXPLORER
# ==============================================================================

class RGBDDatasetExplorer:
    """Explore and analyze RGB-D Scenes Dataset v2."""
    
    def __init__(self, imgs_path, pc_path):
        self.imgs_path = Path(imgs_path)
        self.pc_path = Path(pc_path)
        self.scenes = []
        self._scan_dataset()
    
    def _scan_dataset(self):
        """Scan dataset and collect scene information."""
        if not self.imgs_path.exists():
            print(f"⚠️  Images path not found: {self.imgs_path}")
            return
        
        # Find all scenes
        for scene_dir in sorted(self.imgs_path.iterdir()):
            if scene_dir.is_dir() and scene_dir.name.startswith('scene_'):
                scene_num = scene_dir.name.split('_')[1]
                
                # Count images
                color_images = list(scene_dir.glob('*-color.png'))
                depth_images = list(scene_dir.glob('*-depth.png'))
                
                # Find corresponding point cloud
                pc_file = self.pc_path / f"{scene_num}.ply" if self.pc_path.exists() else None
                label_file = self.pc_path / f"{scene_num}.label" if self.pc_path.exists() else None
                
                scene_info = {
                    'name': scene_dir.name,
                    'number': scene_num,
                    'path': scene_dir,
                    'num_frames': len(color_images),
                    'color_images': color_images,
                    'depth_images': depth_images,
                    'pc_file': pc_file,
                    'label_file': label_file,
                    'has_pc': pc_file.exists() if pc_file else False,
                    'has_labels': label_file.exists() if label_file else False
                }
                
                self.scenes.append(scene_info)
    
    def print_summary(self):
        """Print dataset summary."""
        print("=" * 70)
        print("RGB-D SCENES DATASET v2 - SUMMARY")
        print("=" * 70)
        print(f"Images path: {self.imgs_path}")
        print(f"Point clouds path: {self.pc_path}")
        print(f"\nTotal scenes: {len(self.scenes)}")
        print(f"Total frames: {sum(s['num_frames'] for s in self.scenes)}")
        print()
        
        print("Scene Details:")
        print("-" * 70)
        print(f"{'Scene':<12} {'Frames':<10} {'PC':<8} {'Labels':<8} {'Path'}")
        print("-" * 70)
        
        for scene in self.scenes:
            pc_status = "Yes" if scene['has_pc'] else "No"
            label_status = "Yes" if scene['has_labels'] else "No"
            print(f"{scene['name']:<12} {scene['num_frames']:<10} {pc_status:<8} {label_status:<8} {scene['path'].name}")
        
        print("=" * 70)
    
    def view_scene(self, scene_idx=0, frame_idx=0):
        """
        View a specific frame from a scene.
        
        Args:
            scene_idx: Scene index (0-13)
            frame_idx: Frame index within scene
        """
        if scene_idx >= len(self.scenes):
            print(f"Error: Scene {scene_idx} not found")
            return
        
        scene = self.scenes[scene_idx]
        
        if frame_idx >= scene['num_frames']:
            print(f"Error: Frame {frame_idx} not found in {scene['name']}")
            return
        
        # Load color and depth images
        color_file = scene['color_images'][frame_idx]
        depth_file = scene['depth_images'][frame_idx]
        
        print(f"\nViewing: {scene['name']}, Frame {frame_idx}")
        print(f"Color: {color_file.name}")
        print(f"Depth: {depth_file.name}")
        
        color_img = cv2.imread(str(color_file))
        depth_img = cv2.imread(str(depth_file), cv2.IMREAD_ANYDEPTH)
        
        if color_img is None or depth_img is None:
            print("Error loading images")
            return
        
        # Normalize depth for visualization
        depth_vis = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
        
        # Display side by side
        combined = np.hstack([color_img, depth_colored])
        cv2.imshow(f"{scene['name']} - Frame {frame_idx} (Color | Depth)", combined)
        
        print("\nPress any key to continue, ESC to exit viewer...")
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return key != 27  # Return False if ESC pressed
    
    def browse_dataset(self):
        """Interactive dataset browser."""
        print("\n" + "=" * 70)
        print("INTERACTIVE DATASET BROWSER")
        print("=" * 70)
        print("Controls:")
        print("  → (Right Arrow) - Next frame")
        print("  ← (Left Arrow)  - Previous frame")
        print("  ↑ (Up Arrow)    - Next scene")
        print("  ↓ (Down Arrow)  - Previous scene")
        print("  ESC             - Exit")
        print("  S               - Save current frame info")
        print("=" * 70)
        
        scene_idx = 0
        frame_idx = 0
        
        while True:
            scene = self.scenes[scene_idx]
            
            # Load images
            color_file = scene['color_images'][frame_idx]
            depth_file = scene['depth_images'][frame_idx]
            
            color_img = cv2.imread(str(color_file))
            depth_img = cv2.imread(str(depth_file), cv2.IMREAD_ANYDEPTH)
            
            # Visualize depth
            depth_vis = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
            
            # Add info overlay
            info_text = [
                f"Scene: {scene['name']} ({scene_idx+1}/{len(self.scenes)})",
                f"Frame: {frame_idx+1}/{scene['num_frames']}",
                f"File: {color_file.name}",
            ]
            
            display = color_img.copy()
            for i, text in enumerate(info_text):
                cv2.putText(display, text, (10, 30 + i*30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Combine
            combined = np.hstack([display, depth_colored])
            cv2.imshow("RGB-D Dataset Browser (↑↓ scenes | ←→ frames | ESC quit)", combined)
            
            # Handle input
            key = cv2.waitKey(0) & 0xFF
            
            if key == 27:  # ESC
                break
            elif key == 83 or key == 81:  # Right arrow (next frame)
                frame_idx = (frame_idx + 1) % scene['num_frames']
            elif key == 82 or key == 84:  # Left arrow (previous frame)
                frame_idx = (frame_idx - 1) % scene['num_frames']
            elif key == 80:  # Up arrow (next scene)
                scene_idx = (scene_idx + 1) % len(self.scenes)
                frame_idx = 0
            elif key == 81:  # Down arrow (previous scene)
                scene_idx = (scene_idx - 1) % len(self.scenes)
                frame_idx = 0
            elif key == ord('s') or key == ord('S'):
                print(f"\n📸 Current frame info:")
                print(f"   Scene: {scene['name']}")
                print(f"   Frame: {frame_idx}")
                print(f"   Color: {color_file}")
                print(f"   Depth: {depth_file}")
        
        cv2.destroyAllWindows()
        print("\n>> Browser closed")
    
    def extract_objects(self, output_dir=OUTPUT_DIR):
        """
        Extract object regions from scenes for training.
        Creates a dataset suitable for box detection training.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        (output_path / 'images').mkdir(exist_ok=True)
        (output_path / 'depth').mkdir(exist_ok=True)
        (output_path / 'labels').mkdir(exist_ok=True)
        
        print("\n" + "=" * 70)
        print("EXTRACTING OBJECT REGIONS")
        print("=" * 70)
        
        total_extracted = 0
        
        for scene in self.scenes:
            print(f"\nProcessing {scene['name']}...")
            
            # Sample frames (every 10th frame to avoid redundancy)
            sampled_frames = range(0, scene['num_frames'], 10)
            
            for frame_idx in sampled_frames:
                color_file = scene['color_images'][frame_idx]
                depth_file = scene['depth_images'][frame_idx]
                
                color_img = cv2.imread(str(color_file))
                depth_img = cv2.imread(str(depth_file), cv2.IMREAD_ANYDEPTH)
                
                if color_img is None or depth_img is None:
                    continue
                
                # Save as training sample
                sample_id = f"{scene['number']}_{frame_idx:05d}"
                
                cv2.imwrite(str(output_path / 'images' / f'{sample_id}.png'), color_img)
                cv2.imwrite(str(output_path / 'depth' / f'{sample_id}.png'), depth_img)
                
                # Create dummy label file (would need manual annotation for real training)
                # Format: class_id x_center y_center width height (normalized)
                label_path = output_path / 'labels' / f'{sample_id}.txt'
                with open(label_path, 'w') as f:
                    f.write("# Placeholder - needs manual annotation\n")
                
                total_extracted += 1
        
        print(f"\n>> Extracted {total_extracted} training samples to {output_path}")
        
        # Create dataset info file
        info = {
            'dataset': 'RGB-D Scenes v2',
            'source': str(self.imgs_path),
            'num_scenes': len(self.scenes),
            'num_samples': total_extracted,
            'image_size': '640x480',
            'format': 'PNG (RGB + 16-bit depth)'
        }
        
        with open(output_path / 'dataset_info.json', 'w') as f:
            json.dump(info, f, indent=2)
        
        print(f">> Dataset info saved to {output_path / 'dataset_info.json'}")
        
        return output_path

# ==============================================================================
# MAIN APPLICATION
# ==============================================================================

def main():
    """Main dataset exploration application."""
    
    print("=" * 70)
    print("RGB-D DATASET EXPLORER")
    print("=" * 70)
    print()
    
    # Initialize explorer
    explorer = RGBDDatasetExplorer(IMGS_PATH, PC_PATH)
    
    if len(explorer.scenes) == 0:
        print("⚠️  No scenes found!")
        print(f"Make sure dataset is at: {IMGS_PATH}")
        return
    
    # Print summary
    explorer.print_summary()
    
    # Interactive menu
    while True:
        print("\nWhat would you like to do?")
        print("  1. Browse dataset interactively")
        print("  2. View specific scene/frame")
        print("  3. Extract training samples")
        print("  4. View dataset statistics")
        print("  5. Exit")
        
        choice = input("\nEnter choice (1-5): ").strip()
        
        if choice == '1':
            explorer.browse_dataset()
        
        elif choice == '2':
            scene_idx = int(input(f"Enter scene index (0-{len(explorer.scenes)-1}): "))
            frame_idx = int(input("Enter frame index: "))
            explorer.view_scene(scene_idx, frame_idx)
        
        elif choice == '3':
            output_dir = input(f"Output directory [{OUTPUT_DIR}]: ").strip()
            if not output_dir:
                output_dir = OUTPUT_DIR
            explorer.extract_objects(output_dir)
        
        elif choice == '4':
            explorer.print_summary()
        
        elif choice == '5':
            print("\n>> Goodbye!")
            break
        
        else:
            print("Invalid choice, please try again")

if __name__ == "__main__":
    main()

