"""
Interactive Placement GUI Application
=====================================

Step-by-step workflow for package placement analysis:
1. Point at box → measure dimensions
2. Point at desk → scan surface
3. Analyze feasibility (animated)
4. Show results with visualization

Usage:
    python placement_gui.py
    
Controls:
    SPACE - Capture/Next step
    B - Go back
    R - Reset
    S - Save screenshot
    M - Toggle detection method
    ESC - Quit
"""

import sys
import os
from pathlib import Path

# Add src directory to path so we can import modules
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

from pykinect2 import PyKinectRuntime, PyKinectV2
import numpy as np
import cv2
import ctypes
import time
from datetime import datetime
from enum import Enum
from typing import Optional, Tuple, List

# Import our modules
from core.box_detector import BoxDetector, DetectedBox
from core.desk_monitor import DeskSpaceAnalyzer, ransac_plane, plane_axes
from core.placement_feasibility import (
    PlacementFeasibilityAnalyzer, BoxDimensions, FreeRegion, PlacementResult
)
from core.visualization_effects import VisualEffects


class AppState(Enum):
    """Application workflow states."""
    BOX_MEASUREMENT = 1
    DESK_SCANNING = 2
    ANALYSIS = 3
    RESULTS = 4


class PlacementGUI:
    """Main GUI application."""
    
    def __init__(self):
        """Initialize the application."""
        print("=" * 70)
        print("SPO-T: Adaptive Package Placement System")
        print("=" * 70)
        print("Initializing Kinect...")
        
        # Initialize Kinect
        self.kinect = PyKinectRuntime.PyKinectRuntime(
            PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Depth
        )
        
        # Initialize components
        self.box_detector = BoxDetector(use_ml_model=False)
        self.desk_analyzer = DeskSpaceAnalyzer()
        self.feasibility_analyzer = PlacementFeasibilityAnalyzer()
        self.visual_effects = VisualEffects()
        
        # State management
        self.state = AppState.BOX_MEASUREMENT
        self.detected_box: Optional[DetectedBox] = None
        self.box_dimensions: Optional[BoxDimensions] = None
        self.free_regions: List[FreeRegion] = []
        self.placement_result: Optional[PlacementResult] = None
        self.analysis_start_time: Optional[float] = None
        self.desk_plane_info: Optional[Tuple] = None  # (normal, center, u_axis, v_axis, u_bins, v_bins, grid)
        
        # Display settings
        self.window_name = "SPO-T Placement System"
        self.display_width = 1280
        self.display_height = 720
        
        # Detection method
        self.use_yolo = False
        
        # Frame buffers
        self.color_frame = None
        self.depth_frame = None
        self.cam_3d = None
        
        print("✓ Initialization complete!")
        print("\n" + "=" * 70)
        print("CONTROLS:")
        print("  SPACE - Capture/Next step")
        print("  B     - Go back")
        print("  R     - Reset to beginning")
        print("  S     - Save screenshot")
        print("  M     - Toggle detection method (CV/YOLO)")
        print("  ESC   - Quit")
        print("=" * 70 + "\n")
        
        self._show_welcome_message()
    
    def _show_welcome_message(self):
        """Show initial instructions."""
        print(f"\n{'='*70}")
        print("STEP 1: BOX MEASUREMENT")
        print("="*70)
        print("→ Point the camera at the box you want to place")
        print("→ Make sure the box is clearly visible")
        print("→ Press SPACE when ready to capture dimensions")
        print("="*70 + "\n")
    
    def run(self):
        """Main application loop."""
        while True:
            # Get frames from Kinect
            self._update_frames()
            
            # Process current state
            if self.state == AppState.BOX_MEASUREMENT:
                display = self._process_box_measurement()
            elif self.state == AppState.DESK_SCANNING:
                display = self._process_desk_scanning()
            elif self.state == AppState.ANALYSIS:
                display = self._process_analysis()
            elif self.state == AppState.RESULTS:
                display = self._process_results()
            else:
                display = self.color_frame
            
            # Add UI overlay
            display = self._add_ui_overlay(display)
            
            # Show display
            cv2.imshow(self.window_name, display)
            
            # Handle input
            key = cv2.waitKey(1) & 0xFF
            if not self._handle_input(key):
                break
        
        # Cleanup
        cv2.destroyAllWindows()
        print("\n✓ Application closed")
    
    def _update_frames(self):
        """Update frames from Kinect."""
        if self.kinect.has_new_color_frame():
            frame = self.kinect.get_last_color_frame()
            c = frame.reshape((1080, 1920, 4)).astype(np.uint8)
            self.color_frame = c[..., :3][:, :, ::-1]  # RGBA to BGR
            
            # Resize for display
            self.color_frame = cv2.resize(self.color_frame, 
                                         (self.display_width, self.display_height))
        
        if self.kinect.has_new_depth_frame():
            self.depth_frame = self.kinect.get_last_depth_frame().reshape((424, 512)).astype(np.uint16)
            
            # Convert to 3D
            self.cam_3d = self._depth_to_3d(self.depth_frame)
    
    def _depth_to_3d(self, depth: np.ndarray) -> np.ndarray:
        """Convert depth frame to 3D point cloud."""
        depth_flat = depth.flatten().astype(np.uint16)
        CSP = np.zeros((depth_flat.shape[0],), dtype=PyKinectV2._CameraSpacePoint)
        
        depth_ptr = ctypes.cast(depth_flat.ctypes.data, ctypes.POINTER(ctypes.c_ushort))
        csp_ptr = ctypes.cast(CSP.ctypes.data, ctypes.POINTER(PyKinectV2._CameraSpacePoint))
        
        self.kinect._mapper.MapDepthFrameToCameraSpace(
            ctypes.c_uint(depth_flat.shape[0]),
            depth_ptr,
            ctypes.c_uint(CSP.shape[0]),
            csp_ptr
        )
        
        cam = np.frombuffer(CSP, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4")])
        cam_3d = np.column_stack([cam["x"], cam["y"], cam["z"]]).reshape(424, 512, 3)
        cam_3d[np.isinf(cam_3d)] = np.nan
        
        return cam_3d
    
    def _process_box_measurement(self) -> np.ndarray:
        """Process Step 1: Box Measurement."""
        if self.color_frame is None or self.cam_3d is None:
            return np.zeros((self.display_height, self.display_width, 3), dtype=np.uint8)
        
        display = self.color_frame.copy()
        
        # Detect boxes
        boxes = self.box_detector.detect(display, self.depth_frame, self.kinect._mapper)
        
        # Visualize detections
        for box in boxes:
            x, y, w, h = box.bbox_2d
            
            # Scale coordinates from depth resolution to display resolution
            scale_x = self.display_width / 1920
            scale_y = self.display_height / 1080
            x = int(x * scale_x)
            y = int(y * scale_y)
            w = int(w * scale_x)
            h = int(h * scale_y)
            
            # Draw box
            cv2.rectangle(display, (x, y), (x + w, y + h), (255, 0, 255), 2)
            
            # Draw dimensions
            dims_text = f"{box.dimensions_3d[0]*100:.1f}x{box.dimensions_3d[1]*100:.1f}x{box.dimensions_3d[2]*100:.1f}cm"
            cv2.putText(display, dims_text, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        
        # Store largest box
        if boxes:
            self.detected_box = max(boxes, key=lambda b: b.volume_m3)
        
        return display
    
    def _process_desk_scanning(self) -> np.ndarray:
        """Process Step 2: Desk Scanning."""
        if self.color_frame is None or self.cam_3d is None:
            return np.zeros((self.display_height, self.display_width, 3), dtype=np.uint8)
        
        display = self.color_frame.copy()
        
        # Detect desk plane
        flat = self.cam_3d.reshape(-1, 3)
        valid = np.isfinite(flat).all(axis=1)
        P = flat[valid]
        
        if P.shape[0] > 2000:
            n, d0, inliers, ctr = ransac_plane(P)
            
            if n is not None:
                # Calculate plane angle (deviation from horizontal)
                horizontal = np.array([0, 1, 0], dtype=np.float32)
                angle = np.degrees(np.arccos(np.clip(np.dot(n, horizontal), -1.0, 1.0)))
                
                # Show angle indicator
                display = self.visual_effects.draw_angle_indicator(
                    display, (self.display_width - 100, 100), angle - 90, threshold=15
                )
                
                # If angle is good, show free space overlay
                if abs(angle - 90) <= 15:
                    # Transform to plane coordinates
                    u_axis, v_axis = plane_axes(n)
                    vec = flat - ctr
                    uvh = np.full((flat.shape[0], 3), np.nan, np.float32)
                    uvh[valid, 0] = vec[valid] @ u_axis
                    uvh[valid, 1] = vec[valid] @ v_axis
                    uvh[valid, 2] = vec[valid] @ n
                    UVH = uvh.reshape(424, 512, 3)
                    
                    # Create occupancy grid
                    plane_mask = np.abs(UVH[..., 2]) < 0.010
                    
                    if np.count_nonzero(plane_mask) > 500:
                        u_vals = UVH[..., 0][plane_mask]
                        v_vals = UVH[..., 1][plane_mask]
                        u_min, u_max = np.nanmin(u_vals), np.nanmax(u_vals)
                        v_min, v_max = np.nanmin(v_vals), np.nanmax(v_vals)
                        
                        u_bins = np.arange(u_min, u_max, 0.005)
                        v_bins = np.arange(v_min, v_max, 0.005)
                        
                        if len(u_bins) > 0 and len(v_bins) > 0:
                            grid = np.zeros((len(v_bins), len(u_bins)), np.uint8)
                            
                            H = UVH[..., 2]
                            U = UVH[..., 0]
                            V = UVH[..., 1]
                            
                            occ = (H >= 0.005) & (H <= 0.08) & np.isfinite(H)
                            
                            if np.count_nonzero(occ) > 0:
                                ui = np.clip(((U[occ] - u_min) / 0.005).astype(int), 0, len(u_bins) - 1)
                                vi = np.clip(((V[occ] - v_min) / 0.005).astype(int), 0, len(v_bins) - 1)
                                grid[vi, ui] = 1
                            
                            # Clean grid
                            free = (grid == 0).astype(np.uint8) * 255
                            free = cv2.morphologyEx(free, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
                            free = cv2.morphologyEx(free, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
                            
                            # Draw free space overlay
                            display = self._draw_free_space_overlay(
                                display, free, u_bins, v_bins, ctr, u_axis, v_axis
                            )
                            
                            # Store desk info for analysis
                            self.desk_plane_info = (n, ctr, u_axis, v_axis, u_bins, v_bins, grid)
        
        return display
    
    def _draw_free_space_overlay(self, image, free_mask, u_bins, v_bins, ctr, u_axis, v_axis):
        """Draw green overlay for free space regions."""
        overlay = image.copy()
        
        # Sample free space points
        for v_idx in range(0, len(v_bins), 2):
            for u_idx in range(0, len(u_bins), 2):
                if free_mask[v_idx, u_idx] > 0:
                    u_val = u_bins[u_idx]
                    v_val = v_bins[v_idx]
                    pos_3d = ctr + u_val * u_axis + v_val * v_axis
                    
                    # Project to color space
                    csp = PyKinectV2._CameraSpacePoint(pos_3d[0], pos_3d[1], pos_3d[2])
                    col_pt = self.kinect._mapper.MapCameraPointToColorSpace(csp)
                    
                    if np.isfinite(col_pt.x) and np.isfinite(col_pt.y):
                        x = int(col_pt.x * self.display_width / 1920)
                        y = int(col_pt.y * self.display_height / 1080)
                        
                        if 0 <= x < self.display_width and 0 <= y < self.display_height:
                            cv2.circle(overlay, (x, y), 3, (0, 255, 0), -1)
        
        # Blend
        result = cv2.addWeighted(image, 0.6, overlay, 0.4, 0)
        return result
    
    def _process_analysis(self) -> np.ndarray:
        """Process Step 3: Animated Analysis."""
        if self.analysis_start_time is None:
            self.analysis_start_time = time.time()
            
            # Run actual analysis
            if self.box_dimensions and self.desk_plane_info:
                n, ctr, u_axis, v_axis, u_bins, v_bins, grid = self.desk_plane_info
                
                # Find free regions
                free_mask = (grid == 0).astype(np.uint8) * 255
                free_mask = cv2.morphologyEx(free_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
                free_mask = cv2.morphologyEx(free_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
                
                # Find connected components
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                    (free_mask > 0).astype(np.uint8), connectivity=8
                )
                
                self.free_regions = []
                for i in range(1, num_labels):
                    area_cells = stats[i, cv2.CC_STAT_AREA]
                    area_m2 = area_cells * 0.005 * 0.005
                    
                    if area_m2 < 0.001:
                        continue
                    
                    cx, cy = centroids[i]
                    cx_int = int(np.clip(cx, 0, len(u_bins) - 1))
                    cy_int = int(np.clip(cy, 0, len(v_bins) - 1))
                    
                    region = FreeRegion(
                        id=i,
                        area_m2=area_m2,
                        centroid_uv=(u_bins[cx_int], v_bins[cy_int]),
                        centroid_grid=(cx, cy),
                        bbox_grid=(stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                                  stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]),
                        mask=(labels == i).astype(np.uint8)
                    )
                    self.free_regions.append(region)
                
                # Run feasibility analysis
                self.placement_result = self.feasibility_analyzer.analyze(
                    self.box_dimensions,
                    self.free_regions,
                    grid,
                    u_bins,
                    v_bins
                )
        
        # Show animation for 2 seconds
        elapsed = time.time() - self.analysis_start_time
        
        if elapsed > 2.0:
            # Move to results
            self.state = AppState.RESULTS
            self._show_results_message()
            return self.color_frame.copy()
        
        # Animated progress
        display = self.color_frame.copy()
        progress = elapsed / 2.0
        
        # Progress bar
        display = self.visual_effects.draw_progress_bar(
            display,
            (self.display_width // 2 - 200, self.display_height // 2),
            400,
            progress,
            "Analyzing placement options..."
        )
        
        # Spinner
        display = self.visual_effects.draw_status_indicator(
            display,
            (self.display_width // 2, self.display_height // 2 - 80),
            'processing',
            icon_size=60
        )
        
        return display
    
    def _process_results(self) -> np.ndarray:
        """Process Step 4: Results Display."""
        if self.color_frame is None or self.placement_result is None:
            return np.zeros((self.display_height, self.display_width, 3), dtype=np.uint8)
        
        display = self.color_frame.copy()
        
        if self.placement_result.feasible and self.placement_result.best_candidate:
            # SUCCESS - Draw placement visualization
            candidate = self.placement_result.best_candidate
            
            # Get 3D position
            if self.desk_plane_info:
                n, ctr, u_axis, v_axis, u_bins, v_bins, grid = self.desk_plane_info
                
                u_pos, v_pos = candidate.position_uv
                pos_3d = ctr + u_pos * u_axis + v_pos * v_axis
                
                # Project to color space
                csp = PyKinectV2._CameraSpacePoint(pos_3d[0], pos_3d[1], pos_3d[2])
                col_pt = self.kinect._mapper.MapCameraPointToColorSpace(csp)
                
                x = int(col_pt.x * self.display_width / 1920)
                y = int(col_pt.y * self.display_height / 1080)
                
                if 0 <= x < self.display_width and 0 <= y < self.display_height:
                    # Draw 3D box projection
                    box_w = int(self.box_dimensions.length * 200)  # Scale for visualization
                    box_h = int(self.box_dimensions.width * 200)
                    
                    display = self.visual_effects.draw_3d_box_projection(
                        display, (x, y), box_w, box_h, depth_offset=30,
                        color=(0, 255, 0), alpha=0.7
                    )
                    
                    # Draw clearance arrows
                    display = self.visual_effects.draw_clearance_arrows(
                        display, (x, y), (box_w, box_h),
                        candidate.clearance, scale=200, color=(255, 255, 0)
                    )
                    
                    # Success checkmark
                    display = self.visual_effects.draw_status_indicator(
                        display, (x, y - 150), 'success', icon_size=60
                    )
        else:
            # FAILURE - Show X and reason
            display = self.visual_effects.draw_status_indicator(
                display,
                (self.display_width // 2, self.display_height // 2),
                'fail',
                icon_size=80
            )
        
        return display
    
    def _add_ui_overlay(self, image: np.ndarray) -> np.ndarray:
        """Add UI overlay with state info and instructions."""
        display = image.copy()
        
        # Top bar
        cv2.rectangle(display, (0, 0), (self.display_width, 60), (40, 40, 40), -1)
        cv2.rectangle(display, (0, 0), (self.display_width, 60), (255, 255, 255), 2)
        
        # State indicator
        state_text = self._get_state_text()
        cv2.putText(display, state_text, (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        # Detection method
        method = "YOLO" if self.use_yolo else "Traditional CV"
        cv2.putText(display, f"Detection: {method}", (self.display_width - 250, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Bottom instructions
        instructions = self._get_instructions()
        cv2.rectangle(display, (0, self.display_height - 50), 
                     (self.display_width, self.display_height), (40, 40, 40), -1)
        cv2.putText(display, instructions, (20, self.display_height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Results info panel (if in results state)
        if self.state == AppState.RESULTS and self.placement_result:
            panel = self._create_results_panel()
            # Overlay panel on right side
            h, w = panel.shape[:2]
            x = self.display_width - w - 20
            y = 80
            if x > 0 and y > 0 and y + h < self.display_height:
                display[y:y+h, x:x+w] = cv2.addWeighted(display[y:y+h, x:x+w], 0.3, panel, 0.7, 0)
        
        return display
    
    def _get_state_text(self) -> str:
        """Get text for current state."""
        if self.state == AppState.BOX_MEASUREMENT:
            return "[1/4] Box Measurement"
        elif self.state == AppState.DESK_SCANNING:
            return "[2/4] Desk Scanning"
        elif self.state == AppState.ANALYSIS:
            return "[3/4] Analysis"
        elif self.state == AppState.RESULTS:
            return "[4/4] Results"
        return ""
    
    def _get_instructions(self) -> str:
        """Get instructions for current state."""
        if self.state == AppState.BOX_MEASUREMENT:
            return "Point at box → SPACE to capture | B-Back | R-Reset | ESC-Quit"
        elif self.state == AppState.DESK_SCANNING:
            return "Point at desk → SPACE to scan | B-Back | R-Reset | ESC-Quit"
        elif self.state == AppState.ANALYSIS:
            return "Analyzing..."
        elif self.state == AppState.RESULTS:
            return "SPACE-Continue | B-Back | R-Restart | S-Save | ESC-Quit"
        return ""
    
    def _create_results_panel(self) -> np.ndarray:
        """Create info panel for results."""
        if not self.placement_result:
            return np.zeros((100, 300, 3), dtype=np.uint8)
        
        lines = []
        
        if self.placement_result.feasible:
            lines.append("✓ PLACEMENT FEASIBLE")
            lines.append("")
            if self.placement_result.best_candidate:
                c = self.placement_result.best_candidate
                lines.append(f"Score: {c.score*100:.0f}%")
                lines.append(f"Orientation: {c.orientation:.0f}°")
                lines.append("")
                lines.append("Clearances:")
                lines.append(f"  Front: {c.clearance.get('front', 0)*100:.1f}cm")
                lines.append(f"  Back:  {c.clearance.get('back', 0)*100:.1f}cm")
                lines.append(f"  Left:  {c.clearance.get('left', 0)*100:.1f}cm")
                lines.append(f"  Right: {c.clearance.get('right', 0)*100:.1f}cm")
        else:
            lines.append("✗ NOT FEASIBLE")
            lines.append("")
            lines.append(f"Reason:")
            lines.append(f"  {self.placement_result.reason}")
        
        lines.append("")
        lines.append(f"Desk Free: {self.placement_result.desk_free_percentage:.1f}%")
        
        panel = self.visual_effects.create_info_panel(
            350, len(lines) * 25 + 60, "Placement Analysis", lines
        )
        
        return panel
    
    def _handle_input(self, key: int) -> bool:
        """
        Handle keyboard input.
        
        Returns:
            True to continue, False to quit
        """
        if key == 27:  # ESC
            return False
        
        elif key == ord(' '):  # SPACE - Next/Capture
            self._handle_space()
        
        elif key == ord('b') or key == ord('B'):  # Back
            self._handle_back()
        
        elif key == ord('r') or key == ord('R'):  # Reset
            self._handle_reset()
        
        elif key == ord('s') or key == ord('S'):  # Save
            self._save_screenshot()
        
        elif key == ord('m') or key == ord('M'):  # Toggle method
            self._toggle_detection_method()
        
        return True
    
    def _handle_space(self):
        """Handle space key (capture/next)."""
        if self.state == AppState.BOX_MEASUREMENT:
            if self.detected_box:
                # Capture box dimensions
                self.box_dimensions = BoxDimensions(
                    self.detected_box.dimensions_3d[0],
                    self.detected_box.dimensions_3d[1],
                    self.detected_box.dimensions_3d[2]
                )
                
                print(f"\n✓ Box captured: {self.box_dimensions.length*100:.1f} x "
                      f"{self.box_dimensions.width*100:.1f} x {self.box_dimensions.height*100:.1f} cm")
                
                # Move to desk scanning
                self.state = AppState.DESK_SCANNING
                self._show_desk_scanning_message()
            else:
                print("⚠ No box detected. Ensure box is visible and try again.")
        
        elif self.state == AppState.DESK_SCANNING:
            if self.desk_plane_info:
                print("\n✓ Desk scanned")
                # Move to analysis
                self.state = AppState.ANALYSIS
                self.analysis_start_time = None
                print("\n" + "="*70)
                print("STEP 3: ANALYZING")
                print("="*70)
            else:
                print("⚠ No desk surface detected. Adjust camera angle and try again.")
        
        elif self.state == AppState.RESULTS:
            # Start over
            self._handle_reset()
    
    def _handle_back(self):
        """Go back one step."""
        if self.state == AppState.DESK_SCANNING:
            self.state = AppState.BOX_MEASUREMENT
            self.detected_box = None
            self.box_dimensions = None
            print("\n← Back to Box Measurement")
            self._show_welcome_message()
        
        elif self.state == AppState.RESULTS:
            self.state = AppState.DESK_SCANNING
            self.placement_result = None
            self.analysis_start_time = None
            print("\n← Back to Desk Scanning")
            self._show_desk_scanning_message()
    
    def _handle_reset(self):
        """Reset to beginning."""
        self.state = AppState.BOX_MEASUREMENT
        self.detected_box = None
        self.box_dimensions = None
        self.free_regions = []
        self.placement_result = None
        self.analysis_start_time = None
        self.desk_plane_info = None
        
        print("\n" + "="*70)
        print("↻ RESET TO BEGINNING")
        print("="*70)
        self._show_welcome_message()
    
    def _save_screenshot(self):
        """Save current view."""
        if self.color_frame is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create data/captures directory if it doesn't exist
            captures_dir = Path("data/captures")
            captures_dir.mkdir(parents=True, exist_ok=True)
            
            filename = captures_dir / f"screenshot_{timestamp}.png"
            cv2.imwrite(str(filename), self.color_frame)
            print(f"✓ Screenshot saved: {filename}")
    
    def _toggle_detection_method(self):
        """Toggle between Traditional CV and YOLO."""
        self.use_yolo = not self.use_yolo
        
        # Update detector
        if self.use_yolo:
            model_path = "models/runs/detect/box_detection/weights/best.pt"
            if Path(model_path).exists():
                self.box_detector = BoxDetector(use_ml_model=True, model_path=model_path)
                print("✓ Switched to YOLO detection")
            else:
                print("⚠ YOLO model not found. Train model first or use Traditional CV.")
                print(f"   Expected location: {model_path}")
                self.use_yolo = False
        else:
            self.box_detector = BoxDetector(use_ml_model=False)
            print("✓ Switched to Traditional CV detection")
    
    def _show_desk_scanning_message(self):
        """Show desk scanning instructions."""
        print(f"\n{'='*70}")
        print("STEP 2: DESK SCANNING")
        print("="*70)
        print("→ Point the camera at the desk surface")
        print("→ Keep camera roughly horizontal (angle indicator will show)")
        print("→ Free space will be shown in green")
        print("→ Press SPACE when ready to analyze")
        print("="*70 + "\n")
    
    def _show_results_message(self):
        """Show results message."""
        print("\n" + "="*70)
        print("STEP 4: RESULTS")
        print("="*70)
        if self.placement_result and self.placement_result.feasible:
            print("✓ PLACEMENT IS FEASIBLE!")
            print(f"  Found {len(self.placement_result.all_candidates)} valid location(s)")
            if self.placement_result.best_candidate:
                c = self.placement_result.best_candidate
                print(f"  Best position score: {c.score*100:.0f}%")
                print(f"  Clearances:")
                print(f"    Front: {c.clearance.get('front', 0)*100:.1f}cm")
                print(f"    Back:  {c.clearance.get('back', 0)*100:.1f}cm")
                print(f"    Left:  {c.clearance.get('left', 0)*100:.1f}cm")
                print(f"    Right: {c.clearance.get('right', 0)*100:.1f}cm")
        else:
            print("✗ PLACEMENT NOT FEASIBLE")
            if self.placement_result:
                print(f"  Reason: {self.placement_result.reason}")
        print("="*70 + "\n")


def main():
    """Entry point."""
    try:
        app = PlacementGUI()
        app.run()
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

