"""
Desk Space Monitor
==================

Real-time monitoring and analysis of available desk space using Kinect v2.

Features:
- Continuous free space percentage tracking
- Multiple free region detection and ranking
- Visual overlay showing free vs. occupied areas
- Clutter alerts when desk gets too full
- Time-series logging of desk occupancy
- Usage heatmap showing frequently occupied areas
- Export free space data (CSV, JSON)

Usage:
    python desk_monitor.py

Controls:
    ESC - Quit
    S   - Save current snapshot and data
    R   - Reset usage heatmap
    H   - Toggle heatmap view

Output Files:
- desk_occupancy_log.csv: Time-series data of free space percentage
- free_space_regions.json: Current free space regions as polygons
- usage_heatmap.png: Heatmap of desk usage over time
"""

from pykinect2 import PyKinectRuntime, PyKinectV2
import numpy as np
import cv2
import ctypes
import time
import json
import csv
from datetime import datetime
from collections import deque

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Desk surface detection
PLANE_INLIER_THRESH = 0.010  # meters (±10mm around plane)
RANSAC_ITERS = 400

# Obstacle detection
H_MIN = 0.005  # Minimum obstacle height (5mm)
H_MAX = 0.08   # Maximum obstacle height (80mm) - ignore tall objects like walls

# Free space analysis
GRID_RES = 0.005  # Grid resolution (5mm cells)
MIN_REGION_AREA = 0.001  # Minimum free region area (m²)
CLUTTER_THRESHOLD = 70  # Alert if >70% of desk is occupied

# Visualization
OVERLAY_ALPHA = 0.6  # Transparency for free space overlay (higher = more visible)
FRAME_BUFFER_SIZE = 100  # Number of frames for smoothing
FREE_SPACE_DOT_SIZE = 5  # Size of free space dots
FREE_SPACE_SUBSAMPLE = 1  # Subsample factor (1 = all points, 2 = every other point)

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def ransac_plane(P, iters=RANSAC_ITERS, thresh=PLANE_INLIER_THRESH):
    """
    RANSAC plane fitting for desk surface detection.
    Returns: (normal, offset, inlier_mask, center)
    """
    rng = np.random.default_rng(1234)
    N = P.shape[0]
    best = (None, None, None)
    best_count = -1
    
    for _ in range(iters):
        idx = rng.choice(N, 3, replace=False)
        p0, p1, p2 = P[idx]
        n = np.cross(p1-p0, p2-p0)
        nn = np.linalg.norm(n)
        if nn < 1e-6:
            continue
        n /= nn
        d = -np.dot(n, p0)
        dist = np.abs(P @ n + d)
        inliers = dist < thresh
        c = inliers.sum()
        if c > best_count:
            best = (n, d, inliers)
            best_count = c
    
    if best[0] is None:
        return None, None, None, None
    
    # Refine with PCA
    Pin = P[best[2]]
    ctr = Pin.mean(axis=0)
    _, _, Vt = np.linalg.svd(Pin - ctr, full_matrices=False)
    n = Vt[-1]
    n /= np.linalg.norm(n)
    d = -np.dot(n, ctr)
    
    # Ensure upward normal
    if np.dot(n, np.array([0,1,0], dtype=np.float32)) < 0:
        n = -n
        d = -d
    
    return n, d, best[2], ctr

def plane_axes(n):
    """Create 2D coordinate system on the plane."""
    zf = np.array([0,0,1], np.float32)
    u = np.cross(n, zf)
    if np.linalg.norm(u) < 1e-6:
        u = np.cross(n, np.array([1,0,0], np.float32))
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    v /= np.linalg.norm(v)
    return u, v


# DESK SPACE ANALYZER CLASS


class DeskSpaceAnalyzer:
    """Analyzes and tracks desk space usage over time."""
    
    def __init__(self):
        self.occupancy_history = []  # [(timestamp, free_pct, free_area_m2), ...]
        self.usage_heatmap = None  # Accumulated occupancy over time
        self.recent_free_pct = deque(maxlen=FRAME_BUFFER_SIZE)  # Smoothing buffer
        self.frame_count = 0
        self.start_time = time.time()
        self.last_alert_time = 0
        
    def update(self, grid, free_mask, grid_res):
        """
        Update analyzer with new frame data.
        
        Args:
            grid: Occupancy grid (1=occupied, 0=free)
            free_mask: Binary mask of free space after cleaning
            grid_res: Grid resolution in meters
        """
        self.frame_count += 1
        
        # Initialize heatmap on first frame
        if self.usage_heatmap is None:
            self.usage_heatmap = np.zeros_like(grid, dtype=np.float32)
        
        # Accumulate occupancy in heatmap
        # Handle variable grid sizes by only updating the overlapping region
        if self.usage_heatmap.shape == grid.shape:
            self.usage_heatmap += grid.astype(np.float32)
        else:
            # Grid size changed - resize heatmap or create intersection
            min_h = min(self.usage_heatmap.shape[0], grid.shape[0])
            min_w = min(self.usage_heatmap.shape[1], grid.shape[1])
            
            # Create new heatmap with current grid size
            new_heatmap = np.zeros_like(grid, dtype=np.float32)
            
            # Copy over the overlapping region from old heatmap
            new_heatmap[:min_h, :min_w] = self.usage_heatmap[:min_h, :min_w]
            
            # Add current grid
            new_heatmap += grid.astype(np.float32)
            
            # Update heatmap
            self.usage_heatmap = new_heatmap
        
        # Calculate free space metrics
        total_cells = grid.size
        free_cells = np.sum(free_mask > 0)
        occupied_cells = total_cells - free_cells
        
        free_percentage = (free_cells / total_cells) * 100
        free_area_m2 = free_cells * grid_res * grid_res
        occupied_area_m2 = occupied_cells * grid_res * grid_res
        
        # Smooth the percentage
        self.recent_free_pct.append(free_percentage)
        smoothed_free_pct = np.mean(self.recent_free_pct)
        
        # Log to history (every 10 frames to reduce data)
        if self.frame_count % 10 == 0:
            timestamp = time.time()
            self.occupancy_history.append((
                timestamp,
                smoothed_free_pct,
                free_area_m2,
                occupied_area_m2
            ))
        
        # Check for clutter alert
        occupied_percentage = 100 - smoothed_free_pct
        is_cluttered = occupied_percentage > CLUTTER_THRESHOLD
        
        # Rate limit alerts (once per 5 seconds)
        current_time = time.time()
        should_alert = is_cluttered and (current_time - self.last_alert_time > 5.0)
        if should_alert:
            self.last_alert_time = current_time
            print(f"⚠️  [ALERT] Desk is {occupied_percentage:.1f}% occupied - consider clearing space!")
        
        return {
            'free_percentage': smoothed_free_pct,
            'free_area_m2': free_area_m2,
            'occupied_area_m2': occupied_area_m2,
            'is_cluttered': is_cluttered,
            'total_area_m2': total_cells * grid_res * grid_res
        }
    
    def find_free_regions(self, free_mask, u_bins, v_bins, grid_res):
        """
        Identify distinct free space regions.
        
        Returns: List of region dictionaries with area, centroid, bounding box
        """
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            (free_mask > 0).astype(np.uint8), connectivity=8
        )
        
        regions = []
        for i in range(1, num_labels):  # Skip background (label 0)
            area_cells = stats[i, cv2.CC_STAT_AREA]
            area_m2 = area_cells * grid_res * grid_res
            
            if area_m2 < MIN_REGION_AREA:
                continue
            
            # Get centroid in grid coordinates
            cx_grid, cy_grid = centroids[i]
            
            # Convert to real-world plane coordinates
            cx_grid_int = int(np.clip(cx_grid, 0, len(u_bins)-1))
            cy_grid_int = int(np.clip(cy_grid, 0, len(v_bins)-1))
            
            u_pos = u_bins[cx_grid_int]
            v_pos = v_bins[cy_grid_int]
            
            # Get bounding box
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            
            regions.append({
                'id': i,
                'area_m2': area_m2,
                'centroid_uv': (u_pos, v_pos),
                'centroid_grid': (cx_grid, cy_grid),
                'bbox_grid': (x, y, w, h)
            })
        
        # Sort by area (largest first)
        regions.sort(key=lambda x: x['area_m2'], reverse=True)
        return regions
    
    def get_usage_heatmap(self):
        """Get normalized usage heatmap for visualization."""
        if self.usage_heatmap is None:
            return None
        normalized = cv2.normalize(self.usage_heatmap, None, 0, 255, cv2.NORM_MINMAX)
        return normalized.astype(np.uint8)
    
    def reset_heatmap(self):
        """Reset usage heatmap."""
        if self.usage_heatmap is not None:
            self.usage_heatmap.fill(0)
        print("[MONITOR] Usage heatmap reset")
    
    def save_data(self, filename_prefix='desk_monitor'):
        """Save all collected data to files."""
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save occupancy history to CSV
        csv_filename = f'{filename_prefix}_{timestamp_str}.csv'
        with open(csv_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'free_percentage', 'free_area_m2', 'occupied_area_m2'])
            writer.writerows(self.occupancy_history)
        print(f"✓ Saved occupancy log: {csv_filename}")
        
        # Save usage heatmap
        if self.usage_heatmap is not None:
            heatmap = self.get_usage_heatmap()
            heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_HOT)
            heatmap_filename = f'{filename_prefix}_heatmap_{timestamp_str}.png'
            cv2.imwrite(heatmap_filename, heatmap_colored)
            print(f"✓ Saved usage heatmap: {heatmap_filename}")
        
        # Print summary statistics
        if len(self.occupancy_history) > 0:
            free_pcts = [h[1] for h in self.occupancy_history]
            print(f"\n📊 Session Statistics:")
            print(f"   Duration: {time.time() - self.start_time:.1f} seconds")
            print(f"   Frames processed: {self.frame_count}")
            print(f"   Avg free space: {np.mean(free_pcts):.1f}%")
            print(f"   Min free space: {np.min(free_pcts):.1f}%")
            print(f"   Max free space: {np.max(free_pcts):.1f}%")

# ==============================================================================
# MAIN MONITORING LOOP
# ==============================================================================

def main():
    """Main desk space monitoring application."""
    
    print("=" * 70)
    print("DESK SPACE MONITOR")
    print("=" * 70)
    print("Initializing Kinect sensor...")
    
    # Initialize Kinect
    k = PyKinectRuntime.PyKinectRuntime(
        PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Depth
    )
    
    # Initialize analyzer
    analyzer = DeskSpaceAnalyzer()
    
    # Visualization state
    show_heatmap = False
    
    print("✓ Kinect initialized")
    print("\nControls:")
    print("  ESC - Quit and save data")
    print("  S   - Save snapshot")
    print("  R   - Reset usage heatmap")
    print("  H   - Toggle heatmap view")
    print("\nMonitoring desk space...\n")
    
    while True:
        got_c = k.has_new_color_frame()
        got_d = k.has_new_depth_frame()
        
        if not (got_c and got_d):
            cv2.waitKey(1)
            continue
        
        # =====================================================================
        # STEP 1: Acquire Frames
        # =====================================================================
        d = k.get_last_depth_frame().reshape((424,512)).astype(np.uint16)
        c = k.get_last_color_frame().reshape((1080,1920,4)).astype(np.uint8)
        color_bgr = c[...,:3][:,:,::-1]
        
        # =====================================================================
        # STEP 2: Convert Depth to 3D Point Cloud
        # =====================================================================
        depth_flat = d.flatten().astype(np.uint16)
        CSP = np.zeros((depth_flat.shape[0],), dtype=PyKinectV2._CameraSpacePoint)
        
        depth_ptr = ctypes.cast(depth_flat.ctypes.data, ctypes.POINTER(ctypes.c_ushort))
        csp_ptr = ctypes.cast(CSP.ctypes.data, ctypes.POINTER(PyKinectV2._CameraSpacePoint))
        
        k._mapper.MapDepthFrameToCameraSpace(
            ctypes.c_uint(depth_flat.shape[0]),
            depth_ptr,
            ctypes.c_uint(CSP.shape[0]),
            csp_ptr
        )
        
        cam = np.frombuffer(CSP, dtype=[("x","<f4"),("y","<f4"),("z","<f4")])
        cam = np.column_stack([cam["x"], cam["y"], cam["z"]]).reshape(424,512,3)
        cam[np.isinf(cam)] = np.nan
        
        # =====================================================================
        # STEP 3: Detect Desk Surface
        # =====================================================================
        flat = cam.reshape(-1,3)
        valid = np.isfinite(flat).all(axis=1)
        P = flat[valid]
        
        if P.shape[0] < 2000:
            cv2.waitKey(1)
            continue
        
        n, d0, inliers, ctr = ransac_plane(P)
        if n is None:
            cv2.waitKey(1)
            continue
        
        # =====================================================================
        # STEP 4: Transform to Plane Coordinates
        # =====================================================================
        u_axis, v_axis = plane_axes(n)
        vec = flat - ctr
        uvh = np.full((flat.shape[0],3), np.nan, np.float32)
        uvh[valid,0] = vec[valid] @ u_axis
        uvh[valid,1] = vec[valid] @ v_axis
        uvh[valid,2] = vec[valid] @ n
        UVH = uvh.reshape(424,512,3)
        
        # =====================================================================
        # STEP 5: Create Occupancy Grid
        # =====================================================================
        plane_mask = np.abs(UVH[...,2]) < PLANE_INLIER_THRESH
        
        if np.count_nonzero(plane_mask) < 500:
            cv2.waitKey(1)
            continue
        
        u_vals = UVH[...,0][plane_mask]
        v_vals = UVH[...,1][plane_mask]
        u_min, u_max = np.nanmin(u_vals), np.nanmax(u_vals)
        v_min, v_max = np.nanmin(v_vals), np.nanmax(v_vals)
        
        u_bins = np.arange(u_min, u_max, GRID_RES)
        v_bins = np.arange(v_min, v_max, GRID_RES)
        
        if len(u_bins) == 0 or len(v_bins) == 0:
            cv2.waitKey(1)
            continue
        
        grid = np.zeros((len(v_bins), len(u_bins)), np.uint8)
        
        H = UVH[...,2]
        U = UVH[...,0]
        V = UVH[...,1]
        
        occ = (H >= H_MIN) & (H <= H_MAX) & np.isfinite(H)
        
        if np.count_nonzero(occ) > 0:
            ui = np.clip(((U[occ] - u_min)/GRID_RES).astype(int), 0, len(u_bins)-1)
            vi = np.clip(((V[occ] - v_min)/GRID_RES).astype(int), 0, len(v_bins)-1)
            grid[vi, ui] = 1
        
        # Clean up grid
        free = (grid == 0).astype(np.uint8)*255
        free = cv2.morphologyEx(free, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
        free = cv2.morphologyEx(free, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))
        
        # =====================================================================
        # STEP 6: Analyze Free Space
        # =====================================================================
        metrics = analyzer.update(grid, free, GRID_RES)
        regions = analyzer.find_free_regions(free, u_bins, v_bins, GRID_RES)
        
        # =====================================================================
        # STEP 7: Visualization
        # =====================================================================
        
        # Create main display
        display = color_bgr.copy()
        
        # Draw free space overlay - BRIGHT and VISIBLE
        overlay = np.zeros_like(color_bgr)  # Start with black canvas
        
        for v_idx in range(0, len(v_bins), FREE_SPACE_SUBSAMPLE):
            for u_idx in range(0, len(u_bins), FREE_SPACE_SUBSAMPLE):
                if free[v_idx, u_idx] > 0:
                    u_val = u_bins[u_idx]
                    v_val = v_bins[v_idx]
                    pos_3d = ctr + u_val*u_axis + v_val*v_axis
                    
                    csp = PyKinectV2._CameraSpacePoint(pos_3d[0], pos_3d[1], pos_3d[2])
                    col_pt = k._mapper.MapCameraPointToColorSpace(csp)
                    
                    # Check for valid mapping (avoid infinity values)
                    if np.isfinite(col_pt.x) and np.isfinite(col_pt.y):
                        x, y = int(col_pt.x), int(col_pt.y)
                        
                        if 0 <= x < 1920 and 0 <= y < 1080:
                            # Draw bright green circles for free space
                            cv2.circle(overlay, (x, y), FREE_SPACE_DOT_SIZE, (0, 255, 0), -1)
        
        # Blend overlay with higher visibility
        display = cv2.addWeighted(color_bgr, 1-OVERLAY_ALPHA, overlay, OVERLAY_ALPHA, 0)
        
        # Draw info panel
        info_y = 30
        line_height = 35
        
        # Background for text
        cv2.rectangle(display, (5, 5), (450, 200), (0, 0, 0), -1)
        cv2.rectangle(display, (5, 5), (450, 200), (255, 255, 255), 2)
        
        # Status text
        status_color = (0, 255, 0) if not metrics['is_cluttered'] else (0, 0, 255)
        cv2.putText(display, f"Free Space: {metrics['free_percentage']:.1f}%", 
                   (15, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        info_y += line_height
        
        cv2.putText(display, f"Free Area: {metrics['free_area_m2']:.4f} m²", 
                   (15, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        info_y += line_height
        
        cv2.putText(display, f"Occupied: {metrics['occupied_area_m2']:.4f} m²", 
                   (15, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        info_y += line_height
        
        cv2.putText(display, f"Regions: {len(regions)}", 
                   (15, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        info_y += line_height
        
        if metrics['is_cluttered']:
            cv2.putText(display, "⚠ CLUTTERED", 
                       (15, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Show main display
        cv2.imshow("Desk Space Monitor", display)
        
        # Optional: Show usage heatmap
        if show_heatmap:
            heatmap = analyzer.get_usage_heatmap()
            if heatmap is not None:
                heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_HOT)
                heatmap_resized = cv2.resize(heatmap_colored, (800, 600))
                cv2.putText(heatmap_resized, "Usage Heatmap (Hot = Frequently Occupied)", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow("Usage Heatmap", heatmap_resized)
        
        # Grid visualization
        grid_vis = cv2.applyColorMap((grid * 255).astype(np.uint8), cv2.COLORMAP_JET)
        free_vis = cv2.cvtColor(free, cv2.COLOR_GRAY2BGR)
        combined = np.hstack([grid_vis, free_vis])
        cv2.imshow("Grid View (Occupied | Free)", combined)
        
        # =====================================================================
        # STEP 8: Handle User Input
        # =====================================================================
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # ESC
            print("\n[MONITOR] Shutting down...")
            break
        elif key == ord('s') or key == ord('S'):  # Save
            analyzer.save_data()
            cv2.imwrite(f'desk_snapshot_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png', display)
            print("✓ Snapshot saved")
        elif key == ord('r') or key == ord('R'):  # Reset heatmap
            analyzer.reset_heatmap()
        elif key == ord('h') or key == ord('H'):  # Toggle heatmap
            show_heatmap = not show_heatmap
            if not show_heatmap:
                cv2.destroyWindow("Usage Heatmap")
    
    # =========================================================================
    # CLEANUP AND FINAL SAVE
    # =========================================================================
    print("\nSaving final data...")
    analyzer.save_data()
    cv2.destroyAllWindows()
    print("\n✓ Desk monitoring session complete!")
    print("=" * 70)

if __name__ == "__main__":
    main()

