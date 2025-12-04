"""
Box Detection Module for Kinect RGB-D Data
===========================================

Detects boxes and packages on desk surfaces using RGB-D data.
Can be trained on the UW RGB-D Object Dataset or run with pre-trained models.

Features:
- 3D bounding box detection using depth information
- Object classification (box types, sizes)
- Integration with desk_monitor.py and main.py
- Support for YOLOv5/YOLOv8 models trained on RGB-D data
- Traditional CV fallback (contour-based detection)

Dataset: https://rgbd-dataset.cs.washington.edu/index.html
The UW RGB-D Object Dataset contains 300 objects including boxes, packages, 
and containers that can be used for training.

Usage:
    python box_detector.py

Or import for use with other scripts:
    from box_detector import BoxDetector
    detector = BoxDetector()
    boxes = detector.detect(color_frame, depth_frame)
"""

from pykinect2 import PyKinectRuntime, PyKinectV2
import numpy as np
import cv2
import ctypes
from dataclasses import dataclass
from typing import List, Tuple, Optional
import time

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Detection parameters
MIN_BOX_AREA = 500  # Minimum pixel area for box detection
MAX_BOX_AREA = 50000  # Maximum pixel area
MIN_ASPECT_RATIO = 0.3  # Minimum width/height ratio
MAX_ASPECT_RATIO = 3.0  # Maximum width/height ratio
MIN_BOX_HEIGHT = 0.02  # Minimum height above surface (meters)
MAX_BOX_HEIGHT = 0.40  # Maximum height (meters)

# RANSAC plane detection
PLANE_INLIER_THRESH = 0.010
RANSAC_ITERS = 400

# Visualization
BOX_COLOR = (255, 0, 255)  # Magenta for boxes
BOX_THICKNESS = 3

# ==============================================================================
# DATA STRUCTURES
# ==============================================================================

@dataclass
class DetectedBox:
    """Container for detected box information."""
    bbox_2d: Tuple[int, int, int, int]  # (x, y, width, height) in color image
    centroid_3d: np.ndarray  # 3D position in camera space
    dimensions_3d: Tuple[float, float, float]  # (width, height, depth) in meters
    height_above_surface: float  # Height above desk surface (meters)
    confidence: float  # Detection confidence (0-1)
    label: str  # "box", "package", "container", etc.
    volume_m3: float  # Estimated volume in cubic meters

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def ransac_plane(P, iters=RANSAC_ITERS, thresh=PLANE_INLIER_THRESH):
    """RANSAC plane fitting for surface detection."""
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
    
    Pin = P[best[2]]
    ctr = Pin.mean(axis=0)
    _, _, Vt = np.linalg.svd(Pin - ctr, full_matrices=False)
    n = Vt[-1]
    n /= np.linalg.norm(n)
    d = -np.dot(n, ctr)
    
    if np.dot(n, np.array([0,1,0], dtype=np.float32)) < 0:
        n = -n
        d = -d
    
    return n, d, best[2], ctr

def plane_axes(n):
    """Create 2D coordinate system on plane."""
    zf = np.array([0,0,1], np.float32)
    u = np.cross(n, zf)
    if np.linalg.norm(u) < 1e-6:
        u = np.cross(n, np.array([1,0,0], np.float32))
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    v /= np.linalg.norm(v)
    return u, v

# ==============================================================================
# BOX DETECTOR CLASS
# ==============================================================================

class BoxDetector:
    """
    Detects boxes and packages using RGB-D data.
    
    Can use either:
    1. Traditional CV methods (contour-based, default)
    2. Deep learning models trained on RGB-D Object Dataset (optional)
    """
    
    def __init__(self, use_ml_model=False, model_path=None):
        """
        Initialize box detector.
        
        Args:
            use_ml_model: If True, use deep learning model (requires training)
            model_path: Path to trained model weights
        """
        self.use_ml_model = use_ml_model
        self.model = None
        
        if use_ml_model and model_path:
            self._load_model(model_path)
        
        self.detection_count = 0
        self.last_plane = None
    
    def _load_model(self, model_path):
        """Load pre-trained YOLOv8 detection model."""
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            print(f"✓ Loaded YOLOv8 model from {model_path}")
        except ImportError:
            print("[ERROR] ultralytics package not installed")
            print("Install with: pip install ultralytics")
            print("Falling back to traditional CV detection")
            self.use_ml_model = False
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            print("Falling back to traditional CV detection")
            self.use_ml_model = False
    
    def detect_ml(self, color_bgr, cam_3d, plane_normal, plane_center):
        """
        Detect boxes using trained YOLOv8 model.
        
        Args:
            color_bgr: Color image (H,W,3) in BGR
            cam_3d: 3D point cloud (H,W,3) in camera space
            plane_normal: Normal vector of detected plane
            plane_center: Center point of plane
        
        Returns:
            List of DetectedBox objects
        """
        if self.model is None:
            print("[WARNING] No model loaded, falling back to traditional detection")
            return self.detect_traditional(color_bgr, cam_3d, plane_normal, plane_center)
        
        detected_boxes = []
        
        # Calculate height map for 3D measurements
        vec = cam_3d.reshape(-1, 3) - plane_center
        valid_mask = np.isfinite(vec).all(axis=1)
        heights = np.full(vec.shape[0], np.nan)
        heights[valid_mask] = vec[valid_mask] @ plane_normal
        height_map = heights.reshape(cam_3d.shape[:2])
        
        # Run YOLOv8 inference
        results = self.model(color_bgr, verbose=False)
        
        # Process detections
        for result in results:
            boxes = result.boxes
            
            for i in range(len(boxes)):
                # Get bounding box in xyxy format
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Get confidence and class
                confidence = float(boxes.conf[i])
                class_id = int(boxes.cls[i])
                class_name = self.model.names[class_id]
                
                # Skip low confidence detections
                if confidence < 0.3:
                    continue
                
                # Get ROI from depth/3D data
                h, w = height_map.shape
                x1_c = max(0, min(x1, w-1))
                x2_c = max(0, min(x2, w))
                y1_c = max(0, min(y1, h-1))
                y2_c = max(0, min(y2, h))
                
                if x2_c <= x1_c or y2_c <= y1_c:
                    continue
                
                roi_heights = height_map[y1_c:y2_c, x1_c:x2_c]
                roi_cam_3d = cam_3d[y1_c:y2_c, x1_c:x2_c]
                
                # Filter valid points
                valid_roi = (roi_heights > MIN_BOX_HEIGHT) & (roi_heights < MAX_BOX_HEIGHT)
                
                if np.count_nonzero(valid_roi) < 10:
                    continue
                
                # Get 3D points
                roi_points = roi_cam_3d.reshape(-1, 3)
                valid_roi_flat = valid_roi.flatten()
                valid_points = roi_points[valid_roi_flat]
                
                if valid_points.shape[0] < 10:
                    continue
                
                # Compute 3D bounding box
                min_pt = valid_points.min(axis=0)
                max_pt = valid_points.max(axis=0)
                
                # Dimensions in meters
                dims = max_pt - min_pt
                depth_cm = dims[2] * 100
                width_cm = dims[0] * 100
                height_cm = dims[1] * 100
                
                # Volume in liters
                volume = (dims[0] * dims[1] * dims[2]) * 1000
                
                # Average height above surface
                avg_height = roi_heights[valid_roi].mean() * 100
                
                # Classify by size (override YOLO's classification with size-based)
                if volume < 0.5:
                    size_category = 'small_box'
                elif volume < 5.0:
                    size_category = 'medium_box'
                else:
                    size_category = 'large_box'
                
                detected_boxes.append(DetectedBox(
                    bbox_2d=(x1, y1, x2, y2),
                    bbox_3d=(min_pt, max_pt),
                    dimensions=(width_cm, depth_cm, height_cm),
                    volume=volume,
                    height_above_surface=avg_height,
                    category=class_name,
                    confidence=confidence
                ))
        
        return detected_boxes
    
    def detect_traditional(self, color_bgr, cam_3d, plane_normal, plane_center):
        """
        Detect boxes using traditional computer vision (contours + depth).
        
        Args:
            color_bgr: Color image (H,W,3) in BGR
            cam_3d: 3D point cloud (H,W,3) in camera space
            plane_normal: Normal vector of detected plane
            plane_center: Center point of plane
        
        Returns:
            List of DetectedBox objects
        """
        detected_boxes = []
        
        # Calculate height above plane for all points
        vec = cam_3d.reshape(-1, 3) - plane_center
        valid_mask = np.isfinite(vec).all(axis=1)
        heights = np.full(vec.shape[0], np.nan)
        heights[valid_mask] = vec[valid_mask] @ plane_normal
        height_map = heights.reshape(cam_3d.shape[:2])
        
        # Create mask for potential boxes (objects above surface)
        box_mask = (height_map >= MIN_BOX_HEIGHT) & (height_map <= MAX_BOX_HEIGHT)
        box_mask = box_mask.astype(np.uint8) * 255
        
        # ALWAYS resize to color image resolution for consistent processing
        # Depth is 512x424, Color is 1920x1080
        target_size = (color_bgr.shape[1], color_bgr.shape[0])
        box_mask_resized = cv2.resize(box_mask, target_size, interpolation=cv2.INTER_NEAREST)
        height_map_resized = cv2.resize(height_map, target_size, interpolation=cv2.INTER_LINEAR)
        cam_3d_resized = cv2.resize(cam_3d, target_size, interpolation=cv2.INTER_LINEAR)
        
        # Use resized versions for all processing
        box_mask = box_mask_resized
        height_map = height_map_resized
        cam_3d = cam_3d_resized
        
        # Morphological operations to clean up
        kernel = np.ones((5,5), np.uint8)
        box_mask = cv2.morphologyEx(box_mask, cv2.MORPH_CLOSE, kernel)
        box_mask = cv2.morphologyEx(box_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(box_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area
            if area < MIN_BOX_AREA or area > MAX_BOX_AREA:
                continue
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by aspect ratio (boxes should be reasonably rectangular)
            aspect_ratio = w / h if h > 0 else 0
            if aspect_ratio < MIN_ASPECT_RATIO or aspect_ratio > MAX_ASPECT_RATIO:
                continue
            
            # Calculate 3D properties
            # Get depth values within bounding box
            roi_heights = height_map[y:y+h, x:x+w]
            roi_cam_3d = cam_3d[y:y+h, x:x+w]
            
            # Handle empty ROI
            if roi_heights.size == 0 or roi_cam_3d.size == 0:
                continue
            
            valid_roi = np.isfinite(roi_heights)
            if np.count_nonzero(valid_roi) < 10:
                continue
            
            # Calculate centroid in 3D
            # Properly index the 3D array using the 2D mask
            if roi_cam_3d.ndim == 3:
                # Reshape roi_cam_3d to 2D array of points
                roi_points = roi_cam_3d.reshape(-1, 3)
                # Flatten the mask to match
                valid_roi_flat = valid_roi.flatten()
                # Now we can index properly
                valid_points = roi_points[valid_roi_flat]
            else:
                continue
            
            if valid_points.shape[0] == 0:
                continue
                
            centroid_3d = np.mean(valid_points, axis=0)
            
            # Calculate dimensions
            valid_heights = roi_heights[valid_roi]
            height_3d = np.max(valid_heights) - np.min(valid_heights)
            
            # Estimate width and depth from bounding box and depth
            # This is approximate - assumes box is roughly aligned with camera
            avg_depth = np.mean(valid_points[:, 2])  # Z coordinate
            if avg_depth <= 0:
                continue
            
            # Convert pixel dimensions to meters using depth
            # Kinect FOV: ~57 degrees horizontal, ~43 degrees vertical
            width_3d = (w / color_bgr.shape[1]) * 2 * avg_depth * np.tan(np.radians(57/2))
            depth_3d = (h / color_bgr.shape[0]) * 2 * avg_depth * np.tan(np.radians(43/2))
            
            # Height above surface
            height_above = np.mean(valid_heights)
            
            # Calculate volume
            volume = width_3d * depth_3d * height_3d
            
            # Determine label based on size
            if volume < 0.001:  # < 1 liter
                label = "small_box"
            elif volume < 0.01:  # < 10 liters
                label = "medium_box"
            else:
                label = "large_box"
            
            # Calculate confidence based on detection quality
            # Higher confidence for well-defined rectangular shapes
            rect_area = w * h
            contour_to_rect_ratio = area / rect_area if rect_area > 0 else 0
            confidence = min(contour_to_rect_ratio, 1.0)
            
            box = DetectedBox(
                bbox_2d=(x, y, w, h),
                centroid_3d=centroid_3d,
                dimensions_3d=(width_3d, height_3d, depth_3d),
                height_above_surface=float(height_above),
                confidence=confidence,
                label=label,
                volume_m3=volume
            )
            
            detected_boxes.append(box)
        
        return detected_boxes
    
    def detect(self, color_bgr, depth_frame, kinect_mapper=None):
        """
        Main detection function.
        
        Args:
            color_bgr: Color image in BGR format
            depth_frame: Depth frame from Kinect (424x512)
            kinect_mapper: Kinect coordinate mapper object
        
        Returns:
            List of DetectedBox objects
        """
        # Convert depth to 3D point cloud
        if kinect_mapper is None:
            print("[BOX_DETECTOR] Error: kinect_mapper required")
            return []
        
        # Map depth to camera space
        depth_flat = depth_frame.flatten().astype(np.uint16)
        CSP = np.zeros((depth_flat.shape[0],), dtype=PyKinectV2._CameraSpacePoint)
        
        depth_ptr = ctypes.cast(depth_flat.ctypes.data, ctypes.POINTER(ctypes.c_ushort))
        csp_ptr = ctypes.cast(CSP.ctypes.data, ctypes.POINTER(PyKinectV2._CameraSpacePoint))
        
        kinect_mapper.MapDepthFrameToCameraSpace(
            ctypes.c_uint(depth_flat.shape[0]),
            depth_ptr,
            ctypes.c_uint(CSP.shape[0]),
            csp_ptr
        )
        
        cam = np.frombuffer(CSP, dtype=[("x","<f4"),("y","<f4"),("z","<f4")])
        cam_3d = np.column_stack([cam["x"], cam["y"], cam["z"]]).reshape(424,512,3)
        cam_3d[np.isinf(cam_3d)] = np.nan
        
        # Detect plane
        flat = cam_3d.reshape(-1,3)
        valid = np.isfinite(flat).all(axis=1)
        P = flat[valid]
        
        if P.shape[0] < 2000:
            return []
        
        n, d, inliers, ctr = ransac_plane(P)
        if n is None:
            return []
        
        self.last_plane = (n, ctr)
        
        # Detect boxes
        if self.use_ml_model and self.model is not None:
            # Use trained ML model
            boxes = self.detect_ml(color_bgr, cam_3d, n, ctr)
        else:
            # Use traditional CV
            boxes = self.detect_traditional(color_bgr, cam_3d, n, ctr)
        
        self.detection_count += len(boxes)
        return boxes
    
    def visualize_detections(self, image, boxes: List[DetectedBox], show_3d_info=True):
        """
        Draw detected boxes on image.
        
        Args:
            image: Image to draw on
            boxes: List of DetectedBox objects
            show_3d_info: If True, show 3D dimensions and volume
        
        Returns:
            Image with visualizations
        """
        vis = image.copy()
        
        for i, box in enumerate(boxes):
            x, y, w, h = box.bbox_2d
            
            # Draw bounding box
            cv2.rectangle(vis, (x, y), (x+w, y+h), BOX_COLOR, BOX_THICKNESS)
            
            # Draw label
            label_text = f"{box.label} ({box.confidence:.2f})"
            cv2.putText(vis, label_text, (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, BOX_COLOR, 2)
            
            if show_3d_info:
                # Show dimensions
                dim_text = f"{box.dimensions_3d[0]*100:.1f}x{box.dimensions_3d[1]*100:.1f}x{box.dimensions_3d[2]*100:.1f}cm"
                cv2.putText(vis, dim_text, (x, y+h+20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, BOX_COLOR, 1)
                
                # Show volume
                vol_text = f"Vol: {box.volume_m3*1000:.1f}L"
                cv2.putText(vis, vol_text, (x, y+h+40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, BOX_COLOR, 1)
            
            # Draw centroid
            cx, cy = x + w//2, y + h//2
            cv2.circle(vis, (cx, cy), 5, BOX_COLOR, -1)
        
        return vis

# ==============================================================================
# STANDALONE APPLICATION
# ==============================================================================

def main():
    """Standalone box detection application."""
    
    print("=" * 70)
    print("BOX DETECTOR - RGB-D Object Detection")
    print("=" * 70)
    print("Initializing Kinect...")
    
    # Initialize Kinect
    k = PyKinectRuntime.PyKinectRuntime(
        PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Depth
    )
    
    # Initialize detector
    detector = BoxDetector()
    
    print("✓ Ready!")
    print("\nControls:")
    print("  ESC - Quit")
    print("  S   - Save detection image")
    print("  I   - Toggle 3D info display")
    print("\nDetecting boxes...\n")
    
    show_3d_info = True
    frame_count = 0
    
    while True:
        got_c = k.has_new_color_frame()
        got_d = k.has_new_depth_frame()
        
        if not (got_c and got_d):
            cv2.waitKey(1)
            continue
        
        # Get frames
        d = k.get_last_depth_frame().reshape((424,512)).astype(np.uint16)
        c = k.get_last_color_frame().reshape((1080,1920,4)).astype(np.uint8)
        color_bgr = c[...,:3][:,:,::-1]
        
        # Detect boxes
        boxes = detector.detect(color_bgr, d, k._mapper)
        
        # Visualize
        vis = detector.visualize_detections(color_bgr, boxes, show_3d_info)
        
        # Add info panel
        info_text = f"Boxes detected: {len(boxes)} | Total: {detector.detection_count}"
        cv2.rectangle(vis, (5, 5), (550, 45), (0, 0, 0), -1)
        cv2.putText(vis, info_text, (15, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display
        cv2.imshow("Box Detector", vis)
        
        # Print detection info
        if len(boxes) > 0 and frame_count % 30 == 0:  # Every second
            print(f"\n[FRAME {frame_count}] Detected {len(boxes)} boxes:")
            for i, box in enumerate(boxes):
                print(f"  Box {i+1}: {box.label}")
                print(f"    Dimensions: {box.dimensions_3d[0]*100:.1f} x "
                      f"{box.dimensions_3d[1]*100:.1f} x {box.dimensions_3d[2]*100:.1f} cm")
                print(f"    Volume: {box.volume_m3*1000:.2f} liters")
                print(f"    Height above surface: {box.height_above_surface*100:.1f} cm")
        
        frame_count += 1
        
        # Handle input
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('s') or key == ord('S'):
            filename = f'box_detection_{time.strftime("%Y%m%d_%H%M%S")}.png'
            cv2.imwrite(filename, vis)
            print(f"✓ Saved: {filename}")
        elif key == ord('i') or key == ord('I'):
            show_3d_info = not show_3d_info
    
    cv2.destroyAllWindows()
    print("\n✓ Box detection complete!")
    print(f"Total boxes detected: {detector.detection_count}")
    print("=" * 70)

if __name__ == "__main__":
    main()

