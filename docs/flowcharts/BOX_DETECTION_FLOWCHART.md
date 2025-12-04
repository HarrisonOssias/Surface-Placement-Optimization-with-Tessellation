# Box Detection Pipeline - Flowchart Steps

## High-Level Overview

```
┌─────────────────────┐
│  START: Kinect      │
│  Capture Frame      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Acquire RGB-D Data │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Preprocess Data    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Detect Surface     │
│  (RANSAC Plane)     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Detect Boxes       │
│  (CV or ML)         │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Calculate 3D       │
│  Properties         │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Visualize &        │
│  Display Results    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  END: Output Boxes  │
└─────────────────────┘
```

---

## Detailed Step-by-Step Flowchart

### STEP 1: Initialize System

```
START
  │
  ▼
┌─────────────────────────────────────┐
│ Initialize Kinect Sensor            │
│ - Connect to Kinect v2              │
│ - Initialize color camera (1920x1080)│
│ - Initialize depth sensor (512x424) │
│ - Create coordinate mapper          │
└──────────────┬──────────────────────┘
               │
               ▼
       ┌───────────────┐
       │ Ready?        │───NO──→ [ERROR: Kinect Not Found]
       └───────┬───────┘
               │ YES
               ▼
     [Continue to Frame Capture]
```

---

### STEP 2: Acquire RGB-D Data

```
┌─────────────────────────────────────┐
│ Capture Current Frame               │
│ - Get color frame (1920x1080x4)    │
│ - Get depth frame (512x424)        │
│ - Timestamp the capture            │
└──────────────┬──────────────────────┘
               │
               ▼
       ┌───────────────┐
       │ Valid frames? │───NO──→ [Skip frame, continue]
       └───────┬───────┘
               │ YES
               ▼
┌─────────────────────────────────────┐
│ Convert Depth to 3D Point Cloud    │
│ - Map depth to camera space        │
│   (424x512) → (424x512x3)          │
│ - Each pixel → (X, Y, Z) meters    │
│ - Filter invalid points (inf/nan)  │
└──────────────┬──────────────────────┘
               │
               ▼
       ┌───────────────┐
       │ Enough valid  │───NO──→ [Skip frame]
       │ points?       │
       │ (>2000)       │
       └───────┬───────┘
               │ YES
               ▼
     [Continue to Surface Detection]
```

**Data at this point:**
- `color_bgr`: RGB image (1920x1080x3)
- `depth_frame`: Depth map (512x424) in millimeters
- `cam_3d`: 3D point cloud (512x424x3) in meters

---

### STEP 3: Detect Surface (RANSAC Plane Detection)

```
┌─────────────────────────────────────┐
│ RANSAC Plane Detection              │
│                                     │
│ Initialize: iterations = 0          │
└──────────────┬──────────────────────┘
               │
               ▼
       ┌───────────────────────┐
       │ While iter < 400      │
       └───────┬───────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ RANSAC Iteration                    │
│ 1. Randomly sample 3 points         │
│ 2. Fit plane: ax + by + cz + d = 0 │
│ 3. Calculate normal vector n=(a,b,c)│
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ Find Inliers                        │
│ - For each point P:                 │
│   distance = |P·n + d|              │
│ - If distance < threshold (1cm):    │
│   Mark as inlier                    │
└──────────────┬──────────────────────┘
               │
               ▼
       ┌───────────────┐
       │ Inliers > 50% │───YES──→ [Store as best plane]
       │ of points?    │
       └───────┬───────┘
               │ NO
               ▼
       [Next iteration]
               │
               ▼
       ┌───────────────┐
       │ Iterations    │───NO──→ [Loop back]
       │ complete?     │
       └───────┬───────┘
               │ YES
               ▼
┌─────────────────────────────────────┐
│ Refine Best Plane                   │
│ - Use all inliers (not just 3)     │
│ - SVD on inlier points              │
│ - Calculate plane normal n          │
│ - Calculate plane center            │
│ - Orient normal upward (dot [0,1,0])│
└──────────────┬──────────────────────┘
               │
               ▼
       ┌───────────────┐
       │ Plane found?  │───NO──→ [ERROR: No surface]
       └───────┬───────┘
               │ YES
               ▼
     [Continue to Box Detection]
```

**Output:**
- `plane_normal`: Unit vector perpendicular to surface (nx, ny, nz)
- `plane_center`: 3D point at center of surface (cx, cy, cz)
- `inlier_mask`: Boolean mask of points on the plane

---

### STEP 4A: Box Detection - Traditional CV Method

```
┌─────────────────────────────────────┐
│ Calculate Height Above Plane       │
│ For each 3D point P:               │
│   height = (P - plane_center)·n    │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ Create Height Map                   │
│ - Reshape heights to image (424x512)│
│ - Resize to color res (1080x1920)  │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ Filter Objects Above Surface        │
│ - Keep heights: 2cm < h < 40cm     │
│ - Create binary mask                │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ Morphological Operations            │
│ 1. Closing (5x5 kernel)             │
│    - Fill small gaps                │
│ 2. Opening (5x5 kernel)             │
│    - Remove small noise             │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ Find Contours                       │
│ - Detect connected components       │
│ - Extract boundaries                │
└──────────────┬──────────────────────┘
               │
               ▼
       ┌───────────────────────┐
       │ For Each Contour      │
       └───────┬───────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ Filter by Geometry                  │
│ - Area: 500 < A < 50,000 pixels    │
│ - Aspect ratio: 0.3 < AR < 3.0     │
└──────────────┬──────────────────────┘
               │
               ▼
       ┌───────────────┐
       │ Pass filters? │───NO──→ [Discard contour]
       └───────┬───────┘
               │ YES
               ▼
┌─────────────────────────────────────┐
│ Calculate 2D Bounding Box           │
│ - Get min/max X, Y from contour    │
│ - bbox_2d = (x1, y1, x2, y2)       │
└──────────────┬──────────────────────┘
               │
               ▼
     [Continue to 3D Calculation]
```

---

### STEP 4B: Box Detection - ML Method (YOLOv8)

```
┌─────────────────────────────────────┐
│ Prepare Input Image                 │
│ - Ensure RGB format                 │
│ - Size: 1920x1080x3                │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ Run YOLOv8 Inference                │
│ results = model(color_bgr)          │
│ - Automatic resizing to 640x640    │
│ - Forward pass through network     │
│ - Get detections with confidences  │
└──────────────┬──────────────────────┘
               │
               ▼
       ┌───────────────────────┐
       │ For Each Detection    │
       └───────┬───────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ Extract Detection Info              │
│ - Bounding box (x1,y1,x2,y2)       │
│ - Confidence score                  │
│ - Class ID and name                 │
└──────────────┬──────────────────────┘
               │
               ▼
       ┌───────────────┐
       │ Confidence    │───NO──→ [Discard detection]
       │ > 0.3?        │
       └───────┬───────┘
               │ YES
               ▼
┌─────────────────────────────────────┐
│ Map to Depth Resolution             │
│ - Scale bbox from 1920x1080        │
│   to 512x424                        │
└──────────────┬──────────────────────┘
               │
               ▼
     [Continue to 3D Calculation]
```

---

### STEP 5: Calculate 3D Properties

```
┌─────────────────────────────────────┐
│ Extract ROI from Data               │
│ - roi_heights = height_map[y1:y2,x1:x2]│
│ - roi_cam_3d = cam_3d[y1:y2,x1:x2]  │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ Filter Valid Points in ROI          │
│ - Keep points: 2cm < height < 40cm │
│ - Remove NaN/Inf values            │
└──────────────┬──────────────────────┘
               │
               ▼
       ┌───────────────┐
       │ Enough valid  │───NO──→ [Discard box]
       │ points?       │
       │ (>10)         │
       └───────┬───────┘
               │ YES
               ▼
┌─────────────────────────────────────┐
│ Calculate 3D Bounding Box           │
│ - min_point = min(valid_points)    │
│ - max_point = max(valid_points)    │
│ - bbox_3d = (min_point, max_point) │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ Calculate Dimensions                │
│ dims = max_point - min_point       │
│ - Width = dims[0] × 100 (cm)       │
│ - Depth = dims[2] × 100 (cm)       │
│ - Height = dims[1] × 100 (cm)      │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ Calculate Volume                    │
│ volume = width × depth × height    │
│ volume_liters = volume / 1000      │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ Calculate Height Above Surface      │
│ avg_height = mean(roi_heights) × 100│
│ (in centimeters)                    │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ Classify Box Size                   │
│ If volume < 0.5L:    "small_box"   │
│ Elif volume < 5.0L:  "medium_box"  │
│ Else:                "large_box"   │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ Create DetectedBox Object           │
│ - bbox_2d: 2D rectangle            │
│ - bbox_3d: 3D min/max points       │
│ - dimensions: (W, D, H) in cm      │
│ - volume: liters                   │
│ - height_above_surface: cm         │
│ - category: size classification    │
│ - confidence: (ML only)            │
└──────────────┬──────────────────────┘
               │
               ▼
       [Add to boxes list]
               │
               ▼
       ┌───────────────┐
       │ More boxes    │───YES──→ [Loop back to next box]
       │ to process?   │
       └───────┬───────┘
               │ NO
               ▼
     [Continue to Visualization]
```

---

### STEP 6: Visualization

```
┌─────────────────────────────────────┐
│ Create Visualization Image          │
│ vis = color_image.copy()            │
└──────────────┬──────────────────────┘
               │
               ▼
       ┌───────────────────────┐
       │ For Each Detected Box │
       └───────┬───────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ Draw Bounding Box                   │
│ - Rectangle at (x1,y1) to (x2,y2)  │
│ - Color by size:                    │
│   · Small: Magenta                  │
│   · Medium: Cyan                    │
│   · Large: Yellow                   │
│ - Thickness: 2px                    │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ Draw Label                          │
│ - Text: "{category} ({confidence})" │
│ - Position: Above bounding box     │
│ - Background: Semi-transparent     │
└──────────────┬──────────────────────┘
               │
               ▼
       ┌───────────────┐
       │ Show 3D info? │───NO──→ [Skip details]
       └───────┬───────┘
               │ YES
               ▼
┌─────────────────────────────────────┐
│ Draw 3D Information                 │
│ - "Dim: W×D×H cm"                   │
│ - "Vol: X.XX L"                     │
│ - Position: Below bounding box     │
└──────────────┬──────────────────────┘
               │
               ▼
       [Next box]
               │
               ▼
┌─────────────────────────────────────┐
│ Add Header Information              │
│ - "Boxes detected: N | Total: T"   │
│ - Background panel                  │
│ - Top-left corner                   │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ Display Image                       │
│ cv2.imshow("Box Detector", vis)     │
└──────────────┬──────────────────────┘
               │
               ▼
     [Continue to User Input]
```

---

### STEP 7: User Input & Loop Control

```
┌─────────────────────────────────────┐
│ Check for User Input                │
│ key = cv2.waitKey(1)                │
└──────────────┬──────────────────────┘
               │
               ▼
       ┌───────────────┐
       │ ESC pressed?  │───YES──→ [Cleanup & EXIT]
       └───────┬───────┘
               │ NO
               ▼
       ┌───────────────┐
       │ 'S' pressed?  │───YES──→ [Save image to file]
       └───────┬───────┘              └──┬─────────┘
               │ NO                      │
               ▼                         │
       ┌───────────────┐                │
       │ 'I' pressed?  │───YES──→ [Toggle 3D info display]
       └───────┬───────┘              └──┬─────────┘
               │ NO                      │
               │◄────────────────────────┘
               ▼
┌─────────────────────────────────────┐
│ Log Detection (every 30 frames)     │
│ Print to console:                   │
│ [FRAME N] Detected X boxes:         │
│   Box 1: category                   │
│     Dimensions: ...                 │
│     Volume: ...                     │
└──────────────┬──────────────────────┘
               │
               ▼
       [Loop back to STEP 2: Acquire Frame]
```

---

## Data Flow Summary

```
INPUT                PROCESSING              OUTPUT
─────                ──────────              ──────

Kinect Sensor
    │
    ├─→ Color Frame ──────────┐
    │   (1920×1080×4)         │
    │                         │
    └─→ Depth Frame ─────┐    │
        (512×424)        │    │
                         ▼    ▼
                   ┌──────────────┐
                   │  Coordinate  │
                   │    Mapper    │
                   └──────┬───────┘
                          │
                          ▼
                   3D Point Cloud
                   (512×424×3)
                          │
                          ▼
                   ┌──────────────┐
                   │    RANSAC    │
                   │Plane Detection│
                   └──────┬───────┘
                          │
                ┌─────────┴─────────┐
                │                   │
         Traditional CV          YOLOv8 ML
                │                   │
                │                   │
                └─────────┬─────────┘
                          │
                          ▼
                ┌──────────────────┐
                │ 3D Box Properties│
                └──────┬───────────┘
                       │
                       ▼
              List[DetectedBox]
                       │
        ┌──────────────┼──────────────┐
        │              │              │
        ▼              ▼              ▼
   Visualization   Console Log    CSV Export
   (OpenCV)        (Terminal)     (Optional)
```

---

## Key Algorithms Detail

### RANSAC (Random Sample Consensus)
```
Input: Point cloud P with N points
Output: Plane parameters (normal n, offset d)

REPEAT for K iterations:
    1. Sample 3 random points from P
    2. Fit plane through 3 points
    3. Count inliers (points within threshold)
    4. If inliers > best_so_far:
        best_model = current_plane
        best_inliers = current_inliers
    
RETURN refined plane using SVD on all inliers
```

### Height Calculation
```
For each 3D point P = (px, py, pz):
    vector = P - plane_center
    height = dot(vector, plane_normal)
    
Where:
    plane_normal = (nx, ny, nz)  # Unit vector
    plane_center = (cx, cy, cz)  # 3D point on plane
```

### 3D Bounding Box
```
Given valid 3D points V = [(x1,y1,z1), (x2,y2,z2), ...]

min_point = (min(x), min(y), min(z))
max_point = (max(x), max(y), max(z))

width = (max_x - min_x) * 100   # Convert to cm
depth = (max_z - min_z) * 100
height = (max_y - min_y) * 100

volume = (width × depth × height) / 1000  # Liters
```

---

## Performance Considerations

**Frame Rate Bottlenecks:**
1. **Coordinate Mapping** (20-30ms)
   - Depth → 3D point cloud
   - Most expensive operation

2. **RANSAC** (10-15ms)
   - 400 iterations
   - Can be optimized by reducing iterations

3. **Detection Method:**
   - Traditional CV: ~5-10ms
   - YOLOv8n (GPU): ~8-12ms
   - YOLOv8m (GPU): ~15-25ms

**Total Frame Processing Time:**
- Traditional CV: ~40-55ms (18-25 FPS)
- YOLOv8n: ~45-60ms (16-22 FPS)
- YOLOv8m: ~55-75ms (13-18 FPS)

---

## Error Handling Points

```
┌─────────────────────┐
│ Potential Failures  │
└─────────────────────┘

1. Kinect Connection
   └─→ Check at initialization
       └─→ Retry or exit gracefully

2. Invalid Frames
   └─→ Check color/depth validity
       └─→ Skip frame, continue loop

3. RANSAC Failure
   └─→ No plane found (< 50% inliers)
       └─→ Return empty box list

4. No Valid Points in ROI
   └─→ Box too small or noisy
       └─→ Discard detection

5. Coordinate Mapping Error
   └─→ Inf/NaN in 3D points
       └─→ Filter with np.isfinite()
```

---

## Integration Points

```
box_detector.py
    │
    ├─→ Can be imported by:
    │   - desk_monitor.py
    │   - main.py
    │   - Custom scripts
    │
    └─→ Outputs:
        - List[DetectedBox]
        - Visualization image
        - Console logging
        - Optional CSV export
```


