# Desk Space Detection Pipeline - Flowchart Steps

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
│  Detect Desk        │
│  Surface (RANSAC)   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Transform to       │
│  Plane Coordinates  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Create Occupancy   │
│  Grid               │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Analyze Free       │
│  Space Regions      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Update Heatmap &   │
│  Time Series        │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Visualize Results  │
│  & Alert User       │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Log Data to CSV    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Loop or Exit       │
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
       │ Kinect Ready? │───NO──→ [ERROR: Connection Failed]
       └───────┬───────┘
               │ YES
               ▼
┌─────────────────────────────────────┐
│ Initialize DeskSpaceAnalyzer        │
│ - history_size = 300 frames        │
│ - free_space_history = []          │
│ - usage_heatmap = None             │
│ - last_clutter_alert = 0           │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ Initialize Display Windows          │
│ - Main: "Desk Space Monitor"       │
│ - Info panel dimensions             │
└──────────────┬──────────────────────┘
               │
               ▼
       [Ready - Begin Frame Loop]
```

---

### STEP 2: Acquire RGB-D Data

```
┌─────────────────────────────────────┐
│ Capture Current Frame               │
│ - color_frame (1920x1080x4 BGRA)   │
│ - depth_frame (512x424 uint16)     │
│ - frame_count++                     │
└──────────────┬──────────────────────┘
               │
               ▼
       ┌───────────────┐
       │ Valid frames? │───NO──→ [Skip frame, retry]
       └───────┬───────┘
               │ YES
               ▼
┌─────────────────────────────────────┐
│ Convert Color Frame                 │
│ - Reshape to (1080, 1920, 4)       │
│ - Extract BGR channels [:,:,:3]     │
│ - Convert to uint8                  │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ Map Depth to Camera Space (3D)     │
│ - Flatten depth: (424x512) → array │
│ - Call MapDepthFrameToCameraSpace  │
│ - Output: CameraSpacePoints array  │
│ - Reshape to (424, 512, 3)         │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ Filter Invalid Points               │
│ - Replace inf with NaN              │
│ - Create valid_mask (isfinite)     │
│ - Extract valid points P            │
└──────────────┬──────────────────────┘
               │
               ▼
       ┌───────────────┐
       │ Enough points │───NO──→ [Skip frame]
       │ (> 2000)?     │
       └───────┬───────┘
               │ YES
               ▼
     [Continue to Surface Detection]
```

**Data at this point:**
- `color_bgr`: RGB image (1920x1080x3)
- `depth_frame`: Depth map (512x424) millimeters
- `cam_3d`: 3D point cloud (512x424x3) meters
- `P`: Filtered valid 3D points (Nx3)

---

### STEP 3: Detect Desk Surface (RANSAC)

```
┌─────────────────────────────────────┐
│ RANSAC Plane Detection              │
│ - Max iterations: 400               │
│ - Inlier threshold: 1cm             │
│ - Required inliers: > 50%           │
└──────────────┬──────────────────────┘
               │
               ▼
       ┌───────────────────────┐
       │ For i in range(400)   │
       └───────┬───────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ Sample & Fit Plane                  │
│ 1. Sample 3 random points           │
│ 2. Compute plane equation:          │
│    n = cross(p1-p0, p2-p0)          │
│    n = n / ||n||                    │
│    d = -dot(n, centroid)            │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ Count Inliers                       │
│ - distance = |P @ n + d|            │
│ - inliers = (distance < 0.01)      │
│ - count = sum(inliers)              │
└──────────────┬──────────────────────┘
               │
               ▼
       ┌───────────────┐
       │ Best so far?  │───YES──→ [Store plane & inliers]
       └───────┬───────┘              └──────┬──────────┘
               │ NO                          │
               └─────────────────────────────┘
               │
               ▼
       [Next iteration]
               │
               ▼
┌─────────────────────────────────────┐
│ Refine Best Plane (SVD)             │
│ - Use all inlier points             │
│ - Compute centroid                  │
│ - SVD: U, S, Vt = svd(points-ctr)  │
│ - Normal = last row of Vt           │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ Orient Normal Upward                │
│ - If dot(n, [0,1,0]) < 0:           │
│   n = -n                             │
│   d = -d                             │
└──────────────┬──────────────────────┘
               │
               ▼
       ┌───────────────┐
       │ Plane found?  │───NO──→ [Skip frame]
       └───────┬───────┘
               │ YES
               ▼
     [Continue to Coordinate Transform]
```

**Output:**
- `plane_normal`: (nx, ny, nz) - perpendicular to desk
- `plane_center`: (cx, cy, cz) - center of desk surface
- `d`: Plane offset parameter
- `inlier_mask`: Boolean mask of desk points

---

### STEP 4: Transform to Plane Coordinates

```
┌─────────────────────────────────────┐
│ Create Plane Coordinate System     │
│ - Normal n (already have)           │
│ - Compute u-axis:                   │
│   u = cross(n, [0,0,1])             │
│   u = u / ||u||                     │
│ - Compute v-axis:                   │
│   v = cross(n, u)                   │
│   v = v / ||v||                     │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ Project 3D Points to 2D Plane       │
│ For each 3D point P:                │
│   rel = P - plane_center            │
│   px = dot(rel, u)                  │
│   py = dot(rel, v)                  │
│   height = dot(rel, n)              │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ Filter Plane Inliers                │
│ - Keep only points where:           │
│   |height| < 0.01m (within 1cm)    │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ Determine Plane Bounds              │
│ - px_min, px_max = min/max(px)     │
│ - py_min, py_max = min/max(py)     │
│ - width_m = px_max - px_min         │
│ - depth_m = py_max - py_min         │
└──────────────┬──────────────────────┘
               │
               ▼
       ┌───────────────┐
       │ Valid bounds? │───NO──→ [Skip frame]
       │ (> 0.1m)      │
       └───────┬───────┘
               │ YES
               ▼
     [Continue to Grid Creation]
```

**Output:**
- `plane_2d`: Array of (px, py) coordinates
- `plane_heights`: Height above/below plane
- `bounds`: (px_min, px_max, py_min, py_max)
- `u_axis`, `v_axis`: Plane coordinate system

---

### STEP 5: Create Occupancy Grid

```
┌─────────────────────────────────────┐
│ Calculate Grid Dimensions           │
│ - grid_res = 0.005m (5mm per cell) │
│ - nx = ceil(width_m / grid_res)    │
│ - ny = ceil(depth_m / grid_res)    │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ Initialize Grid                     │
│ - grid = zeros((ny, nx), dtype=uint8)│
│ - 0 = free space                    │
│ - 1 = occupied                      │
└──────────────┬──────────────────────┘
               │
               ▼
       ┌───────────────────────┐
       │ For Each 3D Point     │
       └───────┬───────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ Calculate Height Above Plane        │
│ rel = point - plane_center          │
│ h = dot(rel, plane_normal)          │
└──────────────┬──────────────────────┘
               │
               ▼
       ┌───────────────┐
       │ Is obstacle?  │───NO──→ [Skip point]
       │ 0.5cm < h < 8cm│
       └───────┬───────┘
               │ YES
               ▼
┌─────────────────────────────────────┐
│ Project to Grid                     │
│ px = dot(rel, u_axis)               │
│ py = dot(rel, v_axis)               │
│ gx = int((px - px_min) / grid_res)  │
│ gy = int((py - py_min) / grid_res)  │
└──────────────┬──────────────────────┘
               │
               ▼
       ┌───────────────┐
       │ Within grid?  │───NO──→ [Skip point]
       │ 0≤gx<nx       │
       │ 0≤gy<ny       │
       └───────┬───────┘
               │ YES
               ▼
┌─────────────────────────────────────┐
│ Mark Grid Cell as Occupied          │
│ grid[gy, gx] = 1                    │
└──────────────┬──────────────────────┘
               │
               ▼
       [Process next point]
               │
               ▼
┌─────────────────────────────────────┐
│ Apply Morphological Closing         │
│ - Kernel: 3x3 ellipse               │
│ - Fills small gaps                  │
│ - Connects nearby objects           │
└──────────────┬──────────────────────┘
               │
               ▼
     [Continue to Free Space Analysis]
```

**Output:**
- `grid`: Binary occupancy grid (ny × nx)
  - 0 = free/available space
  - 1 = occupied by objects
- `grid_res`: Resolution (0.005m)
- `grid_origin`: (px_min, py_min)

---

### STEP 6: Analyze Free Space

```
┌─────────────────────────────────────┐
│ Calculate Free Space Metrics        │
│ - total_cells = nx × ny             │
│ - occupied_cells = sum(grid == 1)   │
│ - free_cells = total_cells - occ    │
│ - free_percent = 100 × free/total   │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ Calculate Physical Areas            │
│ - cell_area_m2 = grid_res²          │
│ - free_area_m2 = free_cells × area  │
│ - occupied_area_m2 = occ × area     │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ Find Separate Free Regions          │
│ - Invert grid: free_mask = (grid==0)│
│ - Connected components analysis:    │
│   num_regions, labels =             │
│     cv2.connectedComponents(free)   │
└──────────────┬──────────────────────┘
               │
               ▼
       ┌───────────────────────┐
       │ For Each Region       │
       └───────┬───────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ Calculate Region Properties         │
│ - region_mask = (labels == i)       │
│ - region_area = sum(region_mask)    │
│ - region_area_m2 = area × cell_area │
│ - region_centroid (cy, cx)          │
└──────────────┬──────────────────────┘
               │
               ▼
       ┌───────────────┐
       │ Area > min?   │───NO──→ [Discard region]
       │ (> 0.01 m²)   │
       └───────┬───────┘
               │ YES
               ▼
┌─────────────────────────────────────┐
│ Find Region Bounding Box            │
│ - y_coords, x_coords = where(mask)  │
│ - x1, x2 = min(x), max(x)           │
│ - y1, y2 = min(y), max(y)           │
│ - bbox = (x1, y1, x2, y2)           │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ Store Region Info                   │
│ - region_id                         │
│ - area_m2                           │
│ - centroid (grid coords)            │
│ - bounding_box                      │
└──────────────┬──────────────────────┘
               │
               ▼
       [Process next region]
               │
               ▼
┌─────────────────────────────────────┐
│ Sort Regions by Area                │
│ - Largest regions first             │
│ - Keep top 5 regions                │
└──────────────┬──────────────────────┘
               │
               ▼
     [Continue to Heatmap Update]
```

**Output:**
- `free_percentage`: Float (0-100%)
- `free_area_m2`: Physical area available
- `occupied_area_m2`: Physical area used
- `free_regions`: List of region info dicts
  - Sorted by area (largest first)
  - Top 5 regions retained

---

### STEP 7: Update Usage Heatmap & History

```
┌─────────────────────────────────────┐
│ Check Heatmap Initialization        │
│ If usage_heatmap is None:           │
│   usage_heatmap = zeros_like(grid)  │
│   heatmap.dtype = float32           │
└──────────────┬──────────────────────┘
               │
               ▼
       ┌───────────────┐
       │ Grid size     │───NO──→ [Create new heatmap]
       │ changed?      │              └─────┬─────┘
       └───────┬───────┘                    │
               │ YES                        │
               │                            │
               ▼                            │
┌─────────────────────────────────────┐   │
│ Handle Grid Resize                  │   │
│ - Create new heatmap (new size)     │   │
│ - Copy overlapping region from old  │   │
│ - Update heatmap reference          │   │
└──────────────┬──────────────────────┘   │
               │                            │
               │◄───────────────────────────┘
               ▼
┌─────────────────────────────────────┐
│ Accumulate Current Frame            │
│ usage_heatmap += grid.astype(float32)│
│ - Adds 1.0 to occupied cells        │
│ - Adds 0.0 to free cells            │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ Update Free Space History           │
│ history.append({                    │
│   'frame': frame_count,             │
│   'timestamp': time.time(),         │
│   'free_pct': free_percentage,      │
│   'free_area': free_area_m2,        │
│   'occupied_area': occupied_area_m2,│
│   'num_regions': len(free_regions)  │
│ })                                  │
└──────────────┬──────────────────────┘
               │
               ▼
       ┌───────────────┐
       │ History full? │───YES──→ [Remove oldest entry]
       │ (> 300)       │
       └───────┬───────┘
               │ NO
               ▼
┌─────────────────────────────────────┐
│ Calculate Time Series Stats         │
│ - avg_free_pct = mean(history)      │
│ - trend (increasing/decreasing)     │
│ - std_deviation                     │
└──────────────┬──────────────────────┘
               │
               ▼
     [Continue to Alert Check]
```

**Data Structures:**
```python
usage_heatmap: np.ndarray (ny × nx, float32)
  # Accumulated occupancy over time
  # Higher values = more frequently occupied

free_space_history: List[Dict]
  [{
    'frame': int,
    'timestamp': float,
    'free_pct': float,
    'free_area': float,
    'occupied_area': float,
    'num_regions': int
  }, ...]
```

---

### STEP 8: Alert System

```
┌─────────────────────────────────────┐
│ Check Clutter Threshold             │
│ occupied_pct = 100 - free_pct       │
└──────────────┬──────────────────────┘
               │
               ▼
       ┌───────────────┐
       │ Occupied > 70%│───NO──→ [No alert]
       └───────┬───────┘          └────┬────┘
               │ YES                   │
               ▼                       │
       ┌───────────────┐              │
       │ Time since    │───NO──→──────┘
       │ last alert    │
       │ > 60 sec?     │
       └───────┬───────┘
               │ YES
               ▼
┌─────────────────────────────────────┐
│ Generate Clutter Alert              │
│ - Print warning to console          │
│ - Display visual warning            │
│ - Update last_alert_time            │
│ - Log alert event                   │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ Calculate Alert Details             │
│ - Current occupancy: X%             │
│ - Largest free region: Y m²         │
│ - Recommendation: "Clear desk"      │
└──────────────┬──────────────────────┘
               │
               ▼
     [Continue to Visualization]
```

**Alert Conditions:**
- Clutter alert: occupied > 70%
- Rate limited: 1 per 60 seconds
- Visual + console notification

---

### STEP 9: Create Visualization

```
┌─────────────────────────────────────┐
│ Start with Color Image              │
│ vis = color_bgr.copy()              │
│ height, width = vis.shape[:2]       │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ Create Free Space Overlay           │
│ - overlay = color_bgr.copy()        │
└──────────────┬──────────────────────┘
               │
               ▼
       ┌───────────────────────┐
       │ For Each Grid Cell    │
       │ (subsampled 1:4)      │
       └───────┬───────────────┘
               │
               ▼
       ┌───────────────┐
       │ Is free space?│───NO──→ [Skip cell]
       │ grid[y,x]==0  │
       └───────┬───────┘
               │ YES
               ▼
┌─────────────────────────────────────┐
│ Map Grid → Color Space              │
│ - px = px_min + gx × grid_res       │
│ - py = py_min + gy × grid_res       │
│ - cam_pt = ctr + px×u + py×v        │
│ - col_pt = MapCameraToColorSpace    │
└──────────────┬──────────────────────┘
               │
               ▼
       ┌───────────────┐
       │ Valid color   │───NO──→ [Skip point]
       │ coordinates?  │
       │ isfinite(x,y) │
       └───────┬───────┘
               │ YES
               ▼
┌─────────────────────────────────────┐
│ Draw Free Space Indicator           │
│ - cv2.circle(overlay, (cx,cy),      │
│              radius=2,               │
│              color=GREEN,            │
│              filled)                 │
└──────────────┬──────────────────────┘
               │
               ▼
       [Process next cell]
               │
               ▼
┌─────────────────────────────────────┐
│ Blend Overlay with Original         │
│ alpha = 0.5                          │
│ vis = cv2.addWeighted(               │
│   vis, 1-alpha,                      │
│   overlay, alpha, 0)                 │
└──────────────┬──────────────────────┘
               │
               ▼
     [Continue to Draw Regions]
```

---

### STEP 9B: Draw Free Regions

```
       ┌───────────────────────┐
       │ For Each Free Region  │
       │ (top 5 by area)       │
       └───────┬───────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ Get Region Bounding Box             │
│ bbox = region['bounding_box']       │
│ (x1, y1, x2, y2) in grid coords     │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ Convert to Pixel Coordinates        │
│ - Map grid corners to camera space  │
│ - Map camera space to color space   │
│ - Get pixel (x,y) for each corner   │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ Draw Region Rectangle               │
│ - Color: cyan                       │
│ - Thickness: 2 pixels               │
│ - cv2.rectangle(vis, pt1, pt2, ...)│
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ Draw Region Label                   │
│ - Text: f"#{i} ({area:.3f} m²)"   │
│ - Position: above bbox              │
│ - Background: semi-transparent      │
│ - Font: cv2.FONT_HERSHEY_SIMPLEX    │
└──────────────┬──────────────────────┘
               │
               ▼
       [Next region]
               │
               ▼
     [Continue to Info Panel]
```

---

### STEP 9C: Create Info Panel

```
┌─────────────────────────────────────┐
│ Define Info Panel Area              │
│ - panel_height = 180 pixels         │
│ - panel_width = width               │
│ - Position: top of image            │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ Create Semi-Transparent Panel       │
│ - panel = zeros((180, width, 3))    │
│ - panel[:] = [0, 0, 0] (black)      │
│ - Blend with alpha = 0.7            │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ Draw Panel Text                     │
│ Line 1: "FREE SPACE: XX.X%"         │
│         Color: Green if >50%        │
│                Yellow if 30-50%      │
│                Red if <30%           │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ Line 2: "Area: X.XXX m² free"       │
│         "      Y.XXX m² occupied"    │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ Line 3: "Regions: N usable areas"   │
│         "Largest: X.XXX m²"          │
└──────────────┬──────────────────────┘
               │
               ▼
       ┌───────────────┐
       │ Cluttered?    │───YES──→ [Add warning text]
       │ (> 70% occ)   │          "⚠ CLUTTER ALERT"
       └───────┬───────┘          Red, blinking
               │ NO
               ▼
┌─────────────────────────────────────┐
│ Draw Usage Trend                    │
│ - Calculate 10-frame average        │
│ - Arrow: ↑ (improving)              │
│         → (stable)                  │
│         ↓ (getting cluttered)       │
└──────────────┬──────────────────────┘
               │
               ▼
     [Continue to Optional Views]
```

---

### STEP 9D: Optional Visualization Views

```
       ┌───────────────┐
       │ Show heatmap? │───NO──→ [Skip]
       │ (key 'H')     │          └──┬──┘
       └───────┬───────┘             │
               │ YES                 │
               ▼                     │
┌─────────────────────────────────────┐
│ Generate Heatmap Visualization      │
│ - Normalize: 0 to max(heatmap)      │
│ - Apply colormap: COLORMAP_JET      │
│   Blue = rarely used                │
│   Red = frequently used             │
│ - Resize to display size            │
│ - Display in separate window        │
└──────────────┬──────────────────────┘
               │                     │
               │◄────────────────────┘
               ▼
       ┌───────────────┐
       │ Show grid?    │───NO──→ [Skip]
       │ (key 'G')     │          └──┬──┘
       └───────┬───────┘             │
               │ YES                 │
               ▼                     │
┌─────────────────────────────────────┐
│ Generate Grid Visualization         │
│ - Convert grid to RGB image         │
│   Free = white (255,255,255)        │
│   Occupied = black (0,0,0)          │
│ - Resize to display size (2x)       │
│ - Display in separate window        │
└──────────────┬──────────────────────┘
               │                     │
               │◄────────────────────┘
               ▼
     [Continue to Display]
```

---

### STEP 10: Display Results

```
┌─────────────────────────────────────┐
│ Show Main Visualization             │
│ cv2.imshow("Desk Space Monitor", vis)│
└──────────────┬──────────────────────┘
               │
               ▼
       ┌───────────────┐
       │ Optional      │───YES──→ [cv2.imshow("Heatmap")]
       │ windows open? │
       └───────┬───────┘
               │ NO
               ▼
┌─────────────────────────────────────┐
│ Console Output (every 30 frames)    │
│ Print:                              │
│ - Free space: XX.X%                 │
│ - Trend: (↑/→/↓)                    │
│ - Top 3 regions                     │
│ - Alerts (if any)                   │
└──────────────┬──────────────────────┘
               │
               ▼
     [Continue to User Input]
```

---

### STEP 11: Handle User Input

```
┌─────────────────────────────────────┐
│ Wait for Key Press                  │
│ key = cv2.waitKey(1)                │
│ timeout = 1 ms                      │
└──────────────┬──────────────────────┘
               │
               ▼
       ┌───────────────┐
       │ ESC pressed?  │───YES──→ [Jump to Cleanup]
       └───────┬───────┘
               │ NO
               ▼
       ┌───────────────┐
       │ 'S' pressed?  │───YES──→ [Save snapshot]
       └───────┬───────┘          └─────┬──────┘
               │ NO                     │
               ▼                        │
       ┌───────────────┐               │
       │ 'H' pressed?  │───YES──→ [Toggle heatmap view]
       └───────┬───────┘          └─────┬──────┘
               │ NO                     │
               ▼                        │
       ┌───────────────┐               │
       │ 'G' pressed?  │───YES──→ [Toggle grid view]
       └───────┬───────┘          └─────┬──────┘
               │ NO                     │
               ▼                        │
       ┌───────────────┐               │
       │ 'R' pressed?  │───YES──→ [Reset heatmap]
       └───────┬───────┘          └─────┬──────┘
               │ NO                     │
               │◄───────────────────────┘
               ▼
     [Continue to Data Logging]
```

**Key Controls:**
- `ESC`: Exit program
- `S`: Save current snapshot
- `H`: Toggle heatmap display
- `G`: Toggle grid display
- `R`: Reset usage heatmap

---

### STEP 12: Data Logging (Periodic)

```
       ┌───────────────┐
       │ Every N       │───NO──→ [Skip logging]
       │ frames?       │          └────┬─────┘
       │ (N=30 or 60)  │               │
       └───────┬───────┘               │
               │ YES                   │
               ▼                       │
┌─────────────────────────────────────┐
│ Prepare CSV Data                    │
│ row = {                             │
│   'timestamp': datetime.now(),      │
│   'frame': frame_count,             │
│   'free_percentage': free_pct,      │
│   'free_area_m2': free_area,        │
│   'occupied_area_m2': occupied,     │
│   'num_regions': len(regions),      │
│   'largest_region_m2': regions[0]   │
│ }                                   │
└──────────────┬──────────────────────┘
               │                       │
               ▼                       │
┌─────────────────────────────────────┐
│ Append to CSV File                  │
│ - File: desk_monitor_TIMESTAMP.csv  │
│ - Create if doesn't exist           │
│ - Append row                        │
└──────────────┬──────────────────────┘
               │                       │
               │◄──────────────────────┘
               ▼
     [Loop back to Frame Capture]
```

**CSV Format:**
```csv
timestamp,frame,free_percentage,free_area_m2,occupied_area_m2,num_regions,largest_region_m2
2024-11-15 10:30:15,0,65.3,0.523,0.277,3,0.215
2024-11-15 10:30:16,30,64.8,0.519,0.281,3,0.212
...
```

---

### STEP 13: Cleanup & Exit

```
┌─────────────────────────────────────┐
│ User Pressed ESC                    │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ Save Final Data                     │
│ - Flush CSV file                    │
│ - Save final heatmap image          │
│   File: desk_monitor_heatmap_TS.png │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ Generate Summary Statistics         │
│ - Session duration                  │
│ - Total frames processed            │
│ - Average free space %              │
│ - Min/max free space                │
│ - Number of clutter alerts          │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ Print Summary to Console            │
│ ================================    │
│ Session Summary:                    │
│   Duration: XX:XX:XX                │
│   Frames: XXXX                      │
│   Avg Free Space: XX.X%             │
│   Range: XX.X% - XX.X%              │
│   Clutter Alerts: X                 │
│ ================================    │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ Close Kinect Sensor                 │
│ - Release hardware                  │
│ - Free memory                       │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ Close OpenCV Windows                │
│ cv2.destroyAllWindows()             │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ END                                 │
│ Exit Code: 0                        │
└─────────────────────────────────────┘
```

---

## Data Flow Summary

```
INPUT                 PROCESSING                OUTPUT
─────                 ──────────                ──────

Kinect Sensor
    │
    ├─→ Color Frame ────────────┐
    │   (1920×1080×4)           │
    │                           │
    └─→ Depth Frame ────┐       │
        (512×424)       │       │
                        ▼       ▼
                  ┌──────────────────┐
                  │  Depth → 3D      │
                  │  Coordinate Map  │
                  └────────┬─────────┘
                           │
                           ▼
                    3D Point Cloud
                    (512×424×3)
                           │
                           ▼
                  ┌──────────────────┐
                  │  RANSAC Plane    │
                  │  Detection       │
                  └────────┬─────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │  Transform to    │
                  │  Plane 2D Coords │
                  └────────┬─────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │  Occupancy Grid  │
                  │  Creation        │
                  └────────┬─────────┘
                           │
                ┌──────────┴──────────┐
                │                     │
                ▼                     ▼
        ┌────────────┐        ┌────────────┐
        │ Free Space │        │   Usage    │
        │  Analysis  │        │  Heatmap   │
        └──────┬─────┘        └──────┬─────┘
               │                     │
               └──────────┬──────────┘
                          │
              ┌───────────┴───────────┐
              │                       │
              ▼                       ▼
      ┌──────────────┐        ┌──────────────┐
      │Visualization │        │  Data Logs   │
      │  (OpenCV)    │        │    (CSV)     │
      └──────────────┘        └──────────────┘
              │                       │
              └───────────┬───────────┘
                          │
                          ▼
                    User Display
```

---

## Key Algorithms Detail

### Plane Coordinate Transform
```
Given:
  - 3D point P = (px, py, pz)
  - Plane center C = (cx, cy, cz)
  - Plane normal n = (nx, ny, nz)
  - Plane axes u, v (perpendicular to n)

Compute:
  relative = P - C
  plane_x = dot(relative, u)
  plane_y = dot(relative, v)
  height = dot(relative, n)

Result:
  2D position: (plane_x, plane_y)
  Height above plane: height
```

### Grid Cell Mapping
```
Given:
  - 2D plane coordinates (px, py)
  - Grid origin (px_min, py_min)
  - Grid resolution (grid_res = 0.005m)

Compute:
  gx = floor((px - px_min) / grid_res)
  gy = floor((py - py_min) / grid_res)

Result:
  Grid cell indices: (gx, gy)
```

### Free Space Percentage
```
total_cells = grid.shape[0] × grid.shape[1]
occupied_cells = sum(grid == 1)
free_cells = total_cells - occupied_cells

free_percentage = (free_cells / total_cells) × 100
free_area_m² = free_cells × (grid_res)²
occupied_area_m² = occupied_cells × (grid_res)²
```

### Usage Heatmap Normalization
```
# Accumulation (each frame):
usage_heatmap += grid.astype(float32)

# Visualization:
max_value = max(usage_heatmap)
normalized = (usage_heatmap / max_value) × 255
colored = cv2.applyColorMap(normalized, COLORMAP_JET)

Where:
  Blue (low) = rarely occupied
  Red (high) = frequently occupied
```

---

## Performance Considerations

**Frame Processing Time:**
1. **Depth→3D Mapping**: 20-30ms
   - Bottleneck: Kinect SDK coordinate mapper
2. **RANSAC**: 10-15ms
   - 400 iterations
3. **Grid Creation**: 5-10ms
   - Loop over valid points
4. **Free Space Analysis**: 3-5ms
   - Connected components
5. **Visualization**: 10-20ms
   - Mapping, drawing, blending

**Total: ~50-80ms per frame (12-20 FPS)**

**Optimization Opportunities:**
- Reduce RANSAC iterations (200 instead of 400)
- Subsample points for grid creation
- Lower grid resolution (0.01m instead of 0.005m)
- Skip visualization every N frames

---

## Memory Usage

```
Data Structure              Size            Notes
─────────────────────────────────────────────────
color_frame                 ~8 MB           1920×1080×4 bytes
depth_frame                 ~0.4 MB         512×424×2 bytes
cam_3d point cloud          ~2.5 MB         512×424×3×4 bytes
occupancy grid              ~variable       ny×nx (typically 100KB)
usage_heatmap               ~variable       ny×nx×4 (float32)
free_space_history          ~50 KB          300 frames × ~170 bytes
visualization images        ~25 MB          Multiple copies of color

Total: ~35-40 MB typical
```

---

## Error Handling & Edge Cases

```
┌─────────────────────────────┐
│ Potential Issues            │
└─────────────────────────────┘

1. Kinect Disconnection
   └─→ Check frame validity each iteration
       └─→ Retry or exit gracefully

2. No Plane Detected
   └─→ RANSAC finds < 50% inliers
       └─→ Skip frame, continue loop
       └─→ Display "No surface detected"

3. Grid Size Changes
   └─→ Desk bounds vary frame-to-frame
       └─→ Resize heatmap
       └─→ Copy overlapping region

4. Color Mapping Overflow
   └─→ Coordinates outside color image
       └─→ Check isfinite(x, y)
       └─→ Skip invalid points

5. Empty Free Regions
   └─→ Desk 100% occupied
       └─→ Display alert
       └─→ No regions to draw

6. Memory Leak
   └─→ History grows unbounded
       └─→ Limit to 300 frames
       └─→ Remove oldest entries
```

---

## Integration with Other Modules

```
desk_monitor.py
    │
    ├─→ Uses: ransac_plane()
    │         plane_axes()
    │         (shared with main.py)
    │
    ├─→ Can integrate: box_detector.py
    │         └─→ Detect objects causing clutter
    │
    └─→ Outputs:
        ├─→ Real-time visualization
        ├─→ CSV time-series data
        ├─→ Heatmap images
        └─→ Console alerts
```

---

## Output Files

```
Generated Files:
├─→ desk_monitor_YYYYMMDD_HHMMSS.csv
│   └─→ Time-series log
│       Columns: timestamp, frame, free_%, 
│               free_area_m2, occupied_area_m2,
│               num_regions, largest_region_m2
│
├─→ desk_monitor_heatmap_YYYYMMDD_HHMMSS.png
│   └─→ Final usage heatmap
│       Blue = rarely used areas
│       Red = frequently used areas
│
└─→ desk_snapshot_YYYYMMDD_HHMMSS.png
    └─→ Saved on 'S' key press
        Current view with overlays
```

---

## Comparison: desk_monitor.py vs main.py

```
Feature                    desk_monitor.py          main.py
──────────────────────────────────────────────────────────────
Purpose                    Monitor free space       Find placement
Surface Detection          RANSAC (same)            RANSAC (same)
Occupancy Grid             Yes (full desk)          Yes (obstacles only)
Free Space Analysis        Percentage, regions      Feasible positions
Optimization               N/A                      Distance transform
Heatmap                    Usage over time          N/A
Time-Series Logging        Yes (CSV)                N/A
Alerts                     Clutter warnings         N/A
Visualization              Free space overlay       Placement marker
Update Rate                Real-time continuous     Per-frame analysis
```


