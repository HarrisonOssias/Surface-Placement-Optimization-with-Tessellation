"""
K-PLACE: Kinect-based Package Placement Optimizer
==================================================

This script uses a Microsoft Kinect v2 sensor to find the optimal location
to place a package on a desk surface, avoiding obstacles.

ALGORITHM OVERVIEW:
-------------------
1. CAPTURE: Get depth frame from Kinect (512x424 pixels, distance in mm)
2. CONVERT: Transform depth image to 3D point cloud (X,Y,Z in meters)
3. DETECT: Use RANSAC to find the dominant plane (desk surface)
4. TRANSFORM: Convert 3D points to 2D plane coordinates (u,v,height)
5. GRID: Create occupancy grid marking obstacles (5mm resolution)
6. FEASIBILITY: Use erosion to find where package fits safely
7. OPTIMIZE: Choose location farthest from obstacles (distance transform)
8. VISUALIZE: Project result onto color image and display

COORDINATE SYSTEMS:
-------------------
- Camera Space (X,Y,Z): 3D coordinates from Kinect's perspective (meters)
  * X: right, Y: up, Z: forward (away from camera)
- Plane Space (u,v,h): 2D coordinates on the desk surface
  * u: horizontal position on desk, v: depth position, h: height above desk
- Grid Space (i,j): Discrete grid cells for occupancy map
  * Each cell = 5mm x 5mm
- Color Space (x,y): Pixel coordinates in the 1920x1080 color image

OUTPUTS:
--------
- Color camera feed with green placement marker
- Depth visualization (grayscale)
- Feasibility heatmap (blue=bad, red=good placement locations)

NOTE: Must run on Windows with Kinect SDK 2.0 installed (not WSL)
"""

from pykinect2 import PyKinectRuntime, PyKinectV2
import numpy as np
import cv2, math, os, time
import ctypes

# ==============================================================================
# CONFIGURATION PARAMETERS
# ==============================================================================
# These parameters control the package size, safety margins, and detection sensitivity.
# Adjust these based on your specific package dimensions and surface conditions.

PACKAGE_L = 0.30  # Package long dimension (meters) - e.g., 30 cm
PACKAGE_W = 0.20  # Package short dimension (meters) - e.g., 20 cm
MARGIN    = 0.015 # Safety buffer around package (meters) - prevents edge placement

GRID_RES  = 0.005 # Grid cell size (meters) - smaller = more precise but slower
                  # 5mm cells provide good balance between accuracy and performance

H_MIN = 0.005     # Minimum obstacle height (meters) - anything above 5mm is considered
H_MAX = 0.08      # Maximum obstacle height (meters) - ignore tall objects (like walls)
                  # This range focuses on typical desktop obstacles

PLANE_INLIER_THRESH = 0.010  # Distance tolerance for plane fitting (±10mm)
                             # Points within this distance are considered part of the plane

RANSAC_ITERS = 600  # Number of RANSAC iterations for plane detection
                    # More iterations = more robust but slower
# ==============================================================================

def ransac_plane(P, iters=RANSAC_ITERS, thresh=PLANE_INLIER_THRESH):
    """
    RANSAC Plane Fitting Algorithm
    ===============================
    
    This function finds the dominant plane in a 3D point cloud (the desk surface).
    
    RANSAC (RANdom SAmple Consensus) works by:
    1. Randomly selecting 3 points from the point cloud
    2. Fitting a plane through those 3 points
    3. Counting how many other points lie close to that plane (inliers)
    4. Repeating many times and keeping the plane with the most inliers
    5. Refining the result using all inlier points
    
    Input:
        P: Nx3 numpy array of 3D points (X, Y, Z coordinates in meters)
        iters: Number of random samples to try (more = more robust)
        thresh: Distance threshold for a point to be considered on the plane
    
    Output:
        n: Plane normal vector (unit vector perpendicular to the plane)
        d: Plane offset (distance from origin along normal)
        inliers: Boolean mask indicating which points are on the plane
        ctr: Center point of all inlier points
        
    Plane equation: n·P + d = 0  (for any point P on the plane)
    """
    rng = np.random.default_rng(1234)
    N = P.shape[0]
    best = (None, None, None)  # (n, d, inlier_mask)
    best_count = -1
    
    # Try many random samples to find the best plane
    for _ in range(iters):
        idx = rng.choice(N, 3, replace=False)
        p0, p1, p2 = P[idx]
        
        # Calculate plane normal from cross product of two edge vectors
        n = np.cross(p1-p0, p2-p0)
        nn = np.linalg.norm(n)
        if nn < 1e-6: continue  # Skip degenerate cases (collinear points)
        n /= nn  # Normalize to unit vector
        
        # Calculate plane offset
        d = -np.dot(n, p0)
        
        # Count inliers (points close to this plane)
        dist = np.abs(P @ n + d)
        inliers = dist < thresh
        c = inliers.sum()
        
        # Keep track of best plane so far
        if c > best_count:
            best = (n, d, inliers)
            best_count = c
    
    if best[0] is None:
        raise RuntimeError("Plane RANSAC failed.")
    
    # Refine the plane estimate using all inlier points (not just 3)
    # Use PCA (Principal Component Analysis) to find the best-fit plane
    Pin = P[best[2]]  # Get all inlier points
    ctr = Pin.mean(axis=0)  # Find center of mass
    
    # SVD gives us principal directions; smallest component is the normal
    _, _, Vt = np.linalg.svd(Pin - ctr, full_matrices=False)
    n = Vt[-1]  # Last row of Vt is the normal (smallest variance direction)
    n /= np.linalg.norm(n)
    d = -np.dot(n, ctr)
    
    # Ensure normal points upward (Kinect uses Y-up coordinate system)
    if np.dot(n, np.array([0,1,0], dtype=np.float32)) < 0:
        n = -n
        d = -d
    
    return n, d, best[2], ctr

def plane_axes(n):
    """
    Create 2D Coordinate System on the Plane
    =========================================
    
    Given a plane normal vector, this creates two perpendicular axes (u, v) 
    that lie flat on the plane. This allows us to work in 2D coordinates on 
    the plane surface instead of 3D world coordinates.
    
    Think of it like creating a local "grid paper" coordinate system on the desk.
    
    Input:
        n: Plane normal vector (perpendicular to the plane)
    
    Output:
        u: First axis vector (horizontal on the plane)
        v: Second axis vector (horizontal on the plane, perpendicular to u)
        
    These three vectors (u, v, n) form an orthonormal basis (like x, y, z axes).
    """
    zf = np.array([0,0,1], np.float32)
    u = np.cross(n, zf)
    
    # Handle edge case when normal is parallel to Z axis
    if np.linalg.norm(u) < 1e-6:
        u = np.cross(n, np.array([1,0,0], np.float32))
    
    u /= np.linalg.norm(u)  # Normalize to unit vector
    v = np.cross(n, u)       # v is perpendicular to both n and u
    v /= np.linalg.norm(v)
    
    return u, v

def main():
    """
    ============================================================================
    MAIN PROCESSING LOOP
    ============================================================================
    
    This is the core of the application. It continuously:
    1. Captures frames from the Kinect sensor
    2. Processes depth data to find the desk surface and obstacles
    3. Calculates optimal placement location
    4. Displays results to the user
    
    The loop runs until the user presses ESC.
    """
    
    # -------------------------------------------------------------------------
    # INITIALIZATION
    # -------------------------------------------------------------------------
    # Initialize Kinect sensor for both color and depth streams
    # Color: 1920x1080 RGB camera
    # Depth: 512x424 depth sensor (measures distance in millimeters)
    k = PyKinectRuntime.PyKinectRuntime(
        PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Depth
    )
    
    # Preallocate arrays for visualization
    color_bgr = np.zeros((1080,1920,3), np.uint8)  # Color image in BGR format (OpenCV)
    depth_vis = np.zeros((424,512), np.uint8)      # Grayscale depth visualization

    print("[K-PLACE] ESC to quit, S to save a snapshot.")
    
    # -------------------------------------------------------------------------
    # MAIN LOOP - Runs continuously until user quits
    # -------------------------------------------------------------------------
    while True:
        # Check if new frames are available from the sensor
        got_c = k.has_new_color_frame()
        got_d = k.has_new_depth_frame()

        # ---------------------------------------------------------------------
        # PROCESS COLOR FRAME
        # ---------------------------------------------------------------------
        if got_c:
            # Get color frame and convert from RGBA to BGR for OpenCV
            c = k.get_last_color_frame().reshape((1080,1920,4)).astype(np.uint8)
            color_bgr = c[...,:3][:,:,::-1]  # Drop alpha, reverse RGB to BGR

        # These will store results if we find a placement location
        placement_px = None  # (x, y) pixel coordinates for placement marker
        heat_vis = None      # Heatmap showing feasibility scores

        # ---------------------------------------------------------------------
        # PROCESS DEPTH FRAME - This is where the magic happens!
        # ---------------------------------------------------------------------
        if got_d:
            # Get raw depth frame (512x424 pixels, each pixel = distance in mm)
            d = k.get_last_depth_frame().reshape((424,512)).astype(np.uint16)
            
            # Create visualization (scale depth values to 0-255 for display)
            depth_vis = cv2.convertScaleAbs(d, alpha=0.03)

            # -----------------------------------------------------------------
            # STEP 1: Convert Depth Image to 3D Point Cloud
            # -----------------------------------------------------------------
            # Raw depth data is just distances. We need to convert each pixel 
            # into 3D (X,Y,Z) coordinates in real-world meters.
            # 
            # The Kinect SDK provides a mapper function that does this conversion
            # using the camera's calibration parameters (focal length, etc.)
            
            depth_flat = d.flatten().astype(np.uint16)  # Flatten to 1D array
            CSP = np.zeros((depth_flat.shape[0],), dtype=PyKinectV2._CameraSpacePoint)
            
            # Prepare pointers for C++ API call (Kinect SDK is written in C++)
            depth_ptr = ctypes.cast(depth_flat.ctypes.data, ctypes.POINTER(ctypes.c_ushort))
            csp_ptr = ctypes.cast(CSP.ctypes.data, ctypes.POINTER(PyKinectV2._CameraSpacePoint))
            
            # Call Kinect SDK to convert depth → 3D camera space
            # Result: Each pixel now has (X, Y, Z) coordinates in meters
            k._mapper.MapDepthFrameToCameraSpace(
                ctypes.c_uint(depth_flat.shape[0]),  # Number of depth points
                depth_ptr,                            # Input: depth values
                ctypes.c_uint(CSP.shape[0]),         # Number of output points
                csp_ptr                               # Output: 3D points
            )
            
            # Convert from Kinect's structure format to numpy array
            cam = np.frombuffer(CSP, dtype=[("x","<f4"),("y","<f4"),("z","<f4")])
            cam = np.column_stack([cam["x"], cam["y"], cam["z"]]).reshape(424,512,3)
            cam[np.isinf(cam)] = np.nan  # Replace invalid values with NaN

            # -----------------------------------------------------------------
            # STEP 2: Extract Valid 3D Points
            # -----------------------------------------------------------------
            # Some pixels don't have valid depth (too far, too close, or reflective surfaces)
            # Filter out invalid points before processing
            
            flat = cam.reshape(-1,3)  # Flatten to list of points
            valid = np.isfinite(flat).all(axis=1)  # Keep only points with valid X,Y,Z
            P = flat[valid]  # P is now our 3D point cloud
            
            # Need enough points for robust plane fitting
            if P.shape[0] > 2000:
                # ---------------------------------------------------------
                # STEP 3: Find the Desk Surface (Plane Detection)
                # ---------------------------------------------------------
                # Use RANSAC to find the dominant flat surface (the desk)
                # This separates the desk from walls, objects, etc.
                
                try:
                    n, d0, inliers, ctr = ransac_plane(P)
                except Exception:
                    n = None  # Plane detection failed
                
                if n is not None:
                    # -----------------------------------------------------
                    # STEP 4: Transform to Plane Coordinates
                    # -----------------------------------------------------
                    # Instead of working in 3D (X,Y,Z), we now work in plane coordinates:
                    # - u: horizontal position on the desk (meters)
                    # - v: depth position on the desk (meters)  
                    # - h: height above the desk surface (meters)
                    #
                    # This is like looking at the desk from directly above!
                    
                    u_axis, v_axis = plane_axes(n)  # Get coordinate axes on the plane
                    vec = flat - ctr  # Vector from plane center to each point
                    
                    # Project 3D points onto plane coordinate system
                    uvh = np.full((flat.shape[0],3), np.nan, np.float32)
                    uvh[valid,0] = vec[valid] @ u_axis  # u coordinate (horizontal)
                    uvh[valid,1] = vec[valid] @ v_axis  # v coordinate (depth)
                    uvh[valid,2] = vec[valid] @ n       # h coordinate (height above plane)
                    
                    UVH = uvh.reshape(424,512,3)  # Reshape back to image dimensions

                    # -----------------------------------------------------
                    # STEP 5: Create Occupancy Grid
                    # -----------------------------------------------------
                    # Build a 2D grid representing the desk surface.
                    # Each grid cell is 5mm x 5mm (GRID_RES).
                    # Grid values: 0 = free space, 1 = obstacle
                    
                    # Find the extent of the detected plane
                    plane_mask = np.abs(UVH[...,2]) < PLANE_INLIER_THRESH
                    
                    if np.count_nonzero(plane_mask) > 500:
                        u_vals = UVH[...,0][plane_mask]
                        v_vals = UVH[...,1][plane_mask]
                        u_min, u_max = np.nanmin(u_vals), np.nanmax(u_vals)
                        v_min, v_max = np.nanmin(v_vals), np.nanmax(v_vals)
                        
                        # Create grid bins (like histogram bins, but 2D)
                        u_bins = np.arange(u_min, u_max, GRID_RES)
                        v_bins = np.arange(v_min, v_max, GRID_RES)
                        grid = np.zeros((len(v_bins), len(u_bins)), np.uint8)

                        # Mark obstacles: anything between H_MIN and H_MAX above desk
                        H = UVH[...,2]  # Height above plane
                        U = UVH[...,0]  # U coordinate
                        V = UVH[...,1]  # V coordinate
                        
                        occ = (H >= H_MIN) & (H <= H_MAX) & np.isfinite(H)
                        
                        if np.count_nonzero(occ) > 0:
                            # Convert (u,v) coordinates to grid indices
                            ui = np.clip(((U[occ] - u_min)/GRID_RES).astype(int), 0, len(u_bins)-1)
                            vi = np.clip(((V[occ] - v_min)/GRID_RES).astype(int), 0, len(v_bins)-1)
                            grid[vi, ui] = 1  # Mark as occupied

                        # Clean up the grid with morphological operations
                        # (removes noise and fills small gaps)
                        free = (grid == 0).astype(np.uint8)*255  # Invert: 255 = free
                        free = cv2.morphologyEx(free, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))   # Remove small obstacles
                        free = cv2.morphologyEx(free, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8)) # Fill small holes

                        # -----------------------------------------------------
                        # STEP 6: Find Feasible Placement Locations
                        # -----------------------------------------------------
                        # Use erosion to find where the package (with safety margin) can fit.
                        # Erosion shrinks the free space by the package size, leaving only
                        # areas where the entire package would fit without hitting obstacles.
                        
                        # Calculate kernel size based on package dimensions + safety margin
                        ker_u = int(math.ceil((PACKAGE_L + 2*MARGIN)/GRID_RES))
                        ker_v = int(math.ceil((PACKAGE_W + 2*MARGIN)/GRID_RES))
                        ker = np.ones((ker_v, ker_u), np.uint8)  # Rectangular kernel
                        
                        # Erosion: areas where package fits = remaining white pixels
                        feasible = cv2.erode(free, ker, iterations=1)

                        if feasible.max() > 0:  # If any feasible locations exist
                            # ---------------------------------------------
                            # STEP 7: Choose Optimal Placement Location
                            # ---------------------------------------------
                            # Among all feasible locations, pick the one farthest from obstacles.
                            # This maximizes safety and gives the most clearance.
                            
                            # Distance transform: each pixel = distance to nearest obstacle
                            dist2edge = cv2.distanceTransform((free>0).astype(np.uint8), cv2.DIST_L2, 5)
                            
                            # Score = distance to obstacles, but only in feasible regions
                            score = dist2edge * (feasible>0).astype(np.float32)
                            
                            # Find the location with the highest score
                            v_best, u_best = np.unravel_index(np.argmax(score), score.shape)
                            
                            # ---------------------------------------------
                            # STEP 8: Convert Back to Camera/Color Space
                            # ---------------------------------------------
                            # We found the best location in grid coordinates (u,v).
                            # Now convert back to 3D world space, then to color image pixels
                            # so we can draw the marker on the screen.
                            
                            u_c = u_bins[u_best]  # Grid index → meters on plane
                            v_c = v_bins[v_best]
                            
                            # Convert plane coordinates back to 3D world position
                            center3d = ctr + u_c*u_axis + v_c*v_axis
                            
                            # Project 3D position onto color camera image
                            csp = PyKinectV2._CameraSpacePoint(center3d[0], center3d[1], center3d[2])
                            col_pt = k._mapper.MapCameraPointToColorSpace(csp)
                            x, y = int(col_pt.x), int(col_pt.y)
                            
                            # Check if projection is within color image bounds
                            if 0 <= x < 1920 and 0 <= y < 1080:
                                placement_px = (x, y)  # Store for visualization

                            # Create heatmap visualization (optional but helpful)
                            heat = cv2.normalize(score, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                            heat_vis = cv2.applyColorMap(heat, cv2.COLORMAP_JET)  # Blue=bad, Red=good

        # =====================================================================
        # VISUALIZATION AND USER INTERFACE
        # =====================================================================
        # Display the results to the user in real-time windows
        
        color_draw = color_bgr.copy()
        
        # Draw placement marker if we found a valid location
        if placement_px is not None:
            cv2.drawMarker(color_draw, placement_px, (40,220,40), 
                          cv2.MARKER_TILTED_CROSS, 40, 3)
            cv2.putText(color_draw, "PLACEMENT", (placement_px[0]+8, placement_px[1]-8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40,220,40), 2, cv2.LINE_AA)

        # Show all visualization windows
        # cv2.imshow("K-PLACE color", color_draw)      # Main view with placement marker
        cv2.imshow("K-PLACE depth (mm)", depth_vis)  # Depth visualization
        # if heat_vis is not None:
        #     cv2.imshow("K-PLACE feasibility", heat_vis)  # Heatmap of scores

        # =====================================================================
        # USER INPUT HANDLING
        # =====================================================================
        kkey = cv2.waitKey(1) & 0xFF
        
        if kkey == 27:  # ESC key
            break  # Exit the main loop
        
        if kkey == ord('s'):  # S key - save snapshot
            ts = time.strftime("%Y%m%d_%H%M%S")
            os.makedirs("captures", exist_ok=True)
            cv2.imwrite(f"captures/{ts}_color.png", color_draw)
            cv2.imwrite(f"captures/{ts}_depth.png", depth_vis)
            print(f"[K-PLACE] Snapshot saved: {ts}")

    # Cleanup
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
