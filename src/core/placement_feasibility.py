"""
Placement Feasibility Module
============================

Core logic for determining if a box can fit in detected free space regions.
Connects box detection with desk space analysis to make placement decisions.
"""

import numpy as np
import cv2
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class BoxDimensions:
    """Represents 3D dimensions of a box."""
    length: float  # meters
    width: float   # meters
    height: float  # meters
    
    @property
    def footprint_area(self) -> float:
        """Calculate footprint area in m²."""
        return self.length * self.width
    
    @property
    def volume(self) -> float:
        """Calculate volume in m³."""
        return self.length * self.width * self.height


@dataclass
class FreeRegion:
    """Represents a free space region on the desk."""
    id: int
    area_m2: float
    centroid_uv: Tuple[float, float]  # (u, v) in plane coordinates
    centroid_grid: Tuple[float, float]  # (x, y) in grid indices
    bbox_grid: Tuple[int, int, int, int]  # (x, y, w, h) in grid cells
    mask: np.ndarray  # Binary mask of the region


@dataclass
class PlacementCandidate:
    """Represents a potential placement location."""
    position_uv: Tuple[float, float]  # (u, v) in plane coordinates
    position_grid: Tuple[int, int]  # (x, y) in grid indices
    orientation: float  # rotation in degrees (0, 90, 180, 270)
    clearance: Dict[str, float]  # distances to obstacles (front, back, left, right)
    score: float  # placement quality score (0-1)
    region_id: int  # which free region this belongs to
    fits: bool  # whether box actually fits


@dataclass
class PlacementResult:
    """Result of placement feasibility analysis."""
    feasible: bool
    best_candidate: Optional[PlacementCandidate]
    all_candidates: List[PlacementCandidate]
    reason: str  # explanation of decision
    box_dims: BoxDimensions
    desk_free_percentage: float


class PlacementFeasibilityAnalyzer:
    """
    Analyzes whether a box can be placed on a desk.
    
    Takes box dimensions and free space regions, determines feasibility,
    and ranks potential placement locations.
    """
    
    def __init__(self, grid_resolution: float = 0.005, safety_margin: float = 0.015):
        """
        Initialize analyzer.
        
        Args:
            grid_resolution: Size of each grid cell in meters (default 5mm)
            safety_margin: Safety buffer around box in meters (default 15mm)
        """
        self.grid_res = grid_resolution
        self.safety_margin = safety_margin
    
    def analyze(self, 
                box_dims: BoxDimensions,
                free_regions: List[FreeRegion],
                occupancy_grid: np.ndarray,
                u_bins: np.ndarray,
                v_bins: np.ndarray,
                try_orientations: List[float] = [0, 90]) -> PlacementResult:
        """
        Main analysis function.
        
        Args:
            box_dims: Dimensions of box to place
            free_regions: List of detected free space regions
            occupancy_grid: Binary occupancy grid (1=occupied, 0=free)
            u_bins: U-coordinate bins for grid
            v_bins: V-coordinate bins for grid
            try_orientations: Rotation angles to try (degrees)
        
        Returns:
            PlacementResult with feasibility decision and best location
        """
        # Calculate desk statistics
        total_cells = occupancy_grid.size
        free_cells = np.sum(occupancy_grid == 0)
        desk_free_pct = (free_cells / total_cells) * 100 if total_cells > 0 else 0
        
        # Quick checks
        if not free_regions:
            return PlacementResult(
                feasible=False,
                best_candidate=None,
                all_candidates=[],
                reason="No free space detected on desk",
                box_dims=box_dims,
                desk_free_percentage=desk_free_pct
            )
        
        # Check if any region is large enough
        box_area = box_dims.footprint_area
        max_region_area = max(r.area_m2 for r in free_regions)
        
        if max_region_area < box_area * 0.8:  # Need at least 80% of box area
            return PlacementResult(
                feasible=False,
                best_candidate=None,
                all_candidates=[],
                reason=f"Largest free space ({max_region_area*10000:.0f}cm²) too small for box ({box_area*10000:.0f}cm²)",
                box_dims=box_dims,
                desk_free_percentage=desk_free_pct
            )
        
        # Find all placement candidates
        all_candidates = []
        for region in free_regions:
            candidates = self._find_candidates_in_region(
                box_dims, region, occupancy_grid, u_bins, v_bins, try_orientations
            )
            all_candidates.extend(candidates)
        
        # Filter to only candidates that actually fit
        fitting_candidates = [c for c in all_candidates if c.fits]
        
        if not fitting_candidates:
            if all_candidates:
                reason = "Box doesn't fit in any free region (obstacles too close)"
            else:
                reason = "No valid placement positions found"
            
            return PlacementResult(
                feasible=False,
                best_candidate=None,
                all_candidates=all_candidates,
                reason=reason,
                box_dims=box_dims,
                desk_free_percentage=desk_free_pct
            )
        
        # Rank candidates by score (highest first)
        fitting_candidates.sort(key=lambda c: c.score, reverse=True)
        best = fitting_candidates[0]
        
        return PlacementResult(
            feasible=True,
            best_candidate=best,
            all_candidates=fitting_candidates,
            reason=f"Found {len(fitting_candidates)} valid placement location(s)",
            box_dims=box_dims,
            desk_free_percentage=desk_free_pct
        )
    
    def _find_candidates_in_region(self,
                                    box_dims: BoxDimensions,
                                    region: FreeRegion,
                                    occupancy_grid: np.ndarray,
                                    u_bins: np.ndarray,
                                    v_bins: np.ndarray,
                                    orientations: List[float]) -> List[PlacementCandidate]:
        """Find placement candidates within a single free region."""
        candidates = []
        
        # Get region bounding box
        rx, ry, rw, rh = region.bbox_grid
        
        # Try different orientations
        for orientation in orientations:
            # Determine box dimensions in this orientation
            if orientation in [0, 180]:
                box_u = box_dims.length
                box_v = box_dims.width
            else:  # 90, 270
                box_u = box_dims.width
                box_v = box_dims.length
            
            # Convert to grid cells (including safety margin)
            cells_u = int(np.ceil((box_u + 2 * self.safety_margin) / self.grid_res))
            cells_v = int(np.ceil((box_v + 2 * self.safety_margin) / self.grid_res))
            
            # Try placing box at region centroid
            cx, cy = region.centroid_grid
            cx, cy = int(cx), int(cy)
            
            # Check if box fits at centroid
            fits, clearance = self._check_fit(
                cx, cy, cells_u, cells_v, occupancy_grid
            )
            
            # Calculate score based on clearance
            score = self._calculate_score(clearance, fits)
            
            # Convert grid position to plane coordinates
            if 0 <= cx < len(u_bins) and 0 <= cy < len(v_bins):
                u_pos = u_bins[cx]
                v_pos = v_bins[cy]
            else:
                continue  # Invalid position
            
            candidate = PlacementCandidate(
                position_uv=(u_pos, v_pos),
                position_grid=(cx, cy),
                orientation=orientation,
                clearance=clearance,
                score=score,
                region_id=region.id,
                fits=fits
            )
            candidates.append(candidate)
        
        return candidates
    
    def _check_fit(self, 
                   cx: int, cy: int, 
                   cells_u: int, cells_v: int,
                   occupancy_grid: np.ndarray) -> Tuple[bool, Dict[str, float]]:
        """
        Check if box fits at given position.
        
        Returns:
            (fits, clearance_dict)
        """
        h, w = occupancy_grid.shape
        
        # Calculate box boundaries
        half_u = cells_u // 2
        half_v = cells_v // 2
        
        x1 = max(0, cx - half_u)
        x2 = min(w, cx + half_u)
        y1 = max(0, cy - half_v)
        y2 = min(h, cy + half_v)
        
        # Check if we're within grid bounds
        if x1 >= x2 or y1 >= y2:
            return False, {'front': 0, 'back': 0, 'left': 0, 'right': 0}
        
        # Extract region where box would be placed
        box_region = occupancy_grid[y1:y2, x1:x2]
        
        # Check if any cells are occupied
        fits = np.sum(box_region) == 0
        
        # Calculate clearance distances (distance to nearest obstacle)
        clearance = self._calculate_clearance(cx, cy, cells_u, cells_v, occupancy_grid)
        
        return fits, clearance
    
    def _calculate_clearance(self,
                            cx: int, cy: int,
                            cells_u: int, cells_v: int,
                            occupancy_grid: np.ndarray) -> Dict[str, float]:
        """Calculate clearance distances to obstacles in each direction."""
        h, w = occupancy_grid.shape
        half_u = cells_u // 2
        half_v = cells_v // 2
        
        clearance = {'front': 0, 'back': 0, 'left': 0, 'right': 0}
        
        # Front (negative v direction)
        y_front = cy - half_v
        if y_front >= 0:
            for i in range(y_front, -1, -1):
                if i >= h or occupancy_grid[i, cx] == 1:
                    break
                clearance['front'] += self.grid_res
        
        # Back (positive v direction)
        y_back = cy + half_v
        if y_back < h:
            for i in range(y_back, h):
                if occupancy_grid[i, cx] == 1:
                    break
                clearance['back'] += self.grid_res
        
        # Left (negative u direction)
        x_left = cx - half_u
        if x_left >= 0:
            for i in range(x_left, -1, -1):
                if i >= w or occupancy_grid[cy, i] == 1:
                    break
                clearance['left'] += self.grid_res
        
        # Right (positive u direction)
        x_right = cx + half_u
        if x_right < w:
            for i in range(x_right, w):
                if occupancy_grid[cy, i] == 1:
                    break
                clearance['right'] += self.grid_res
        
        return clearance
    
    def _calculate_score(self, clearance: Dict[str, float], fits: bool) -> float:
        """
        Calculate placement quality score.
        
        Higher score = better placement (more clearance, centered)
        
        Returns:
            Score between 0 and 1
        """
        if not fits:
            return 0.0
        
        # Average clearance
        avg_clearance = np.mean(list(clearance.values()))
        
        # Normalize to 0-1 (assume max useful clearance is 20cm)
        score = min(avg_clearance / 0.20, 1.0)
        
        # Bonus for balanced clearance (all sides similar)
        clearances = list(clearance.values())
        clearance_std = np.std(clearances)
        balance_score = 1.0 - min(clearance_std / 0.10, 1.0)
        
        # Combine scores (70% clearance, 30% balance)
        final_score = 0.7 * score + 0.3 * balance_score
        
        return final_score


# Convenience function for quick checks
def can_box_fit(box_length: float, box_width: float, box_height: float,
                free_regions: List[FreeRegion],
                occupancy_grid: np.ndarray,
                u_bins: np.ndarray,
                v_bins: np.ndarray) -> Tuple[bool, Optional[PlacementCandidate]]:
    """
    Quick check if a box can fit on the desk.
    
    Args:
        box_length, box_width, box_height: Box dimensions in meters
        free_regions: List of detected free space regions
        occupancy_grid: Binary occupancy grid
        u_bins, v_bins: Grid coordinate bins
    
    Returns:
        (feasible, best_position) tuple
    """
    box_dims = BoxDimensions(box_length, box_width, box_height)
    analyzer = PlacementFeasibilityAnalyzer()
    result = analyzer.analyze(box_dims, free_regions, occupancy_grid, u_bins, v_bins)
    
    return result.feasible, result.best_candidate

