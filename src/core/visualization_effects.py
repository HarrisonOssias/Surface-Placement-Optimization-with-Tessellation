"""
Visualization Effects Module
============================

Animated visual feedback for the GUI application.
"""

import numpy as np
import cv2
import time
from typing import Tuple, List, Optional, Dict
import math


class VisualEffects:
    """Handles animated visual effects for the GUI."""
    
    def __init__(self):
        """Initialize effects system."""
        self.start_time = time.time()
    
    def get_pulse_alpha(self, frequency: float = 2.0) -> float:
        """
        Get pulsing alpha value for animations.
        
        Args:
            frequency: Pulses per second
        
        Returns:
            Alpha value between 0.3 and 1.0
        """
        t = time.time() - self.start_time
        pulse = (math.sin(t * frequency * 2 * math.pi) + 1) / 2  # 0 to 1
        return 0.3 + 0.7 * pulse  # 0.3 to 1.0
    
    def draw_pulsing_box(self, 
                        image: np.ndarray,
                        center: Tuple[int, int],
                        width: int,
                        height: int,
                        color: Tuple[int, int, int] = (0, 255, 0),
                        thickness: int = 3) -> np.ndarray:
        """
        Draw a pulsing box outline.
        
        Args:
            image: Image to draw on
            center: (x, y) center point
            width: Box width in pixels
            height: Box height in pixels
            color: RGB color
            thickness: Line thickness
        
        Returns:
            Image with pulsing box
        """
        alpha = self.get_pulse_alpha(frequency=2.0)
        
        # Calculate corners
        x1 = int(center[0] - width / 2)
        y1 = int(center[1] - height / 2)
        x2 = int(center[0] + width / 2)
        y2 = int(center[1] + height / 2)
        
        # Create overlay
        overlay = image.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)
        
        # Blend with alpha
        result = cv2.addWeighted(image, 1 - alpha * 0.5, overlay, alpha * 0.5, 0)
        
        return result
    
    def draw_3d_box_projection(self,
                               image: np.ndarray,
                               center: Tuple[int, int],
                               width: int,
                               height: int,
                               depth_offset: int = 30,
                               color: Tuple[int, int, int] = (0, 255, 0),
                               alpha: float = 0.7) -> np.ndarray:
        """
        Draw a 3D-style box projection (isometric view).
        
        Args:
            image: Image to draw on
            center: (x, y) center point
            width: Box width in pixels
            height: Box height in pixels
            depth_offset: Pixel offset for depth effect
            color: RGB color
            alpha: Transparency (0-1)
        
        Returns:
            Image with 3D box
        """
        cx, cy = center
        w2 = width // 2
        h2 = height // 2
        
        # Front face corners
        front_tl = (cx - w2, cy - h2)
        front_tr = (cx + w2, cy - h2)
        front_bl = (cx - w2, cy + h2)
        front_br = (cx + w2, cy + h2)
        
        # Back face corners (offset for depth)
        back_tl = (cx - w2 + depth_offset, cy - h2 - depth_offset)
        back_tr = (cx + w2 + depth_offset, cy - h2 - depth_offset)
        back_bl = (cx - w2 + depth_offset, cy + h2 - depth_offset)
        back_br = (cx + w2 + depth_offset, cy + h2 - depth_offset)
        
        # Create overlay
        overlay = image.copy()
        
        # Draw back face (lighter)
        back_color = tuple(int(c * 0.6) for c in color)
        cv2.line(overlay, back_tl, back_tr, back_color, 2)
        cv2.line(overlay, back_tr, back_br, back_color, 2)
        cv2.line(overlay, back_br, back_bl, back_color, 2)
        cv2.line(overlay, back_bl, back_tl, back_color, 2)
        
        # Draw connecting lines
        cv2.line(overlay, front_tl, back_tl, color, 2)
        cv2.line(overlay, front_tr, back_tr, color, 2)
        cv2.line(overlay, front_bl, back_bl, color, 2)
        cv2.line(overlay, front_br, back_br, color, 2)
        
        # Draw front face (brighter)
        cv2.line(overlay, front_tl, front_tr, color, 3)
        cv2.line(overlay, front_tr, front_br, color, 3)
        cv2.line(overlay, front_br, front_bl, color, 3)
        cv2.line(overlay, front_bl, front_tl, color, 3)
        
        # Blend
        result = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)
        
        return result
    
    def draw_clearance_arrows(self,
                             image: np.ndarray,
                             center: Tuple[int, int],
                             box_size: Tuple[int, int],
                             clearances: Dict[str, float],
                             scale: float = 100.0,
                             color: Tuple[int, int, int] = (255, 255, 0)) -> np.ndarray:
        """
        Draw arrows showing clearance distances.
        
        Args:
            image: Image to draw on
            center: (x, y) center of box
            box_size: (width, height) of box in pixels
            clearances: Dict with 'front', 'back', 'left', 'right' in meters
            scale: Pixels per meter
            color: Arrow color
        
        Returns:
            Image with clearance arrows
        """
        cx, cy = center
        bw, bh = box_size
        
        overlay = image.copy()
        
        # Front arrow (up)
        if clearances.get('front', 0) > 0:
            length = int(clearances['front'] * scale)
            start = (cx, cy - bh // 2)
            end = (cx, cy - bh // 2 - length)
            cv2.arrowedLine(overlay, start, end, color, 2, tipLength=0.3)
            # Label
            label = f"{clearances['front']*100:.1f}cm"
            cv2.putText(overlay, label, (end[0] + 5, end[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Back arrow (down)
        if clearances.get('back', 0) > 0:
            length = int(clearances['back'] * scale)
            start = (cx, cy + bh // 2)
            end = (cx, cy + bh // 2 + length)
            cv2.arrowedLine(overlay, start, end, color, 2, tipLength=0.3)
            label = f"{clearances['back']*100:.1f}cm"
            cv2.putText(overlay, label, (end[0] + 5, end[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Left arrow
        if clearances.get('left', 0) > 0:
            length = int(clearances['left'] * scale)
            start = (cx - bw // 2, cy)
            end = (cx - bw // 2 - length, cy)
            cv2.arrowedLine(overlay, start, end, color, 2, tipLength=0.3)
            label = f"{clearances['left']*100:.1f}cm"
            cv2.putText(overlay, label, (end[0], end[1] - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Right arrow
        if clearances.get('right', 0) > 0:
            length = int(clearances['right'] * scale)
            start = (cx + bw // 2, cy)
            end = (cx + bw // 2 + length, cy)
            cv2.arrowedLine(overlay, start, end, color, 2, tipLength=0.3)
            label = f"{clearances['right']*100:.1f}cm"
            cv2.putText(overlay, label, (end[0], end[1] - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        result = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
        return result
    
    def draw_status_indicator(self,
                            image: np.ndarray,
                            position: Tuple[int, int],
                            status: str,
                            icon_size: int = 60) -> np.ndarray:
        """
        Draw status indicator (checkmark or X).
        
        Args:
            image: Image to draw on
            position: (x, y) center position
            status: 'success', 'fail', or 'processing'
            icon_size: Size of icon in pixels
        
        Returns:
            Image with status icon
        """
        x, y = position
        overlay = image.copy()
        
        if status == 'success':
            # Green checkmark
            color = (0, 255, 0)
            thickness = 5
            # Draw checkmark
            p1 = (x - icon_size // 3, y)
            p2 = (x - icon_size // 6, y + icon_size // 3)
            p3 = (x + icon_size // 2, y - icon_size // 3)
            cv2.line(overlay, p1, p2, color, thickness)
            cv2.line(overlay, p2, p3, color, thickness)
            
        elif status == 'fail':
            # Red X
            color = (0, 0, 255)
            thickness = 5
            offset = icon_size // 3
            cv2.line(overlay, (x - offset, y - offset), (x + offset, y + offset), color, thickness)
            cv2.line(overlay, (x + offset, y - offset), (x - offset, y + offset), color, thickness)
            
        elif status == 'processing':
            # Rotating spinner
            color = (255, 255, 0)
            angle = (time.time() * 180) % 360  # Rotate 180 deg/sec
            radius = icon_size // 2
            
            # Draw arc
            cv2.ellipse(overlay, (x, y), (radius, radius), 
                       int(angle), 0, 270, color, 5)
        
        result = cv2.addWeighted(image, 0.6, overlay, 0.4, 0)
        return result
    
    def draw_progress_bar(self,
                         image: np.ndarray,
                         position: Tuple[int, int],
                         width: int,
                         progress: float,
                         label: str = "") -> np.ndarray:
        """
        Draw a progress bar.
        
        Args:
            image: Image to draw on
            position: (x, y) top-left corner
            width: Bar width in pixels
            progress: Progress value (0.0 to 1.0)
            label: Optional text label
        
        Returns:
            Image with progress bar
        """
        x, y = position
        height = 30
        
        # Background
        cv2.rectangle(image, (x, y), (x + width, y + height), (50, 50, 50), -1)
        cv2.rectangle(image, (x, y), (x + width, y + height), (255, 255, 255), 2)
        
        # Progress fill
        fill_width = int(width * progress)
        if fill_width > 0:
            cv2.rectangle(image, (x, y), (x + fill_width, y + height), (0, 255, 0), -1)
        
        # Label
        if label:
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            text_x = x + (width - text_size[0]) // 2
            text_y = y + (height + text_size[1]) // 2
            cv2.putText(image, label, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return image
    
    def draw_angle_indicator(self,
                           image: np.ndarray,
                           position: Tuple[int, int],
                           angle: float,
                           threshold: float = 15.0) -> np.ndarray:
        """
        Draw camera angle indicator.
        
        Args:
            image: Image to draw on
            position: (x, y) center position
            angle: Current angle in degrees (0 = horizontal)
            threshold: Acceptable angle deviation in degrees
        
        Returns:
            Image with angle indicator
        """
        x, y = position
        size = 80
        
        # Determine color based on angle
        if abs(angle) <= threshold:
            color = (0, 255, 0)  # Green - good
            status = "GOOD"
        else:
            color = (0, 0, 255)  # Red - adjust
            status = "ADJUST"
        
        # Draw circle
        cv2.circle(image, (x, y), size, (255, 255, 255), 2)
        
        # Draw angle line
        line_length = size - 10
        angle_rad = math.radians(angle)
        end_x = int(x + line_length * math.cos(angle_rad - math.pi / 2))
        end_y = int(y + line_length * math.sin(angle_rad - math.pi / 2))
        cv2.line(image, (x, y), (end_x, end_y), color, 3)
        
        # Draw horizontal reference
        cv2.line(image, (x - size, y), (x + size, y), (150, 150, 150), 1)
        
        # Draw status text
        cv2.putText(image, status, (x - 30, y + size + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw angle value
        angle_text = f"{abs(angle):.1f}°"
        cv2.putText(image, angle_text, (x - 25, y + size + 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return image
    
    def create_info_panel(self,
                         width: int,
                         height: int,
                         title: str,
                         lines: List[str],
                         background_color: Tuple[int, int, int] = (40, 40, 40)) -> np.ndarray:
        """
        Create an info panel overlay.
        
        Args:
            width: Panel width
            height: Panel height
            title: Panel title
            lines: List of text lines to display
            background_color: RGB background color
        
        Returns:
            Panel image
        """
        panel = np.full((height, width, 3), background_color, dtype=np.uint8)
        
        # Draw border
        cv2.rectangle(panel, (0, 0), (width - 1, height - 1), (255, 255, 255), 2)
        
        # Draw title
        cv2.rectangle(panel, (0, 0), (width, 40), (60, 60, 60), -1)
        cv2.putText(panel, title, (10, 27),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Draw lines
        y_offset = 60
        for line in lines:
            cv2.putText(panel, line, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 25
        
        return panel

