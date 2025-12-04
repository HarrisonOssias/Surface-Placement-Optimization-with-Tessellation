"""Core modules for desk and box detection."""

from .box_detector import BoxDetector, DetectedBox
from .desk_monitor import DeskSpaceAnalyzer

__all__ = ['BoxDetector', 'DetectedBox', 'DeskSpaceAnalyzer']

