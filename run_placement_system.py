"""
SPO-T: Surface Placement Optimization with Tessellation
=======================================================

Launcher script for the interactive placement GUI.

Usage:
    python run_placement_system.py
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from gui.placement_gui import main

if __name__ == "__main__":
    main()

