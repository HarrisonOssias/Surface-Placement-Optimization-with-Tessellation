"""
Quick Setup Test
================

Verifies that all dependencies are installed and Kinect is accessible.
Run this before using main.py or desk_monitor.py.

Usage:
    python test_setup.py
"""

import sys

print("=" * 70)
print("SPO-T Setup Test")
print("=" * 70)
print()

# Test 1: Python Version
print("1. Testing Python version...")
version = sys.version_info
print(f"   Python {version.major}.{version.minor}.{version.micro}")
if version.major == 3 and version.minor >= 8:
    print("   ✓ Python version OK")
else:
    print(f"   ⚠ Warning: Python 3.8+ recommended, you have {version.major}.{version.minor}")
print()

# Test 2: Import numpy
print("2. Testing numpy...")
try:
    import numpy as np
    print(f"   NumPy version: {np.__version__}")
    print("   ✓ NumPy OK")
except ImportError as e:
    print(f"   ✗ NumPy not found: {e}")
    print("   Install: pip install numpy")
    sys.exit(1)
print()

# Test 3: Import OpenCV
print("3. Testing OpenCV...")
try:
    import cv2
    print(f"   OpenCV version: {cv2.__version__}")
    print("   ✓ OpenCV OK")
except ImportError as e:
    print(f"   ✗ OpenCV not found: {e}")
    print("   Install: pip install opencv-python")
    sys.exit(1)
print()

# Test 4: Import ctypes
print("4. Testing ctypes...")
try:
    import ctypes
    print("   ✓ ctypes OK")
except ImportError as e:
    print(f"   ✗ ctypes not found: {e}")
    sys.exit(1)
print()

# Test 5: Import pykinect2
print("5. Testing pykinect2...")
try:
    from pykinect2 import PyKinectRuntime, PyKinectV2
    print("   ✓ pykinect2 imported successfully")
    print()
    print("   Note: If you saw errors about 'AssertionError: 80' or")
    print("         'ImportError: Wrong version', run:")
    print("         python setup_pykinect2.py")
except ImportError as e:
    print(f"   ✗ pykinect2 not found: {e}")
    print("   Install: pip install pykinect2")
    sys.exit(1)
except AssertionError as e:
    print(f"   ✗ pykinect2 has compatibility issue: {e}")
    print("   Fix: python setup_pykinect2.py")
    sys.exit(1)
print()

# Test 6: Try to initialize Kinect (optional)
print("6. Testing Kinect connection (optional)...")
print("   This will attempt to connect to your Kinect sensor.")
print("   Press Enter to test, or 's' to skip: ", end='')
response = input().strip().lower()

if response != 's':
    try:
        print("   Initializing Kinect...")
        k = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Depth)
        print("   ✓ Kinect initialized successfully!")
        print("   ✓ Kinect sensor is connected and working")
        
        # Try to get a frame
        import time
        print("   Waiting for depth frame...")
        timeout = time.time() + 5
        got_frame = False
        while time.time() < timeout:
            if k.has_new_depth_frame():
                frame = k.get_last_depth_frame()
                print(f"   ✓ Received depth frame ({len(frame)} points)")
                got_frame = True
                break
            time.sleep(0.1)
        
        if not got_frame:
            print("   ⚠ Warning: Kinect initialized but no frames received")
            print("              Check that Kinect is plugged in and powered")
        
    except Exception as e:
        print(f"   ⚠ Could not initialize Kinect: {e}")
        print("   Possible causes:")
        print("   - Kinect not plugged in")
        print("   - Kinect SDK 2.0 not installed")
        print("   - USB 3.0 port required")
        print("   - Another program using Kinect")
else:
    print("   Skipped Kinect test")
print()

# Summary
print("=" * 70)
print("Setup Test Complete!")
print("=" * 70)
print()
print("Next steps:")
print("1. If any tests failed, install missing dependencies")
print("2. If pykinect2 has issues, run: python setup_pykinect2.py")
print("3. Connect your Kinect v2 sensor")
print("4. Run the applications:")
print("   - python main.py          (package placement)")
print("   - python desk_monitor.py  (free space detection)")
print()
print("For more information, see README.md")
print("=" * 70)

