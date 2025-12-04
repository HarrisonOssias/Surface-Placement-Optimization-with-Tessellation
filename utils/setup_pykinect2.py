"""
Setup script to patch pykinect2 for Python 3.8+ compatibility
Run this once after installing pykinect2 to fix compatibility issues.

Usage:
    python setup_pykinect2.py

Issues fixed:
- Structure size assertion error on 64-bit Python
- Comtypes version check error
- time.clock() removed in Python 3.8+
"""
import os
import sys
import shutil

def find_pykinect2_files():
    """Find the pykinect2 installation files."""
    try:
        import pykinect2
        package_dir = os.path.dirname(pykinect2.__file__)
        return {
            'PyKinectV2': os.path.join(package_dir, 'PyKinectV2.py'),
            'PyKinectRuntime': os.path.join(package_dir, 'PyKinectRuntime.py')
        }
    except ImportError:
        print("ERROR: pykinect2 is not installed.")
        print("Please install it first: pip install pykinect2")
        return None

def create_backup(filepath):
    """Create a backup of the original file."""
    backup_path = filepath + '.original'
    if not os.path.exists(backup_path):
        shutil.copy2(filepath, backup_path)
        print(f"  ✓ Backup created: {backup_path}")
        return True
    else:
        print(f"  - Backup already exists: {backup_path}")
        return False

def patch_pykinect_v2(filepath):
    """Patch PyKinectV2.py for compatibility."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    patches = []
    
    # Patch 1: Fix structure size assertion
    if 'assert sizeof(tagSTATSTG) == 72' in content:
        content = content.replace(
            'assert sizeof(tagSTATSTG) == 72, sizeof(tagSTATSTG)',
            'assert sizeof(tagSTATSTG) in [72, 80], sizeof(tagSTATSTG)  # Fixed for 64-bit'
        )
        patches.append("Structure size assertion (64-bit fix)")
    
    # Patch 2: Fix comtypes version check
    if "from comtypes import _check_version; _check_version('')" in content:
        content = content.replace(
            "from comtypes import _check_version; _check_version('')",
            "# from comtypes import _check_version; _check_version('')  # Disabled for compatibility"
        )
        patches.append("Comtypes version check")
    
    if content != original_content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return patches
    return []

def patch_pykinect_runtime(filepath):
    """Patch PyKinectRuntime.py for Python 3.8+ compatibility."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # Replace time.clock() with time.perf_counter()
    if 'time.clock()' in content:
        content = content.replace('time.clock()', 'time.perf_counter()')
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return ['time.clock() → time.perf_counter() (Python 3.8+)']
    return []

def main():
    print("=" * 70)
    print("PyKinect2 Compatibility Patcher for Python 3.8+ / 64-bit Systems")
    print("=" * 70)
    print()
    
    # Find files
    files = find_pykinect2_files()
    if not files:
        return 1
    
    print(f"Found pykinect2 installation:")
    for name, path in files.items():
        print(f"  - {name}: {path}")
    print()
    
    # Patch PyKinectV2.py
    print("Patching PyKinectV2.py...")
    create_backup(files['PyKinectV2'])
    patches_v2 = patch_pykinect_v2(files['PyKinectV2'])
    if patches_v2:
        for patch in patches_v2:
            print(f"  ✓ Applied: {patch}")
    else:
        print("  - Already patched or no changes needed")
    print()
    
    # Patch PyKinectRuntime.py
    print("Patching PyKinectRuntime.py...")
    create_backup(files['PyKinectRuntime'])
    patches_runtime = patch_pykinect_runtime(files['PyKinectRuntime'])
    if patches_runtime:
        for patch in patches_runtime:
            print(f"  ✓ Applied: {patch}")
    else:
        print("  - Already patched or no changes needed")
    print()
    
    # Summary
    print("=" * 70)
    if patches_v2 or patches_runtime:
        print("✓ Patching completed successfully!")
        print()
        print("Your pykinect2 installation is now compatible with:")
        print("  - Python 3.8+ (time.perf_counter)")
        print("  - 64-bit systems (structure sizes)")
        print("  - Modern comtypes versions")
    else:
        print("No patches were needed - already compatible!")
    print()
    print("You can now run your Kinect applications.")
    print("=" * 70)
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

