# SPO-T Feature Comparison

## Two Applications, One Goal: Desk Space Management

### 📦 main.py - Package Placement Optimizer

**Use Case:** "Where should I place this package on my desk?"

**Features:**
- ✅ Finds single optimal placement location
- ✅ Considers package dimensions (30cm × 20cm configurable)
- ✅ Maximizes clearance from obstacles
- ✅ Real-time visual feedback with placement marker
- ✅ Saves snapshots on demand

**Best For:**
- Robotic arms needing precise placement coordinates
- AR applications showing where to place objects
- One-time placement decisions

**Output:**
```
Visual: Green crosshair marker showing optimal location
Console: "[K-PLACE] ESC to quit, S to save a snapshot."
Files: Color and depth snapshots (on 'S' press)
```

---

### 📊 desk_monitor.py - Free Space Detector

**Use Case:** "How much free space do I have on my desk?"

**Features:**
- ✅ Continuous monitoring of all free space
- ✅ Percentage and area metrics (m²)
- ✅ Multiple free region detection and ranking
- ✅ Clutter alerts when >70% occupied
- ✅ Usage heatmap showing frequently occupied areas
- ✅ Time-series data logging (CSV)
- ✅ Visual overlay showing all free space (green dots)

**Best For:**
- Workspace organization and monitoring
- Long-term desk usage analytics
- Clutter detection and alerts
- Research on workspace utilization

**Output:**
```
Visual: 
  - Main view: Green overlay showing all free space
  - Grid view: Occupancy map
  - Heatmap: Usage patterns over time (optional, toggle with 'H')

Console:
  Free Space: 62.3%
  Free Area: 0.285 m²
  Occupied: 0.172 m²
  Regions: 3
  ⚠️ [ALERT] Desk is 71.2% occupied - consider clearing space!

Files:
  - desk_monitor_YYYYMMDD_HHMMSS.csv (time-series data)
  - desk_monitor_heatmap_YYYYMMDD_HHMMSS.png (usage heatmap)
  - desk_snapshot_YYYYMMDD_HHMMSS.png (current view)
```

**Example CSV Output:**
```csv
timestamp,free_percentage,free_area_m2,occupied_area_m2
1699123456.789,65.3,0.298,0.158
1699123457.012,64.8,0.296,0.161
1699123457.245,63.2,0.289,0.168
```

---

## Comparison Table

| Feature | main.py | desk_monitor.py |
|---------|---------|-----------------|
| **Primary Goal** | Find optimal placement | Detect free space |
| **Output Type** | Single point | All free regions |
| **Package Dimensions** | ✅ Required | ❌ N/A |
| **Safety Margin** | ✅ Yes | ❌ No (just free/occupied) |
| **Free Space %** | ❌ No | ✅ Yes |
| **Area Metrics (m²)** | ❌ No | ✅ Yes |
| **Multiple Regions** | ❌ No | ✅ Yes (sorted by size) |
| **Time-Series Logging** | ❌ No | ✅ Yes (CSV) |
| **Usage Heatmap** | ❌ No | ✅ Yes |
| **Clutter Alerts** | ❌ No | ✅ Yes (>70% threshold) |
| **Placement Marker** | ✅ Yes (green cross) | ❌ No |
| **Feasibility Heatmap** | ✅ Yes | ❌ No |
| **Visual Overlay** | ❌ No | ✅ Yes (green dots) |
| **Frame Rate** | ~30 fps | ~25 fps (more processing) |

---

## Which One Should I Use?

### Use **main.py** if you need to:
- 🤖 Control a robotic arm to place objects
- 📍 Get precise 3D coordinates for placement
- 🎯 Find the single best location considering package size
- 🖼️ Display AR markers for human guidance

### Use **desk_monitor.py** if you need to:
- 📊 Monitor workspace utilization over time
- 🔔 Get alerts when desk becomes cluttered
- 📈 Analyze desk usage patterns
- 🗺️ See all available free space at once
- 💾 Log historical data for analysis

### Use **both** if you want:
- Complete workspace management system
- Placement optimization + usage analytics
- Run them simultaneously on different displays

---

## Technical Details

### Shared Components
Both applications share the same core algorithms:
- RANSAC plane detection
- Depth to 3D point cloud conversion
- Plane coordinate transformation
- Occupancy grid creation

### Key Differences

**main.py:**
- Uses **erosion** with package dimensions to find feasible areas
- Uses **distance transform** to score feasibility
- Selects **argmax** for single best location
- Projects result to color camera for visualization

**desk_monitor.py:**
- Uses **connected components** to find separate regions
- Calculates **area metrics** for each region
- **Accumulates** occupancy over time for heatmap
- Provides **statistical analysis** of desk usage

---

## Performance Notes

- Both run at real-time speeds (25-30 fps)
- desk_monitor.py slightly slower due to additional processing
- Memory usage increases over time in desk_monitor.py (heatmap accumulation)
- Both benefit from GPU acceleration (if available via OpenCV)

---

## Future Enhancements

### Potential Additions:
1. **Object Classification**: Identify what objects are on the desk
2. **3D Visualization**: WebGL/Three.js interface showing desk in 3D
3. **Multi-Desk Support**: Monitor multiple desks simultaneously
4. **Integration API**: REST API for querying desk state
5. **Mobile App**: View desk status from phone
6. **Notifications**: Email/Slack alerts when desk is cluttered
7. **ML Predictions**: Predict when desk will become full
8. **Timelapse**: Generate video of desk usage over day/week

---

## Example Use Cases

### Research Lab
- **main.py**: Robot arm places lab equipment in optimal locations
- **desk_monitor.py**: Lab manager monitors bench space utilization

### Office Workspace
- **main.py**: AR headset shows where to place incoming packages
- **desk_monitor.py**: Employee tracks personal workspace organization

### Manufacturing
- **main.py**: Automated system places parts on assembly table
- **desk_monitor.py**: Monitors workstation capacity and bottlenecks

### Smart Home
- **desk_monitor.py**: Reminds user to clean desk when >80% full
- **main.py**: Shows where to place new items when restocking

---

## Contact & Contributions

For questions, issues, or feature requests, please open an issue on GitHub.

Contributions welcome! See CONTRIBUTING.md for guidelines.

