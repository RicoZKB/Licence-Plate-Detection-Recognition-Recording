# Quick Start Guide

## TL;DR - Just Run This

```bash
python3 per9_direction_based.py
```

Press **'o'** to flip IN/OUT direction. Press **ESC** to quit.

---

## The Problem You Had

❌ Exit events not working when pressing 'o' to flip gate side

## The Solution

✅ Three versions provided:
- **per7_format.py** - Simple (force exit all on flip)
- **per8_optimized.py** - Smart (recalculate states on flip)
- **per9_direction_based.py** ⭐ **RECOMMENDED** (no state issues at all)

---

## Why per9 is Best

### Old Design (per7/per8)
```
Problem: OCR tied to gate side logic
         ↓
Flip gate side → OCR zones flip → State confusion → No exit events
```

### New Design (per9) ⭐
```
Solution: OCR independent of gate
          ↓
OCR everywhere → Track trajectory → Detect crossing → Log event
                                                      ↓
                                     Flip just changes IN/OUT mapping
```

---

## Visual Comparison

### Scenario: Vehicle crosses left to right

#### per7_format.py (Simple Reset)
```
Before:  [Vehicle]────→ ║ Gate ║
Press 'o' → Force EXIT all vehicles → Clear state
After:   Vehicle must re-enter to be tracked
```

#### per8_optimized.py (Smart Recalc)
```
Before:  [Vehicle]────→ ║ Gate ║
Press 'o' → Check position → EXIT if on wrong side
After:   Smart handling based on current position
```

#### per9_direction_based.py ⭐ (Direction-Based)
```
Before:  [Vehicle]────→ ║ Gate ║
Press 'o' → Only flip direction mapping
After:   [Vehicle]────→ ║ Gate ║ (tracking continues, no disruption)
```

---

## Configuration

Open [per9_direction_based.py](per9_direction_based.py) and edit:

```python
# Your video file
INPUT_VIDEO_PATH = "input_videos/Trim03.mp4"

# Gate position (0.0 = left, 1.0 = right)
GATE_LINE_X_RATIO = 0.55

# Direction mapping
LEFT_TO_RIGHT_IS_IN = True  # True: L→R is IN, R→L is OUT
                             # False: R→L is IN, L→R is OUT
```

---

## What You'll See

### On Screen
- Gate line drawn vertically
- Bounding boxes around plates (unique colors)
- Track history trails
- "IN" or "OUT" labels when events occur
- FPS counter

### In Terminal
```
[GATE] tracker_id=5 crossed left_to_right → IN, cx=420
[2025-10-26 14:32:15] IN   slot=    01  region_class='横浜310'  suffix='12-34'

[Press 'o']
[DIRECTION FLIP] IN=right→left, OUT=left→right

[GATE] tracker_id=5 crossed right_to_left → OUT, cx=280
[2025-10-26 14:32:42] OUT  slot=    01  region_class='横浜310'  suffix='12-34'
```

### In CSV (logs/parking_log_YYYYMMDD.csv)
```csv
timestamp,object_id,vehicle_type,direction,city,engine_size,kana,four-digit number
2025-10-26 14:32:15,01,car,in,横浜,310,あ,12-34
2025-10-26 14:32:42,01,car,out,横浜,310,あ,12-34
```

---

## Key Features

### ✅ Full-Frame OCR
OCR processes plates **anywhere** in the frame - not restricted to zones

### ✅ ByteTrack Integration
Robust tracking with unique IDs and trajectory history

### ✅ Direction-Based Events
- Left→Right crossing? Check direction mapping → Log IN or OUT
- Right→Left crossing? Check direction mapping → Log opposite

### ✅ No State Confusion
Press 'o' anytime - just flips direction mapping, no disruption

### ✅ Continuous Capture
Best OCR captured throughout tracking, not just at gate

---

## Controls

| Key | Action |
|-----|--------|
| **'o'** | Flip IN/OUT direction mapping |
| **ESC** | Quit application |

---

## Files Overview

### Implementations
1. **per7_format.py** - Your original with simple reset fix
2. **per8_optimized.py** - Your original with smart recalculation fix
3. **per9_direction_based.py** ⭐ - New architecture (recommended)

### Documentation
- **QUICK_START.md** ← You are here
- **SOLUTION_SUMMARY.md** - Complete overview
- **DIRECTION_BASED_APPROACH.md** - per9 technical details
- **GATE_FLIP_FIX.md** - per7/per8 technical details
- **SOLUTION_COMPARISON.md** - per7 vs per8 comparison

### Testing
- **test_gate_flip.sh** - Interactive menu to test all versions

---

## Common Questions

### Q: Which version should I use?
**A:** Use **per9_direction_based.py** - it's the cleanest solution.

### Q: Will my existing video work?
**A:** Yes! Just set `INPUT_VIDEO_PATH` to your video file.

### Q: How do I know which way is IN?
**A:** Look at your camera view. Vehicles entering should cross in one consistent direction. Set `LEFT_TO_RIGHT_IS_IN` accordingly.

### Q: Can I change direction during runtime?
**A:** Yes! Press 'o' anytime - per9 handles it perfectly with no disruption.

### Q: What if OCR is poor?
**A:** Adjust these in the file:
- `MIN_SHARPNESS_LOCK = 60.0` (lower for easier acceptance)
- `CAPTURE_WINDOW_FRAMES = 8` (increase for more samples)
- `MIN_PLATE_SIZE_PX = 18` (lower to detect smaller plates)

---

## Performance Tips

### For Live Camera
```python
USE_CAMERA = True
USE_FILE = False
DETECT_EVERY_N = 2  # Process every 2nd frame for speed
```

### For High-Res Video
```python
TARGET_WIDTH = 1280  # Higher resolution
DETECT_EVERY_N = 1   # Process every frame for accuracy
```

### For Speed
```python
TARGET_WIDTH = 640   # Lower resolution
DETECT_EVERY_N = 2   # Skip frames
WRITE_VIDEO = False  # Don't write output
```

---

## Next Steps

1. **Run it**: `python3 per9_direction_based.py`
2. **Test flip**: Press 'o' and verify events work correctly
3. **Check CSV**: Look in `logs/parking_log_YYYYMMDD.csv`
4. **Adjust settings**: Fine-tune gate position and thresholds
5. **Deploy**: Use in production!

---

## Success Criteria

You'll know it's working when:
- ✅ Vehicles crossing in one direction log IN events
- ✅ Same vehicles crossing back log OUT events
- ✅ Pressing 'o' flips which direction is IN vs OUT
- ✅ CSV file contains both IN and OUT events
- ✅ Slot numbers are reused properly

---

## Still Having Issues?

1. Check gate line is visible with `DRAW_GATE = True`
2. Verify gate position with `GATE_LINE_X_RATIO`
3. Try adjusting `GATE_CROSSING_THRESHOLD` (15 is default)
4. Check terminal for `[GATE]` crossing messages
5. Verify ByteTrack is working (colored boxes on plates)

---

## That's It!

You now have three working solutions, with **per9_direction_based.py** being the recommended one.

**Start here**:
```bash
python3 per9_direction_based.py
```

Everything else is in the documentation files! 🚀
