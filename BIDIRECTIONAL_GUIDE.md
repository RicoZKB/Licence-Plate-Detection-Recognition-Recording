# Bidirectional License Plate Detection Guide

## Overview
The system now supports **TRUE bidirectional** license plate detection and OCR. Vehicles can cross the gate from either direction, and the system will:
- Detect the crossing
- Perform OCR on the license plate
- Log IN/OUT events correctly
- Support the same vehicle crossing multiple times from different directions

## Key Changes Made

### 1. Simplified Exit Logic (Line 615-621)
**OLD**: Exit only triggered when crossing was "not an entry" based on gate orientation
```python
if not event_is_entry and oid in captured_in:  # ❌ Too restrictive
```

**NEW**: Exit triggers when ANY captured vehicle crosses in a NEW direction
```python
if oid in captured_in:  # ✅ Works bidirectionally
```

### 2. Debug Output Added
New flag: `PRINT_GATE_CROSSINGS = True` (line 30)

When enabled, you'll see:
```
[GATE] tracker_id=123 crossed left_to_right, was_captured=NO, cx=450
[GATE] tracker_id=123 crossed right_to_left, was_captured=YES, cx=800
```

## How It Works

### Scenario: Vehicle crosses multiple times

```
Frame 1-50:   Vehicle approaches from LEFT
Frame 51:     [GATE] tracker_id=5 crossed left_to_right, was_captured=NO
              → Capture window opens, OCR begins
Frame 60:     [2025-10-25 10:30:45] IN  slot=01  region_class='横浜310'...
              → Vehicle is now "captured"
Frame 100:    Vehicle returns, approaches gate from RIGHT
Frame 101:    [GATE] tracker_id=5 crossed right_to_left, was_captured=YES
              → EXIT event logged immediately
              → [2025-10-25 10:31:15] OUT slot=01...
              → Capture window reopened for fresh OCR
Frame 120:    Vehicle crosses gate again (right to left)
Frame 121:    [GATE] tracker_id=5 crossed right_to_left, was_captured=NO
              → Fresh OCR begins
Frame 130:    [2025-10-25 10:31:30] IN  slot=01...
```

## Configuration

### Gate Orientation Settings (Line 44-47)
```python
USE_ROI                  = False  # Use gate mode (not ROI box)
ENTRY_LINE_X_RATIO       = 0.55   # Gate position (0.0=left, 1.0=right)
GATE_INSIDE_IS_RIGHT     = True   # Which side is "inside"
```

### Testing Different Orientations

**Test 1: Gate with "inside" on RIGHT**
```python
GATE_INSIDE_IS_RIGHT = True
```
- Vehicle at x=400 (left of gate): OUTSIDE
- Vehicle at x=800 (right of gate): INSIDE
- L→R crossing: Vehicle enters "inside" zone
- R→L crossing: Vehicle exits "inside" zone

**Test 2: Gate with "inside" on LEFT**
```python
GATE_INSIDE_IS_RIGHT = False
```
- Vehicle at x=400 (left of gate): INSIDE
- Vehicle at x=800 (right of gate): OUTSIDE
- L→R crossing: Vehicle exits "inside" zone
- R→L crossing: Vehicle enters "inside" zone

**IMPORTANT**: Regardless of `GATE_INSIDE_IS_RIGHT` setting:
- EXIT events trigger when a captured vehicle crosses in a NEW direction
- This ensures bidirectional operation works correctly

## Runtime Controls

Press during execution:
- `m`: Toggle between ROI mode and Gate mode
- `o`: Toggle gate orientation (left/right as "inside")
- `ESC`: Quit

## Debug Flags (Lines 28-31)

```python
PRINT_EVENTS_TO_TERMINAL = True   # Show all IN/OUT events
PRINT_RAW_OCR_FOUND      = False  # Show raw OCR strings
PRINT_GATE_CROSSINGS     = True   # Show gate crossing detection ⭐ NEW
PASS_OCR_DEBUG_TO_DETECT = False  # Very verbose OCR debug
```

## Troubleshooting

### "Hardly any OUT events happen"
**Cause**: The old logic only triggered exits based on gate orientation
**Fix**: ✅ Fixed! Exits now trigger on ANY direction change for captured vehicles

### "Same vehicle can't be OCR'd twice"
**Cause**: Capture window was locked after first OCR
**Fix**: ✅ Fixed! Window reopens after each gate crossing

### "Vehicle detected but no IN event"
**Possible causes**:
1. OCR didn't find a valid plate suffix (check `PRINT_RAW_OCR_FOUND = True`)
2. Sharpness too low (check `MIN_SHARPNESS_LOCK = 60.0`)
3. Capture window timeout (check `CAPTURE_WINDOW_FRAMES = 6`)

**Debug steps**:
```python
PRINT_GATE_CROSSINGS = True    # See when crossings detected
PRINT_RAW_OCR_FOUND = True     # See what OCR finds
PRINT_EVENTS_TO_TERMINAL = True  # See IN/OUT events
```

## CSV Output Format

```csv
timestamp,object_id,vehicle_type,direction,city,engine_size,kana,four-digit number
2025-10-25 10:30:45,01,car,in,横浜,310,た,12-34
2025-10-25 10:31:15,01,car,out,横浜,310,た,12-34
2025-10-25 10:31:30,01,car,in,横浜,310,た,12-34  ← Same vehicle re-entered!
```

## Testing Recommendations

1. **Set all debug flags to True** for first test
2. **Start with one orientation** (`GATE_INSIDE_IS_RIGHT = True`)
3. **Watch the terminal output** for `[GATE]` messages
4. **Verify both directions work**:
   - Drive vehicle L→R, should see IN event
   - Drive same vehicle R→L, should see OUT event
   - Drive same vehicle L→R again, should see IN event again
5. **Toggle orientation** with `o` key during runtime
6. **Check CSV log** in `logs/parking_log_YYYYMMDD.csv`

## Performance Tips

- `DETECT_EVERY_N = 1`: Process every frame (slower but more accurate)
- `DETECT_EVERY_N = 2`: Skip every other frame (faster)
- `ROI_INFER_MAX_W = 640`: Downscale ROI for speed
- `BYTETRACK_LOST_BUFFER = 30`: How many frames to keep lost tracks

## Architecture

```
ByteTrack (supervision)
    ↓
Track History (deque)
    ↓
detect_gate_crossing() → Returns (crossed, direction)
    ↓
Gate Crossing Handler:
  - If captured: Log OUT, clear state
  - Open capture window
    ↓
OCR Capture Window:
  - Collect best frame (area × sharpness)
  - Early accept if sharp enough
    ↓
Log IN event
```

## Credits
Fixed bidirectional crossing logic on 2025-10-25
