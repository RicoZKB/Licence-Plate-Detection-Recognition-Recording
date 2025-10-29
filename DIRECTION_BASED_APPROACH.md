# Direction-Based Parking System (per9_direction_based.py)

## Key Innovation: Decouple OCR from Gate Logic

### The Problem with Previous Versions
- OCR was restricted to "inside" zone based on `gate_right` setting
- Confusing logic: "inside/outside" mixed with OCR capture
- Gate flip caused tracking confusion
- OCR performance varied by zone

### The New Approach

**Core Principle**: OCR and direction detection are completely independent.

```
┌─────────────────────────────────────────────┐
│                                             │
│  ByteTrack: Tracks plates EVERYWHERE        │
│  OCR: Processes THROUGHOUT entire frame     │
│  Gate: Only determines crossing DIRECTION   │
│                                             │
└─────────────────────────────────────────────┘
```

## Architecture

### 1. Full-Frame Tracking & OCR
- **ByteTrack** tracks all detected plates across the entire frame
- **OCR processes** anywhere a plate is detected (no zone restrictions)
- **Continuous capture**: Best OCR is captured throughout tracking

### 2. Direction-Based Events
- **Gate line**: A vertical line at configurable position (default 55% of width)
- **Crossing detection**: Analyzes trajectory to detect gate crossing
- **Direction mapping**: Simple toggle determines which direction = IN vs OUT

### 3. No "Inside/Outside" Concept
- No confusing `gate_right` or "inside" zones
- Only two things matter:
  1. Did vehicle cross the gate line? (trajectory analysis)
  2. Which direction did it cross? (left→right or right→left)

## Configuration

```python
# Gate position (vertical line)
GATE_LINE_X_RATIO = 0.55  # 55% from left edge

# Direction mapping (simple toggle)
LEFT_TO_RIGHT_IS_IN = True  # True: L→R=IN, R→L=OUT
                             # False: R→L=IN, L→R=OUT
```

## How It Works

### Step-by-Step Flow

1. **Detection**: Plate detected anywhere in frame
2. **Tracking**: ByteTrack assigns tracker_id, builds trajectory
3. **OCR Capture**: System continuously captures best OCR for this tracker
4. **Crossing Detection**: When trajectory crosses gate line, determine direction
5. **Event Logging**:
   - If direction matches IN: Log IN event with best OCR captured
   - If direction matches OUT: Log OUT event, free slot

### Example Scenario

```
Frame Layout:
═════════════════════════════════════════════════════════
                              ║ GATE
     Left Side               ║              Right Side
                              ║
         [Car A] ───→         ║
                              ║         ←─── [Car B]
═════════════════════════════════════════════════════════

With LEFT_TO_RIGHT_IS_IN = True:
- Car A crossing left→right: IN event
- Car B crossing right→left: OUT event

Press 'o' to flip:
With LEFT_TO_RIGHT_IS_IN = False:
- Car A crossing left→right: OUT event
- Car B crossing right→left: IN event
```

## Key Advantages

### 1. OCR Performance
✅ **Works everywhere**: No zone restrictions
✅ **Consistent quality**: Same processing throughout frame
✅ **Continuous improvement**: Captures best OCR across entire trajectory

### 2. Simplified Logic
✅ **No confusing "inside/outside"**: Just directions
✅ **Clear semantics**: left→right vs right→left
✅ **Easy to flip**: Single boolean toggle

### 3. Robust Gate Flip
✅ **No state confusion**: Only direction mapping changes
✅ **No forced exits**: Vehicles continue tracking normally
✅ **Instant switch**: Press 'o' anytime without disruption

### 4. Better Tracking
✅ **Full-frame detection**: Catches plates anywhere
✅ **Longer trajectories**: More data for direction analysis
✅ **Better accuracy**: Direction determined from trajectory, not single position

## Comparison: Old vs New

| Aspect | Old Approach (per7/per8) | New Approach (per9) |
|--------|--------------------------|---------------------|
| **OCR Zone** | Restricted to ROI or gate vicinity | Full frame |
| **Direction Logic** | Based on `gate_right` + position | Based on trajectory crossing |
| **Gate Flip** | Requires state reset/recalculation | Just flips direction mapping |
| **Complexity** | High (inside/outside/capture zones) | Low (just track + cross + log) |
| **OCR Quality** | Varies by zone | Consistent everywhere |
| **Edge Cases** | Many (zone boundaries, state conflicts) | Few (just crossing detection) |

## Usage

### Running the System

```bash
python3 per9_direction_based.py
```

### Controls

- **'o'**: Flip IN/OUT direction mapping (no disruption)
- **ESC**: Quit

### Console Output

```
[INFO] Direction mapping: left→right = IN, right→left = OUT
[INFO] Press 'o' to flip IN/OUT direction, 'm' for mode, ESC to quit

[GATE] tracker_id=5 crossed left_to_right → IN, cx=420
[2025-10-26 14:32:15] IN   slot=    01  region_class='横浜310'  suffix='12-34'  kana='あ'  city='横浜'  raw="横浜 310 あ 12-34"

[GATE] tracker_id=5 crossed right_to_left → OUT, cx=280
[2025-10-26 14:32:42] OUT  slot=    01  region_class='横浜310'  suffix='12-34'  kana='あ'  city='横浜'  raw="横浜 310 あ 12-34"
```

### Flipping Direction

Press 'o' during runtime:

```
[DIRECTION FLIP] IN=right→left, OUT=left→right
[INFO] Direction mapping changed. Tracking continues normally.
```

## Configuration Tips

### Setting Gate Position

```python
GATE_LINE_X_RATIO = 0.55  # Adjust based on your camera view
                           # 0.0 = left edge, 1.0 = right edge
```

### Choosing Direction Mapping

```python
# For entry on left, exit on right:
LEFT_TO_RIGHT_IS_IN = True

# For entry on right, exit on left:
LEFT_TO_RIGHT_IS_IN = False
```

### Fine-Tuning

```python
# Crossing detection sensitivity
GATE_CROSSING_THRESHOLD = 15  # pixels (lower = more sensitive)

# Minimum frames between events for same vehicle
MIN_FRAMES_BETWEEN_EVENTS = 10  # prevents duplicate events

# OCR capture duration
CAPTURE_WINDOW_FRAMES = 8  # frames to find best OCR

# Minimum sharpness to accept OCR immediately
MIN_SHARPNESS_LOCK = 60.0  # Laplacian variance
```

## Why This Approach is Better

### 1. Matches Real-World Usage
In real parking systems, you don't care about "zones" - you care about:
- Did a vehicle enter? (crossed gate going IN direction)
- Did a vehicle exit? (crossed gate going OUT direction)

### 2. OCR Independence
OCR quality should not depend on arbitrary zones. A plate is a plate, wherever it is.

### 3. Simpler Mental Model
```
Old: "Is vehicle inside? What's gate_right? Which side is inside? Flip state?"
New: "Which way did it cross? Is that IN or OUT?"
```

### 4. Production-Ready
- No edge cases with zone boundaries
- No state confusion on configuration changes
- Continuous OCR improvement throughout tracking
- Clean, maintainable code

## Migration from per7/per8

If you're using per7_format.py or per8_optimized.py:

**Replace**:
```python
USE_ROI = False
GATE_INSIDE_IS_RIGHT = True  # Confusing!
```

**With**:
```python
LEFT_TO_RIGHT_IS_IN = True  # Clear and simple!
```

The rest works automatically - no zone management needed.

## Troubleshooting

### Issue: Events not detected
- Check `GATE_LINE_X_RATIO` - gate might be off-screen
- Adjust `GATE_CROSSING_THRESHOLD` - might be too strict
- Check `DRAW_GATE` is True to visualize gate line

### Issue: Duplicate events
- Increase `MIN_FRAMES_BETWEEN_EVENTS`
- Vehicle might be oscillating near gate line

### Issue: Poor OCR
- Check `MIN_PLATE_SIZE_PX` - plates might be too small
- Adjust `MIN_SHARPNESS_LOCK` threshold
- Increase `CAPTURE_WINDOW_FRAMES` for more samples

## Future Enhancements

Possible additions:
1. **Dual gates**: Separate entry/exit gates
2. **Speed detection**: Calculate vehicle speed from trajectory
3. **Dwell time**: Track how long between IN and OUT
4. **Multi-lane**: Multiple gate lines for different lanes
5. **Advanced analytics**: Heat maps, traffic patterns

## Recommendation

**Use per9_direction_based.py for**:
- ✅ New projects
- ✅ Bidirectional gates
- ✅ Production deployments
- ✅ When OCR quality matters throughout
- ✅ Simplified configuration and operation

This is the cleanest, most maintainable approach!
