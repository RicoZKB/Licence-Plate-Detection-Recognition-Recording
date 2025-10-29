# Gate Orientation Fix - TRUE Bidirectional Detection

## The Problem You Reported
> "there is nothing different results came out of this. every result the same in event if change gate inside by pushing o key to change"

The previous fix didn't properly use the `gate_right` variable to determine IN vs OUT. It just detected any crossing and treated the first crossing as entry, which didn't match the Roboflow workflow behavior.

## The Solution

### New Logic (Lines 614-712 in per7_format.py)

The system now **properly interprets crossing direction based on gate orientation**:

```python
if gate_right:
    is_entry_crossing = (crossing_direction == 'left_to_right')
else:
    is_entry_crossing = (crossing_direction == 'right_to_left')
```

Then it handles events based on `is_entry_crossing`:
- **ENTRY crossing** → Open capture window, perform OCR, log IN event
- **EXIT crossing** → Log OUT event, release slot, DO NOT capture

## How It Works Now

### Configuration: `gate_right = True` (Right side is "inside")

```
LEFT (outside)  |  GATE  |  RIGHT (inside parking)
────────────────────────────────────────────────────
                   ↓
         Vehicle crosses L→R
                   ↓
              [GATE] crossed left_to_right → IN
                   ↓
         Opens capture window, performs OCR
                   ↓
              [2025-10-25] IN slot=01

                   ...vehicle is now inside...

                   ↓
         Vehicle crosses R→L
                   ↓
              [GATE] crossed right_to_left → OUT
                   ↓
              [2025-10-25] OUT slot=01
```

### Configuration: `gate_right = False` (Left side is "inside")

```
LEFT (inside parking)  |  GATE  |  RIGHT (outside)
────────────────────────────────────────────────────
                          ↓
            Vehicle crosses R→L
                          ↓
                 [GATE] crossed right_to_left → IN
                          ↓
            Opens capture window, performs OCR
                          ↓
                 [2025-10-25] IN slot=01

                   ...vehicle is now inside...

                          ↓
            Vehicle crosses L→R
                          ↓
                 [GATE] crossed left_to_right → OUT
                          ↓
                 [2025-10-25] OUT slot=01
```

## Key Differences from Previous Version

### BEFORE (Buggy):
- Exit triggered when `oid in captured_in` (any direction change)
- Didn't use `gate_right` to determine event type
- Pressing 'o' key changed internal state but events stayed the same

### AFTER (Fixed):
- Entry/Exit determined by `gate_right` + `crossing_direction`
- Pressing 'o' key **immediately changes which crossings are IN vs OUT**
- Matches Roboflow workflow behavior exactly

## Debug Output

With `PRINT_GATE_CROSSINGS = True`, you'll see:

```
[GATE] tracker_id=5 crossed left_to_right → IN, was_captured=NO, cx=450, gate_right=True
[2025-10-25 10:30:45] IN  slot=01  region_class='横浜310' suffix='12-34'...
[GATE] tracker_id=5 crossed right_to_left → OUT, was_captured=YES, cx=800, gate_right=True
[2025-10-25 10:31:15] OUT slot=01(12-34)...
```

**Then press 'o' to toggle gate_right:**

```
[GATE] tracker_id=6 crossed left_to_right → OUT, was_captured=NO, cx=450, gate_right=False
[GATE] tracker_id=7 crossed right_to_left → IN, was_captured=NO, cx=800, gate_right=False
[2025-10-25 10:32:00] IN  slot=02  region_class='横浜330' suffix='56-78'...
```

## Testing Steps

1. **Start the program**:
   ```bash
   . cenv/bin/activate
   python per7_format.py
   ```

2. **Observe initial behavior** (gate_right=True by default):
   - Vehicle crossing L→R should show: `[GATE] ... → IN`
   - Vehicle crossing R→L should show: `[GATE] ... → OUT`

3. **Press 'o' key** to toggle orientation

4. **Observe flipped behavior** (now gate_right=False):
   - Vehicle crossing L→R should show: `[GATE] ... → OUT`
   - Vehicle crossing R→L should show: `[GATE] ... → IN`

5. **Check CSV log** (`logs/parking_log_YYYYMMDD.csv`):
   - Before pressing 'o': L→R generates "in" rows
   - After pressing 'o': R→L generates "in" rows

## Matching Roboflow Workflow

Your Roboflow image shows:
- **LEFT zone**: "OUT 0" (exit detection zone)
- **RIGHT zone**: "IN 1" (entry detection zone)
- **VERTICAL LINE**: Gate separator

This matches our configuration:
```python
USE_ROI = False                # Use gate mode
ENTRY_LINE_X_RATIO = 0.55      # Gate at 55% of width
GATE_INSIDE_IS_RIGHT = True    # Right side is "inside" (matches Roboflow)
```

## Implementation Details

### Entry Crossing Handler (Lines 633-674)
1. Check if vehicle was already captured (re-entry case)
2. If yes: Log automatic OUT first
3. Open capture window
4. Perform OCR over next N frames
5. Log IN event when OCR succeeds

### Exit Crossing Handler (Lines 676-712)
1. Check if vehicle was captured
2. If yes: Log OUT event, release slot
3. Clear all tracking state
4. DO NOT open capture window (no OCR needed on exit)

## Why This Fixes Your Issue

The previous code had this flaw:
```python
# OLD - didn't use gate_right properly
if oid in captured_in:
    # Handle exit (but always treated second crossing as exit)
```

The new code properly determines event type:
```python
# NEW - uses gate_right to interpret crossing
if gate_right:
    is_entry_crossing = (crossing_direction == 'left_to_right')
else:
    is_entry_crossing = (crossing_direction == 'right_to_left')

if is_entry_crossing:
    # Handle IN
else:
    # Handle OUT
```

Now pressing 'o' to toggle `gate_right` **actually changes which crossings trigger IN vs OUT events**!

## Expected Results

Run the video and you should see:
- Vehicles entering from left (L→R): IN events + OCR
- Same vehicles exiting to left (R→L): OUT events

Then press 'o' and continue watching:
- Vehicles entering from right (R→L): IN events + OCR
- Same vehicles exiting to right (L→R): OUT events

The CSV should show alternating IN/OUT patterns that change when you press 'o'.

## If Issues Persist

1. **Check debug output**: Make sure `PRINT_GATE_CROSSINGS = True`
2. **Verify gate detection**: Look for `[GATE]` messages with correct directions
3. **Check gate position**: Adjust `ENTRY_LINE_X_RATIO` if the line isn't where vehicles cross
4. **Verify crossing threshold**: `GATE_CROSSING_THRESHOLD = 10` pixels (increase if crossings aren't detected)
5. **Check track history**: `TRACK_HISTORY_LEN = 30` (increase if crossings are missed)

## Architecture Summary

```
ByteTrack → Track History (deque of (x,y) positions)
                ↓
    detect_gate_crossing(history, gate_x)
                ↓
    Returns: (crossed, 'left_to_right' | 'right_to_left')
                ↓
    Check if direction changed from last crossing
                ↓
         Interpret with gate_right:
    gate_right=True  → L→R=IN,  R→L=OUT
    gate_right=False → R→L=IN,  L→R=OUT
                ↓
         Handle accordingly:
         IN  → OCR → Log IN
         OUT → Log OUT
```

This is TRUE bidirectional detection that respects gate orientation! 🎉
