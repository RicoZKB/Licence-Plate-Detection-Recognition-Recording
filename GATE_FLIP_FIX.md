# Gate Orientation Flip Fix

## Problem Description

When pressing 'o' to flip the gate orientation (changing which side is "inside"), vehicles that were already being tracked would never generate exit events. This happened because:

1. **State Confusion**: When `gate_right` flips, the definition of "inside" changes, but tracked vehicles still have their old `prev_inside` state
2. **Direction History Conflicts**: The `last_crossing_direction` doesn't get reset, preventing new crossings from being detected
3. **Locked Capture State**: Vehicles with `capture_open[oid] = -1` (locked) never re-evaluate their position

## Solutions Implemented

### Solution 1: Complete State Reset ([per7_format.py](per7_format.py))

**Approach**: Clear all tracking state and force-exit all vehicles when gate flips.

**Pros**:
- Simple and reliable
- Guarantees no state confusion
- Easy to understand and maintain

**Cons**:
- Loses tracking continuity
- All vehicles must exit and re-enter
- Not ideal if you want to preserve tracking

**When to use**:
- When you need guaranteed clean state
- For simpler deployments
- When tracking continuity is not critical

### Solution 2: Intelligent State Recalculation ([per8_optimized.py](per8_optimized.py))

**Approach**: Recalculate which side each vehicle is on based on current position.

**How it works**:
1. Get the last known position of each tracked vehicle from `track_history`
2. Check if vehicle is on the "inside" side with the NEW gate orientation
3. If vehicle is now on "outside", auto-exit it and log the event
4. If vehicle is still on "inside", keep tracking but recalculate its state
5. Clear crossing direction history to force re-detection

**Pros**:
- Preserves tracking for vehicles truly still inside
- More intelligent and user-friendly
- Only exits vehicles that actually should exit
- Better for live/production systems

**Cons**:
- More complex logic
- Depends on track history being accurate
- Edge cases if vehicle is exactly on gate line

**When to use**:
- Production systems where tracking continuity matters
- When you want minimal disruption from orientation changes
- When vehicles may legitimately be on either side

## Testing

Test both versions with this scenario:

1. Start the system (either per7_format.py or per8_optimized.py)
2. Let a vehicle enter from one side (should log IN event)
3. Press 'o' to flip gate orientation
4. Watch vehicle cross the gate in the opposite direction
5. Verify OUT event is logged

### Expected Results

**per7_format.py**:
- Immediate OUT event when 'o' is pressed (force-exit)
- Vehicle must re-enter to generate new IN event

**per8_optimized.py**:
- If vehicle is on "outside" after flip: immediate OUT event
- If vehicle is on "inside" after flip: remains tracked, OUT when it crosses
- More natural behavior based on actual position

## Usage

```bash
# Solution 1 (simple, complete reset)
python per7_format.py

# Solution 2 (intelligent recalculation)
python per8_optimized.py
```

## Keyboard Controls

- **'o'**: Flip gate orientation (inside/outside)
- **'m'**: Toggle between ROI mode and Gate mode
- **ESC**: Quit

## Terminal Output on Gate Flip

**per7_format.py**:
```
[MODE CHANGE] Gate orientation flipped. Inside is now: LEFT
[MODE CHANGE] Clearing all tracking state to prevent conflicts...
  → Auto-exit: slot 01, suffix=12-34
[MODE CHANGE] State cleared. Ready to track with new orientation.
```

**per8_optimized.py**:
```
[MODE CHANGE] Gate orientation flipped. Inside is now: LEFT
[MODE CHANGE] Recalculating states for tracked vehicles...
  → Vehicle 5 now on outside (cx=450, gate=352), auto-exit
    ✓ Logged exit: slot 01, suffix=12-34
  → Vehicle 7 still inside (cx=500, gate=352), keeping
[MODE CHANGE] State recalculated. 1 exited, 1 kept.
```

## Additional Improvements in Both Versions

Both versions now provide:
- Clear console feedback when gate flips
- Proper CSV logging of auto-exit events
- Slot pool cleanup (released slots available for reuse)
- Dictionary cleanup to prevent memory leaks

## Recommendations

- **For testing/development**: Use [per7_format.py](per7_format.py) (simpler)
- **For production**: Use [per8_optimized.py](per8_optimized.py) (smarter)
- **For bidirectional parking**: Strongly recommend [per8_optimized.py](per8_optimized.py)

## Future Enhancements

Possible improvements:
1. Add hysteresis zone around gate line (don't flip if vehicles too close)
2. Warn user if vehicles are near gate when trying to flip
3. Add confirmation prompt before flipping
4. Visual indicator showing which vehicles will exit on flip
5. Undo capability (flip back within N seconds)
