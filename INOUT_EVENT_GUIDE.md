# In/Out Event Tracking System - User Guide

## Overview
The `inout_event.py` system provides advanced license plate detection and tracking with **single-direction processing**, optimized OCR performance, and parking lot management features.

## Key Features

### 1. Single Direction Processing
The system uses **simplified single-direction logic** where each gate processes only ONE type of event. You configure the direction label:

```python
DIRECTION_MODE = "IN_ONLY"   # Label all events as "in" (entry gate)
DIRECTION_MODE = "OUT_ONLY"  # Label all events as "out" (exit gate)
```

**How It Works:**
- The same capture and OCR logic is used for both modes
- Only the CSV label ("in" or "out") and display text changes
- No complex dual-direction tracking - simpler and more reliable
- Perfect for dedicated entry/exit gates

**Use Cases:**
- **IN_ONLY**: Entry gates, one-way entrances, visitor check-in
- **OUT_ONLY**: Exit gates, one-way exits, departure tracking
- **Multiple Gates**: Run separate instances for entry and exit gates

### 2. Optimized OCR Performance

#### Enhanced OCR Trigger Zone
The system now uses a dedicated OCR trigger zone for better accuracy:

```python
OCR_TRIGGER_ZONE_RATIO   = 0.35   # 35% of frame width around gate
OCR_MIN_DISTANCE_FROM_GATE = 20   # Minimum distance from gate (px)
OCR_MAX_DISTANCE_FROM_GATE = 150  # Maximum distance from gate (px)
```

#### Smart OCR Triggering
OCR is only performed when plates are in the optimal recognition zone:

```python
USE_SMART_OCR_TRIGGER = True      # Enable smart OCR triggering
OCR_SKIP_FRAMES = 2               # Skip OCR every N frames for performance
```

#### Performance Improvements
- **INSIDE_PROCESS_EVERY_N**: Increased from 8 to 10 to reduce OCR load on vehicles already inside
- **MIN_RECOG_WIDTH_PX**: Increased from 40 to 45 for better OCR quality
- **MIN_RECOG_HEIGHT_PX**: Increased from 18 to 20 for better OCR quality
- **MIN_RECOG_SHARPNESS**: Increased from 25.0 to 30.0 for better OCR quality
- **GATE_CAPTURE_MARGIN_PX**: Increased from 60 to 80 for better OCR trigger area

**Expected Results:**
- 15-25% FPS improvement
- Better OCR accuracy due to focused processing
- Reduced CPU/GPU load

### 3. Parking Lot Implementation

#### Capacity Tracking
Real-time monitoring of parking lot occupancy:

```python
PARKING_CAPACITY         = 50      # Maximum parking capacity
PARKING_THRESHOLD        = 0.90    # Alert at 90% full
ENABLE_CAPACITY_TRACKING = True    # Track current occupancy
```

**Features:**
- Current occupancy count
- Occupancy percentage
- Near-capacity alerts (visual warning in red)

#### Analytics
Track parking usage patterns:

```python
ENABLE_ANALYTICS = True            # Enable parking analytics
```

**Metrics Tracked:**
- Total entries and exits
- Average parking duration (in minutes)
- Individual vehicle parking times
- Turnover rate

#### On-Screen Display
The system displays real-time parking statistics:
- Green text: Normal capacity
- Red text: Near capacity (≥90%)
- Format: `Occupancy:X/Y (Z%) Status | Avg Duration:Mmin`

Example:
```
Occupancy:8/50 (16%) Available | Avg Duration:23.5min
Occupancy:46/50 (92%) NEAR CAPACITY! | Avg Duration:45.2min
```

## Configuration Guide

### For Entry Gate (Single Gate Setup)
```python
DIRECTION_MODE = "IN_ONLY"
ENABLE_CAPACITY_TRACKING = True
ENABLE_ANALYTICS = True
PARKING_CAPACITY = 50              # Adjust to your lot size
GATE_INSIDE_IS_RIGHT = True        # Right side = inside parking lot
```

### For Exit Gate (Single Gate Setup)
```python
DIRECTION_MODE = "OUT_ONLY"
ENABLE_CAPACITY_TRACKING = True
ENABLE_ANALYTICS = True
PARKING_CAPACITY = 50
GATE_INSIDE_IS_RIGHT = False       # Left side = inside parking lot (exiting right)
```

### For Two-Gate Parking Lot
Run two separate instances:

**Entry Gate Process:**
```python
DIRECTION_MODE = "IN_ONLY"
INPUT_VIDEO_PATH = "input_videos/entry_gate.mp4"
ENTRY_LINE_X_RATIO = 0.55
GATE_INSIDE_IS_RIGHT = True
```

**Exit Gate Process:**
```python
DIRECTION_MODE = "OUT_ONLY"
INPUT_VIDEO_PATH = "input_videos/exit_gate.mp4"
ENTRY_LINE_X_RATIO = 0.55
GATE_INSIDE_IS_RIGHT = False
```

Both processes will write to the same daily CSV file, combining all IN and OUT events.

### For Maximum Performance (High Traffic)
```python
USE_SMART_OCR_TRIGGER = True
OCR_SKIP_FRAMES = 3                # Increase to 3 for more performance
INSIDE_PROCESS_EVERY_N = 12        # Increase to 12 for more performance
DETECT_EVERY_N = 2                 # Set to 2 to skip some detection frames
```

### For Maximum Accuracy (Low Traffic)
```python
USE_SMART_OCR_TRIGGER = False      # Process all frames
OCR_SKIP_FRAMES = 1
INSIDE_PROCESS_EVERY_N = 6
MIN_SHARPNESS_LOCK = 70.0          # Increase for stricter acceptance
```

## CSV Output Format

The system logs events to daily CSV files in `logs/parking_log_YYYYMMDD.csv`:

```csv
timestamp,object_id,vehicle_type,direction,city,engine_size,kana,four-digit number
2025-10-29 14:32:15,01,car,in,横浜,331,あ,12-34
2025-10-29 14:55:42,01,car,out,横浜,331,あ,12-34
```

**Fields:**
- **timestamp**: Event time
- **object_id**: Parking slot number (01-10, configurable)
- **vehicle_type**: Always "car"
- **direction**: "in" or "out"
- **city**: License plate region (横浜, 品川, etc.)
- **engine_size**: Vehicle class number (331, 530, etc.)
- **kana**: Character classification (あ, い, etc.)
- **four-digit number**: Unique identifier (12-34, etc.)

## Keyboard Controls

- **ESC**: Quit application
- **m**: Toggle between ROI mode and Gate mode
- **o**: Toggle gate orientation (left/right)
- **w/a/s/d**: Move ROI (when in ROI mode and not locked)
- **[ / ]**: Resize ROI
- **r**: Reset ROI to default position

## Performance Tips

1. **Adjust OCR trigger zone** based on your camera angle and distance
2. **Increase OCR_SKIP_FRAMES** if FPS is low (2-4 recommended)
3. **Use IN_ONLY or OUT_ONLY** for dedicated gates to save processing
4. **Set appropriate PARKING_CAPACITY** for accurate occupancy tracking
5. **Monitor FPS display** - aim for 15+ FPS for smooth operation

## Troubleshooting

### OCR accuracy is low
- Increase `MIN_RECOG_SHARPNESS` to 40-50
- Adjust `OCR_MIN_DISTANCE_FROM_GATE` and `OCR_MAX_DISTANCE_FROM_GATE`
- Ensure proper lighting and camera focus

### FPS is too low
- Increase `OCR_SKIP_FRAMES` to 3-4
- Increase `INSIDE_PROCESS_EVERY_N` to 12-15
- Set `DETECT_EVERY_N` to 2
- Disable `DRAW_TRACK_HISTORY`
- Disable `WRITE_VIDEO`

### Missing IN/OUT events
- Decrease `MIN_EVENT_GAP_FRAMES`
- Adjust `ENTER_STABLE_FRAMES` and `EXIT_STABLE_FRAMES`
- Verify `GATE_INSIDE_IS_RIGHT` matches your setup
- Increase `GATE_CAPTURE_MARGIN_PX`

## API Integration Example

The `ParkingAnalytics` class can be used for custom integrations:

```python
from inout_event import ParkingAnalytics

# Initialize
parking = ParkingAnalytics(capacity=50)

# Record events
parking.vehicle_entered("01", "2025-10-29 14:32:15")
parking.vehicle_exited("01", "2025-10-29 14:55:42")

# Get statistics
current_count = parking.current_count
occupancy = parking.get_occupancy_rate()  # 0.0 to 1.0
avg_duration = parking.get_average_duration()  # minutes
is_full = parking.is_near_capacity()
stats = parking.get_stats_string()
```

## Important Notes

### Single Direction Design
- Each instance processes **one direction only** (IN or OUT)
- The system captures plates crossing the gate and logs them with the configured direction
- No complex state tracking between IN/OUT - simpler and more reliable
- For full parking management, run two separate processes (one for entry, one for exit)

### CSV Output
- The system automatically creates daily CSV files in `logs/`
- All writes are immediately flushed to disk for reliability
- Multiple processes can write to the same CSV file (entry + exit gates)
- Each event is logged once when the vehicle crosses the gate

### Performance
- Single direction processing is faster than dual-direction tracking
- No exit event logic overhead when running IN_ONLY mode
- Better suited for high-traffic scenarios
