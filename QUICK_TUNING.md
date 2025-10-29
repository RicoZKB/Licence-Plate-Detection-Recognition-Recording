# Quick Tuning Guide for per8_optimized.py

## 🎯 Common Adjustments

### 🚀 Want Better FPS?
```python
# In per8_optimized.py, change these values:

DETECT_EVERY_N = 3              # Was: 2 (skip more frames)
DETECTION_BAND_WIDTH_RATIO = 0.4  # Was: 0.6 (narrower detection band)
TARGET_WIDTH = 480              # Was: 640 (lower resolution)
DRAW_TRACK_TRAILS = False       # Was: True (disable trails)
```
**Expected gain:** +5-10 FPS

---

### 🎯 Want Better Accuracy?
```python
DETECT_EVERY_N = 1              # Was: 2 (detect every frame)
DETECTION_BAND_WIDTH_RATIO = 0.8  # Was: 0.6 (wider detection band)
BYTETRACK_THRESHOLD = 0.2       # Was: 0.3 (more sensitive)
RUN_OCR_ON_CROSSING_ONLY = False  # Was: True (always run OCR)
```
**Expected cost:** -5-8 FPS

---

### 📏 Move the Detection Line
```python
# Line at 50% of width (left):
LINE_START_RATIO = (0.50, 0.0)
LINE_END_RATIO = (0.50, 1.0)

# Line at 60% of width (right):
LINE_START_RATIO = (0.60, 0.0)
LINE_END_RATIO = (0.60, 1.0)

# Horizontal line (not recommended):
LINE_START_RATIO = (0.0, 0.5)
LINE_END_RATIO = (1.0, 0.5)
```

---

### 🎨 Customize Annotations
```python
# Bigger/smaller boxes:
BOX_THICKNESS = 3  # Was: 2

# Bigger/smaller labels:
LABEL_TEXT_SCALE = 0.8  # Was: 0.6
LABEL_TEXT_THICKNESS = 3  # Was: 2

# Longer/shorter track trails:
TRACK_HISTORY_LENGTH = 50  # Was: 30
```

---

### 🔧 ByteTrack Tuning
```python
# More aggressive tracking (keep IDs longer):
BYTETRACK_LOST_BUFFER = 30  # Was: 20

# Stricter matching (less ID switches):
BYTETRACK_MATCH_THRESH = 0.7  # Was: 0.6

# Lower detection threshold (detect more objects):
BYTETRACK_THRESHOLD = 0.2  # Was: 0.3
```

---

## ⚡ Performance Presets

### Preset 1: BALANCED (Recommended)
```python
DETECT_EVERY_N = 2
DETECTION_BAND_WIDTH_RATIO = 0.6
TARGET_WIDTH = 640
DRAW_TRACK_TRAILS = True
RUN_OCR_ON_CROSSING_ONLY = True
```
**Expected: 15-20 FPS, Good accuracy**

---

### Preset 2: MAX FPS
```python
DETECT_EVERY_N = 4
DETECTION_BAND_WIDTH_RATIO = 0.4
TARGET_WIDTH = 480
DRAW_TRACK_TRAILS = False
RUN_OCR_ON_CROSSING_ONLY = True
```
**Expected: 25-30 FPS, Acceptable accuracy**

---

### Preset 3: MAX ACCURACY
```python
DETECT_EVERY_N = 1
DETECTION_BAND_WIDTH_RATIO = 0.8
TARGET_WIDTH = 640
DRAW_TRACK_TRAILS = True
RUN_OCR_ON_CROSSING_ONLY = False
```
**Expected: 8-10 FPS, Best accuracy**

---

## 🎥 Video Source Setup

### Use Camera
```python
USE_CAMERA = True
USE_FILE = False
USE_YOUTUBE = False
```

### Use Video File
```python
USE_CAMERA = False
USE_FILE = True
USE_YOUTUBE = False
INPUT_VIDEO_PATH = "input_videos/your_video.mp4"
```

---

## 📊 Save Output Video
```python
WRITE_VIDEO = True  # Was: False

# Output will be saved to:
# output_videos/output_optimized.avi
```

---

## 🐛 Debug Mode
```python
# Show more info in terminal:
PRINT_EVENTS_TO_TERMINAL = True

# Show FPS on screen:
SHOW_FPS = True

# See what's happening:
# Watch the terminal for IN/OUT events with plate info
```

---

## 💡 Tips

1. **Start with BALANCED preset** and adjust from there
2. **If FPS drops below target:** Increase `DETECT_EVERY_N`
3. **If missing vehicles:** Decrease `DETECT_EVERY_N` or widen detection band
4. **If too many false detections:** Increase `BYTETRACK_THRESHOLD`
5. **If IDs switch too often:** Increase `BYTETRACK_MATCH_THRESH`

---

## 🔄 Quick Test Workflow

1. **Run the script:**
   ```bash
   source cenv/bin/activate
   python per8_optimized.py
   ```

2. **Watch FPS overlay** on screen

3. **Check terminal** for IN/OUT events

4. **Adjust parameters** in the file

5. **Press ESC** to quit

6. **Repeat** from step 1

---

## 📁 Check Logs
```bash
# View today's parking log:
cat logs/parking_log_$(date +%Y%m%d).csv

# Or just:
ls -lt logs/
cat logs/parking_log_*.csv
```

---

**Happy tuning! 🎛️**
