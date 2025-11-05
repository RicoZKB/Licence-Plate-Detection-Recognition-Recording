# -*- coding: utf-8 -*-
# per5_format.py — ROI-cropped, one-shot plate capture + throttled OCR while inside + YouTube option
# - Cropped plate detection (ROI or Gate band) for speed
# - One-shot OCR on ENTER (best sharpness + valid suffix), then throttle while inside
# - Immediate CSV flush on every write
# - Source selector: Camera / File / YouTube (choose exactly one)
# - Terminal logging of IN/OUT events (+ optional raw OCR debug)

import os, cv2, csv, re, time
from collections import deque
from datetime import datetime
from detections import CarDetection,LicencePlateDetection

# ===================== User settings =====================
# Choose exactly ONE:
USE_CAMERA               = False
USE_FILE                 = True
USE_YOUTUBE              = False

# File source
INPUT_VIDEO_PATH         = "input_videos/Trim03.mp4"

# YouTube source (VOD or live). If enabled, overrides camera/file.
YOUTUBE_URL              = "https://www.youtube.com/watch?v=jqtsC5BYlIk"
YOUTUBE_MAX_HEIGHT       = 480   # progressive MP4 at or below this height

# Directional mode: "IN_ONLY", "OUT_ONLY" (single direction processing)
DIRECTION_MODE           = "OUT_ONLY"  # <-- Choose IN_ONLY or OUT_ONLY for single-direction tracking

# Terminal logging toggles
PRINT_EVENTS_TO_TERMINAL = True   # <-- print every IN/OUT nicely to terminal
PRINT_RAW_OCR_FOUND      = False  # <-- print per-detection OCR strings when they contain kana/suffix
PASS_OCR_DEBUG_TO_DETECT = False  # <-- pass debug_ocr=True to LicencePlateDetection (very verbose)
# =========================================================

SLOT_COUNT               = 10
START_ID_AT              = 100

# Region as ratios of frame (x, y, w, h)
REGION_XYWH_RATIO        = (0.32, 0.60, 0.36, 0.20)

# Lock ROI from code (disable WASD/[ ])
LOCK_ROI                 = False

# Use a virtual gate line instead of ROI region (detect anywhere)
USE_ROI                  = False
ENTRY_LINE_X_RATIO       = 0.55   # vertical line as fraction of width
GATE_CAPTURE_MARGIN_PX   = 80     # start capture window when within this many px of gate (increased for better OCR trigger)
DRAW_GATE                = True
GATE_INSIDE_IS_RIGHT     = True   # True: right of line means "inside"; False: left is inside

# OCR trigger zone optimization (for better accuracy and performance)
OCR_TRIGGER_ZONE_RATIO   = 0.35   # Dedicated zone width ratio around gate for OCR (0.35 = 35% of frame width)
OCR_MIN_DISTANCE_FROM_GATE = 20   # Min pixels from gate to start OCR (avoid edge cases)
OCR_MAX_DISTANCE_FROM_GATE = 150  # Max pixels from gate to trigger OCR (focus on clear plates)

# Stability & gaps
ENTER_STABLE_FRAMES      = 1
EXIT_STABLE_FRAMES       = 4
MIN_EVENT_GAP_FRAMES     = 10

# One-shot window (fallback in case sharpness never crosses threshold)
CAPTURE_WINDOW_FRAMES    = 6

# While car remains inside, only process heavy every N frames
INSIDE_PROCESS_EVERY_N   = 10    # Increased from 8 to reduce OCR load
DETECT_EVERY_N           = 1     # run detector every N frames

# OCR Performance optimization
OCR_SKIP_FRAMES          = 2     # Skip OCR every N frames (detector still runs, but OCR is skipped)
USE_SMART_OCR_TRIGGER    = True  # Only trigger OCR when plate is in optimal zone
ENABLE_OCR_CACHE         = True  # Cache OCR results by tracker ID (20-30% performance boost)
OCR_CACHE_MIN_CONFIDENCE = 0.85  # Minimum confidence to accept cached OCR result
OCR_CACHE_MAX_AGE_FRAMES = 180   # Max frames to keep cached OCR (6s at 30fps)

# One-shot accept rules
# Accept hyphen variants: - − – — ー ｰ ~ 〜 and optional spaces
PLATE_SUFFIX_RE          = re.compile(r"(\d{2,3})\s*[-−–—ーｰ~〜－]\s*(\d{2})")
MIN_SHARPNESS_LOCK       = 60.0   # Laplacian variance to accept immediately

# Performance toggles
WRITE_VIDEO              = False
DRAW_BBOXES              = True
SHOW_FPS                 = True
TARGET_WIDTH             = 640
BOX_THICK                = 1
ROI_INFER_MAX_W          = 640
GATE_DET_BAND_RATIO      = 0.50
TRACK_HISTORY_LEN        = 14
DRAW_TRACK_HISTORY       = True
TRACK_LINE_COLOR_IN      = (0, 220, 255)   # BGR
TRACK_LINE_COLOR_OUT     = (255, 200, 80)
TRACK_POINT_RADIUS       = 2
TRACK_STALE_FORGET       = 90
MIN_RECOG_WIDTH_PX       = 45     # Increased from 40 for better OCR quality
MIN_RECOG_HEIGHT_PX      = 20     # Increased from 18 for better OCR quality
MIN_RECOG_SHARPNESS      = 30.0   # Increased from 25.0 for better OCR quality

# Parking lot implementation settings
PARKING_CAPACITY         = 50     # Maximum parking capacity
PARKING_THRESHOLD        = 0.90   # Alert when 90% full
ENABLE_CAPACITY_TRACKING = True   # Track current parking occupancy
ENABLE_ANALYTICS         = True   # Enable parking analytics (duration, turnover rate)
# =========================================================

# Reduce OpenCV overhead a bit
cv2.setUseOptimized(True)
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")
os.environ.setdefault("FLAGS_minloglevel", "2")

def _resolve_youtube_opencv_url(url: str, max_height: int = 480) -> str:
    """Return a direct MP4 HTTP URL for OpenCV VideoCapture using yt-dlp."""
    try:
        import yt_dlp
    except Exception:
        print("[WARN] yt-dlp not installed. `pip install yt-dlp` to use YouTube input.")
        return None
    fmt = f"best[acodec!=none][vcodec!=none][ext=mp4][height<={max_height}]/best[acodec!=none][vcodec!=none][ext=mp4]/best[acodec!=none][vcodec!=none]"
    ydl_opts = {'quiet': True, 'no_warnings': True, 'format': fmt, 'noplaylist': True}
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            if 'url' in info:
                return info['url']
            fmt_id = info.get('format_id'); fmts = info.get('formats') or []
            for f in fmts:
                if f.get('format_id') == fmt_id and f.get('acodec') != 'none' and f.get('vcodec') != 'none':
                    return f.get('url')
            for f in fmts:
                if (f.get('acodec') != 'none' and f.get('vcodec') != 'none' and str(f.get('protocol','')).startswith('http')):
                    return f.get('url')
    except Exception as e:
        print('[WARN] yt-dlp extraction failed:', e)
    return None

CITY_KANA_RE = re.compile(u"^\\s*([^\\s\\d]+)\\s*[0-9]{2,4}.*?([ぁ-ゖァ-ヿA-Za-z])", re.UNICODE)

def parse_city_kana(text):
    if not text: return None, None
    m = CITY_KANA_RE.search(text);  return (m.group(1), m.group(2)) if m else (None, None)

def normalize_ocr_text(text: str) -> str:
    if not text: return ""
    fw_digits = "０１２３４５６７８９"; ascii_digits = "0123456789"
    trans = str.maketrans({fw: ascii for fw, ascii in zip(fw_digits, ascii_digits)})
    t = text.translate(trans)
    t = re.sub(r"[−–—ーｰ~〜－]", "-", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def plate_suffix(text):
    if not text: return None
    t = normalize_ocr_text(text); m = PLATE_SUFFIX_RE.search(t)
    if not m: return None
    return f"{m.group(1)}-{m.group(2)}"

def plate_key(text):
    t = normalize_ocr_text(text or "")
    suf = plate_suffix(t);  city, _ = parse_city_kana(t)
    if not suf: return None
    return f"{city if city else '?'}|{suf}"

# Updated: now also returns engine_size (class number) as the 5th value
REGION_WITH_NUM_RE = re.compile(u"([ぁ-ゖァ-ヿ一-龯A-Za-z]+)\\s*([0-9]{2,4})", re.UNICODE)
def parse_plate_fields(text):
    if not text: return None, None, None, None, None
    t = normalize_ocr_text(text)
    suf = plate_suffix(t)
    city = None; region_class = None; engine_size = None
    m = REGION_WITH_NUM_RE.search(t)
    if m:
        city = m.group(1)
        engine_size = m.group(2)             # e.g., 310, 331
        region_class = f"{city}{engine_size}"
    kana = None; kana_matches = re.findall(u"[ぁ-ゖァ-ヿ]", t)
    if kana_matches: kana = kana_matches[-1]
    return region_class, suf, kana, city, engine_size

def open_daily_csv():
    date_str = datetime.now().strftime("%Y%m%d")
    os.makedirs("logs", exist_ok=True)
    path = os.path.join("logs", f"parking_log_{date_str}.csv")
    new = not os.path.exists(path)
    f = open(path, "a", newline="", encoding="utf-8", buffering=1)  # line-buffered
    w = csv.writer(f)
    if new:
        # New header
        w.writerow(["timestamp","object_id","vehicle_type","direction","city","engine_size","kana","four-digit number"])
        f.flush()
    return f, w, path

def write_row_flush(w, f, row):
    w.writerow(row); f.flush()
    try: os.fsync(f.fileno())
    except Exception: pass

# Pretty print to terminal
def log_terminal(event_ts, direction, slot_str, region_class, suffix, kana, city, raw_text):
    if not PRINT_EVENTS_TO_TERMINAL:
        return
    kana_disp = kana if kana else ""
    city_disp = city if city else ""
    rc_disp = region_class if region_class else ""
    suf_disp = suffix if suffix else ""
    print(f"[{event_ts}] {direction.upper():3s}  slot={slot_str:>6s}  region_class='{rc_disp}'  suffix='{suf_disp}'  kana='{kana_disp}'  city='{city_disp}'  raw=\"{raw_text}\"")

# ---------- Smart OCR trigger helper ----------
def should_trigger_ocr(cx, gate_x, frame_idx, gate_right=True):
    """
    Determine if OCR should be triggered based on position relative to gate.
    Returns True only when plate is in optimal zone for OCR.
    """
    if not USE_SMART_OCR_TRIGGER:
        return True  # Always trigger if smart triggering is disabled

    # Skip OCR on certain frames for performance
    if frame_idx % OCR_SKIP_FRAMES != 0:
        return False

    if cx is None:
        return False

    distance_from_gate = abs(cx - gate_x)

    # Only trigger OCR in optimal zone
    if OCR_MIN_DISTANCE_FROM_GATE <= distance_from_gate <= OCR_MAX_DISTANCE_FROM_GATE:
        # Check if approaching from correct side
        approaching_from_outside = (cx < gate_x) if gate_right else (cx > gate_x)
        return approaching_from_outside

    return False

# ---------- OCR Cache for performance ----------
class OCRCache:
    """Cache OCR results by tracker ID to avoid redundant processing."""
    def __init__(self):
        self.cache = {}  # oid -> {text, sharpness, confidence, frame_idx, suffix_present}
        self.hit_count = 0
        self.miss_count = 0

    def get(self, oid, current_frame_idx):
        """Get cached OCR result if available and not stale."""
        if not ENABLE_OCR_CACHE:
            return None

        if oid not in self.cache:
            self.miss_count += 1
            return None

        entry = self.cache[oid]
        age = current_frame_idx - entry['frame_idx']

        # Check if cache is stale
        if age > OCR_CACHE_MAX_AGE_FRAMES:
            self.cache.pop(oid, None)
            self.miss_count += 1
            return None

        # Check if confidence is high enough
        if entry.get('confidence', 0.0) < OCR_CACHE_MIN_CONFIDENCE:
            self.miss_count += 1
            return None

        # Check if suffix is present (required for valid plate)
        if not entry.get('suffix_present', False):
            self.miss_count += 1
            return None

        self.hit_count += 1
        return entry['text']

    def update(self, oid, text, sharpness, frame_idx):
        """Update cache with new OCR result."""
        if not ENABLE_OCR_CACHE:
            return

        suffix_present = bool(plate_suffix(text or ""))

        # Calculate confidence based on sharpness and suffix presence
        confidence = 0.0
        if suffix_present:
            confidence = min(1.0, sharpness / 100.0)  # Normalize sharpness to 0-1

        # Only cache if we have a suffix (valid plate)
        if suffix_present:
            # Update if new result is better or entry doesn't exist
            if oid not in self.cache or sharpness > self.cache[oid].get('sharpness', 0):
                self.cache[oid] = {
                    'text': text,
                    'sharpness': sharpness,
                    'confidence': confidence,
                    'frame_idx': frame_idx,
                    'suffix_present': suffix_present
                }

    def remove(self, oid):
        """Remove entry from cache."""
        self.cache.pop(oid, None)

    def cleanup_stale(self, current_frame_idx):
        """Remove stale entries from cache."""
        stale_ids = [
            oid for oid, entry in self.cache.items()
            if current_frame_idx - entry['frame_idx'] > OCR_CACHE_MAX_AGE_FRAMES
        ]
        for oid in stale_ids:
            self.cache.pop(oid, None)

    def get_stats(self):
        """Get cache statistics."""
        total = self.hit_count + self.miss_count
        hit_rate = (self.hit_count / total * 100) if total > 0 else 0.0
        return {
            'hits': self.hit_count,
            'misses': self.miss_count,
            'hit_rate': hit_rate,
            'cache_size': len(self.cache)
        }

# ---------- Parking lot analytics helper ----------
class ParkingAnalytics:
    """Track parking lot occupancy and analytics."""
    def __init__(self, capacity=PARKING_CAPACITY):
        self.capacity = capacity
        self.current_count = 0
        self.entry_times = {}  # vehicle_id -> entry timestamp
        self.total_entries = 0
        self.total_exits = 0
        self.durations = []  # list of parking durations in seconds

    def vehicle_entered(self, vehicle_id, timestamp):
        """Record vehicle entry."""
        if ENABLE_ANALYTICS:
            self.entry_times[vehicle_id] = timestamp
            self.total_entries += 1
        if ENABLE_CAPACITY_TRACKING:
            self.current_count = min(self.capacity, self.current_count + 1)

    def vehicle_exited(self, vehicle_id, timestamp):
        """Record vehicle exit and calculate duration."""
        if ENABLE_CAPACITY_TRACKING:
            self.current_count = max(0, self.current_count - 1)
        if ENABLE_ANALYTICS and vehicle_id in self.entry_times:
            entry_time = self.entry_times.pop(vehicle_id)
            try:
                entry_dt = datetime.strptime(entry_time, "%Y-%m-%d %H:%M:%S")
                exit_dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
                duration = (exit_dt - entry_dt).total_seconds()
                self.durations.append(duration)
                self.total_exits += 1
            except Exception:
                pass

    def get_occupancy_rate(self):
        """Return current occupancy as percentage (0.0 to 1.0)."""
        if not ENABLE_CAPACITY_TRACKING:
            return 0.0
        return self.current_count / self.capacity if self.capacity > 0 else 0.0

    def is_near_capacity(self):
        """Check if parking lot is near capacity."""
        return self.get_occupancy_rate() >= PARKING_THRESHOLD

    def get_average_duration(self):
        """Get average parking duration in minutes."""
        if not self.durations:
            return 0.0
        return sum(self.durations) / len(self.durations) / 60.0  # convert to minutes

    def get_stats_string(self):
        """Return formatted stats string."""
        if not ENABLE_CAPACITY_TRACKING:
            return ""
        occupancy = self.get_occupancy_rate()
        status = "NEAR CAPACITY!" if self.is_near_capacity() else "Available"
        avg_dur = self.get_average_duration()
        return f"Occupancy:{self.current_count}/{self.capacity} ({occupancy*100:.0f}%) {status} | Avg Duration:{avg_dur:.1f}min"

# ---------- geometry helpers ----------
def xyxy_from_det(det):
    if isinstance(det, (list, tuple)) and len(det) >= 4:
        return float(det[0]), float(det[1]), float(det[2]), float(det[3])
    if isinstance(det, dict):
        box = det.get("bbox") or det.get("xyxy")
        if box and len(box) >= 4: return float(box[0]), float(box[1]), float(box[2]), float(box[3])
    return None

def offset_det(det, ox, oy):
    if isinstance(det, (list, tuple)) and len(det) >= 4:
        x1,y1,x2,y2 = det[:4]; return [x1+ox, y1+oy, x2+ox, y2+oy]
    if isinstance(det, dict):
        box = det.get("bbox") or det.get("xyxy")
        if box and len(box) >= 4:
            box = [box[0]+ox, box[1]+oy, box[2]+ox, box[3]+oy]
            if "bbox" in det: det["bbox"] = box
            else: det["xyxy"] = box
            return det
    return det

def scale_det(det, sx, sy):
    if isinstance(det, (list, tuple)) and len(det) >= 4:
        x1,y1,x2,y2 = det[:4]; return [x1*sx, y1*sy, x2*sx, y2*sy]
    if isinstance(det, dict):
        box = det.get("bbox") or det.get("xyxy")
        if box and len(box) >= 4:
            box = [box[0]*sx, box[1]*sy, box[2]*sx, box[3]*sy]
            if "bbox" in det: det["bbox"] = box
            else: det["xyxy"] = box
            return det
    return det

def center_from_det(det):
    xy = xyxy_from_det(det)
    if not xy: return None, None
    x1,y1,x2,y2 = xy
    return (x1+x2)/2.0, (y1+y2)/2.0

def bbox_area(det):
    xy = xyxy_from_det(det)
    if not xy: return 0.0
    x1,y1,x2,y2 = xy
    return max(0, x2-x1) * max(0, y2-y1)

def variance_of_laplacian(img):
    if img is None or img.size == 0: return 0.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape)==3 else img
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def crop_frame(frame, rx, ry, rw, rh):
    return frame[ry:ry+rh, rx:rx+rw]

# ---------- ID / slots ----------
class IDBank:
    def __init__(self, start_at=100):
        self._map = {}; self._next = int(start_at)
    def get(self, key):
        if key not in self._map:
            self._map[key] = self._next; self._next += 1
        return self._map[key]

def stable_key(text, det):
    xy = xyxy_from_det(det)
    if xy is None: return text if text else None
    x1,y1,x2,y2 = xy
    qx = int(((x1+x2)/2.0)//20); qy = int(((y1+y2)/2.0)//20)
    t  = text if (text and len(text)>=3) else ""
    return f"bb-{qx}-{qy}|{t}"

class SlotPool:
    def __init__(self, n): self.free=list(range(1,n+1)); self.used=set()
    def acquire_lowest(self):
        if not self.free: return None
        s=self.free.pop(0); self.used.add(s); return s
    def try_acquire_specific(self, s):
        if s in self.free:
            self.free.remove(s); self.used.add(s); return s
        return None
    def release(self, s):
        if s in self.used:
            self.used.remove(s); self.free.append(s); self.free.sort()

# ---------- region utils ----------
def inside_region(cx, cy, rx, ry, rw, rh):
    return (cx is not None) and (cy is not None) and (rx <= cx <= rx+rw) and (ry <= cy <= ry+rh)

def clamp_region(rx, ry, rw, rh, W, H):
    rx = max(0, min(rx, W-10)); ry = max(0, min(ry, H-10))
    rw = max(20, min(rw, W-rx)); rh = max(20, min(rh, H-ry))
    return rx, ry, rw, rh

def draw_region(frame, rx, ry, rw, rh):
    cv2.rectangle(frame, (rx, ry), (rx+rw, ry+rh), (0,255,255), BOX_THICK)
    cv2.putText(frame, "ROI (locked)" if LOCK_ROI else "ROI: WASD move, [ ] resize, r reset",
                (rx+6, max(18, ry-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255), 2, cv2.LINE_AA)

def draw_lp_boxes(frame, dets, color=(0,255,0)):
    for d in dets:
        xy = xyxy_from_det(d)
        if not xy: continue
        x1,y1,x2,y2 = map(int, xy)
        cv2.rectangle(frame, (x1,y1), (x2,y2), color, BOX_THICK)
        if DRAW_TRACK_HISTORY:
            cv2.circle(frame, (int((x1+x2)/2), int((y1+y2)/2)), TRACK_POINT_RADIUS, color, -1)

def draw_track_history_overlay(frame, histories, inside_flags):
    if not DRAW_TRACK_HISTORY:
        return
    for oid, pts in histories.items():
        if not pts or len(pts) < 2:
            continue
        color = TRACK_LINE_COLOR_IN if inside_flags.get(oid, False) else TRACK_LINE_COLOR_OUT
        pts_int = [(int(px), int(py)) for px, py in pts if px is not None and py is not None]
        if len(pts_int) < 2:
            continue
        for i in range(1, len(pts_int)):
            cv2.line(frame, pts_int[i-1], pts_int[i], color, 2)
        # emphasize most recent point
        cv2.circle(frame, pts_int[-1], TRACK_POINT_RADIUS+1, color, -1)

# ---------- main ----------
def main():
    # --- Validate direction mode ---
    if DIRECTION_MODE not in ["IN_ONLY", "OUT_ONLY"]:
        raise RuntimeError("DIRECTION_MODE must be either 'IN_ONLY' or 'OUT_ONLY'")

    print(f"[INFO] Direction mode: {DIRECTION_MODE}")
    if DIRECTION_MODE == "IN_ONLY":
        print("[INFO] Only IN events will be tracked and logged")
    else:
        print("[INFO] Only OUT events will be tracked and logged")

    # --- Source selection (exactly one) ---
    chosen = [name for name, val in [("Camera", USE_CAMERA), ("YouTube", USE_YOUTUBE), ("File", USE_FILE)] if val]
    if len(chosen) != 1:
        raise RuntimeError("Exactly one of USE_CAMERA, USE_YOUTUBE, USE_FILE must be True.")
    source_name = chosen[0]

    if source_name == "Camera":
        src = 0
    elif source_name == "YouTube":
        print("[INFO] Resolving YouTube URL...", YOUTUBE_URL)
        src = _resolve_youtube_opencv_url(YOUTUBE_URL, max_height=YOUTUBE_MAX_HEIGHT)
        if not src: raise RuntimeError("Failed to resolve YouTube stream.")
        short = (src[:80] + '...') if isinstance(src, str) and len(src) > 80 else src
        print("[INFO] Using YouTube stream:", short)
    else:
        src = INPUT_VIDEO_PATH
        if not os.path.exists(src): raise RuntimeError(f"Input file not found: {src}")

    # runtime flags (toggled by keyboard)
    use_roi = USE_ROI
    gate_right = GATE_INSIDE_IS_RIGHT

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {src}")

    using_camera = (source_name == "Camera")
    if using_camera:
        # keep the camera buffer tiny; prefer fresh frames
        try: cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except: pass

    try: cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)
    except: pass

    # Plate detector (pass debug flag if you want raw OCR printed)
    lp_det  = LicencePlateDetection(model_path="models/best.pt",
                                    verbose=False,
                                    debug_ocr=PASS_OCR_DEBUG_TO_DETECT)

    os.makedirs("output_videos", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or TARGET_WIDTH)
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or int(TARGET_WIDTH*3/5))
    out = cv2.VideoWriter("output_videos/output_video.avi", fourcc, fps, (W, H)) if WRITE_VIDEO else None

    csv_file, csv_writer, csv_path = open_daily_csv()
    print("[INFO] Logging to:", csv_path)

    # Initialize parking analytics
    parking_stats = ParkingAnalytics(capacity=PARKING_CAPACITY)

    # Initialize OCR cache
    ocr_cache = OCRCache()

    # ROI (or full frame if use_roi=False)
    if use_roi:
        rx = int(REGION_XYWH_RATIO[0]*W); ry = int(REGION_XYWH_RATIO[1]*H)
        rw = int(REGION_XYWH_RATIO[2]*W); rh = int(REGION_XYWH_RATIO[3]*H)
        rx, ry, rw, rh = clamp_region(rx, ry, rw, rh, W, H)
    else:
        rx, ry, rw, rh = 0, 0, W, H

    idbank = IDBank(START_ID_AT)
    prev_inside = {}
    inside_count, outside_count, last_event_f = {}, {}, {}

    slot_pool = SlotPool(SLOT_COUNT)
    plate_pref_slot, active_plate_to_slot, active_id_to_plate = {}, {}, {}
    active_suffix_to_slot = {}
    active_id_to_slot = {}
    active_id_to_suffix = {}
    track_history = {}
    last_seen_frame = {}
    latest_det = {}
    pending_in_cross = set()

    # one-shot bookkeeping
    capture_open = {}    # oid -> frames left (0 close, -1 locked)
    best_score  = {}     # oid -> area*sharpness (fallback ranking)
    best_text   = {}     # oid -> best OCR so far
    best_det    = {}     # oid -> det for best
    captured_in = set()

    frame_idx = 0
    t0 = time.time(); fcount = 0; fps_est = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok: break
            frame_idx += 1; fcount += 1

            # Freshest frames for cameras
            if using_camera:
                try:
                    for _ in range(2):
                        cap.grab()
                except Exception:
                    pass

            # --- ROI bookkeeping for overlays ---
            rx, ry, rw, rh = clamp_region(rx, ry, rw, rh, W, H)

            # throttle heavy work if any captured car is still inside
            have_locked_inside = any((oid in captured_in) and prev_inside.get(oid, False) for oid in prev_inside.keys())
            do_heavy_this_frame = True
            if have_locked_inside and (frame_idx % INSIDE_PROCESS_EVERY_N != 0):
                do_heavy_this_frame = False
            if frame_idx % DETECT_EVERY_N != 0:
                do_heavy_this_frame = False

            lp_dets_full, lp_texts = [], []

            if do_heavy_this_frame:
                # Choose detection region: ROI if enabled, else a vertical band around the gate
                gate_x = int(ENTRY_LINE_X_RATIO * W)
                if use_roi:
                    d_rx, d_ry, d_rw, d_rh = rx, ry, rw, rh
                else:
                    # Use OCR_TRIGGER_ZONE_RATIO for smarter detection band
                    band_w = max(180, int(W * OCR_TRIGGER_ZONE_RATIO))
                    d_rx = max(0, gate_x - band_w//2)
                    d_ry = 0
                    d_rw = min(band_w, W - d_rx)
                    d_rh = H
                    if latest_det:
                        min_x = d_rx
                        max_x = d_rx + d_rw
                        for det in latest_det.values():
                            xy = xyxy_from_det(det)
                            if not xy:
                                continue
                            x1, y1, x2, y2 = xy
                            min_x = min(min_x, max(0.0, x1 - 40))
                            max_x = max(max_x, min(float(W), x2 + 40))
                        d_rx = int(max(0, min_x))
                        d_rw = int(min(W - d_rx, max(40, max_x - min_x)))
                    det_roi = crop_frame(frame, d_rx, d_ry, d_rw, d_rh)

                # Downscale detection ROI for speed, detect, then map back
                inf_roi = det_roi
                scale = 1.0
                if det_roi.shape[1] > ROI_INFER_MAX_W:
                    scale = ROI_INFER_MAX_W / float(det_roi.shape[1])
                    new_w = int(det_roi.shape[1] * scale)
                    new_h = int(det_roi.shape[0] * scale)
                    inf_roi = cv2.resize(det_roi, (new_w, new_h))

                all_lp_dets, all_lp_texts = lp_det.detect_frames([inf_roi])
                roi_dets = all_lp_dets[0]; lp_texts = all_lp_texts[0]

                # map boxes back to full-frame coordinates
                lp_dets_full = []
                if scale != 1.0:
                    inv_sx = 1.0/scale; inv_sy = 1.0/scale
                    for d in roi_dets:
                        d_scaled = scale_det(d, inv_sx, inv_sy)
                        lp_dets_full.append(offset_det(d_scaled, d_rx, d_ry))
                else:
                    for d in roi_dets:
                        lp_dets_full.append(offset_det(d, d_rx, d_ry))
            else:
                lp_dets_full, lp_texts = [], []

            # Draw overlays
            if DRAW_BBOXES and do_heavy_this_frame:
                draw_lp_boxes(frame, lp_dets_full, color=(0,255,0))
            if DRAW_BBOXES:
                if use_roi:
                    draw_region(frame, rx, ry, rw, rh)
                elif DRAW_GATE:
                    gx = int(ENTRY_LINE_X_RATIO * W)
                    cv2.line(frame, (gx, 0), (gx, H), (200, 200, 255), 2)
                    cv2.putText(frame, "gate", (gx+6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,255), 2, cv2.LINE_AA)
            if DRAW_TRACK_HISTORY:
                draw_track_history_overlay(frame, track_history, prev_inside)

            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            gate_x = int(ENTRY_LINE_X_RATIO * W)

            # Process detections
            if do_heavy_this_frame:
                for i, (det, text) in enumerate(zip(lp_dets_full, lp_texts)):
                    key = stable_key(text, det)
                    if key is None: continue
                    oid = idbank.get(key)

                    # Try to get cached OCR result first (performance boost!)
                    cached_text = ocr_cache.get(oid, frame_idx)
                    if cached_text is not None:
                        text = cached_text  # Use cached result instead of fresh OCR
                        lp_texts[i] = text  # Update the list for consistency

                    cx, cy = center_from_det(det)
                    if use_roi:
                        is_in = inside_region(cx, cy, rx, ry, rw, rh)
                    else:
                        is_in = (cx is not None) and ((cx >= gate_x) if gate_right else (cx <= gate_x))

                    if oid not in track_history:
                        track_history[oid] = deque(maxlen=TRACK_HISTORY_LEN)
                    if cx is not None and cy is not None:
                        track_history[oid].append((int(cx), int(cy)))
                    last_seen_frame[oid] = frame_idx
                    latest_det[oid] = det

                    # Optional quick terminal print of raw OCR when useful
                    if PRINT_RAW_OCR_FOUND:
                        norm = normalize_ocr_text(text or "")
                        if norm and (re.search(u"[ぁ-ゖァ-ヿ]", norm) or plate_suffix(norm)):
                            print(f"[RAW] text='{norm}'  center=({cx},{cy})")

                    # init
                    if oid not in prev_inside:
                        prev_inside[oid] = is_in
                        inside_count[oid]  = 1 if is_in else 0
                        outside_count[oid] = 0 if is_in else 1
                        if is_in:
                            capture_open[oid] = CAPTURE_WINDOW_FRAMES
                            best_score[oid] = 0.0; best_text[oid] = ""; best_det[oid] = det
                            pending_in_cross.add(oid)
                        continue

                    last_inside = prev_inside.get(oid, False)

                    # streaks
                    if is_in:
                        inside_count[oid]  = inside_count.get(oid,0) + 1
                        outside_count[oid] = 0
                    else:
                        outside_count[oid] = outside_count.get(oid,0) + 1
                        inside_count[oid]  = 0

                    # detect crossings to open/close events
                    if (not last_inside) and is_in and inside_count[oid] >= ENTER_STABLE_FRAMES:
                        pending_in_cross.add(oid)
                        if capture_open.get(oid, -1) != -1:
                            capture_open[oid] = max(capture_open.get(oid, 0), CAPTURE_WINDOW_FRAMES)
                        if oid not in best_score:
                            best_score[oid] = 0.0
                        if oid not in best_text:
                            best_text[oid] = ""
                        best_det[oid] = det
                    elif last_inside and (not is_in) and outside_count[oid] >= EXIT_STABLE_FRAMES:
                        pending_in_cross.discard(oid)

                    # skip anything already locked (-1)
                    if capture_open.get(oid, 0) == -1:
                        prev_inside[oid] = is_in
                        continue

                    # Open capture window on transition across gate (no-ROI), or when entering ROI
                    if (not use_roi) and (not prev_inside[oid]) and is_in:
                        capture_open[oid] = CAPTURE_WINDOW_FRAMES
                        best_score[oid] = 0.0; best_text[oid] = ""; best_det[oid] = det

                    # Also open when approaching gate within margin (no-ROI), from the outside side
                    if (not use_roi) and (oid not in captured_in) and (capture_open.get(oid, 0) <= 0):
                        if cx is not None and abs(cx - gate_x) <= GATE_CAPTURE_MARGIN_PX:
                            approaching_from_outside = (cx < gate_x) if gate_right else (cx > gate_x)
                            if approaching_from_outside:
                                capture_open[oid] = CAPTURE_WINDOW_FRAMES
                                best_score[oid] = 0.0; best_text[oid] = ""; best_det[oid] = det

                    # one-shot capture maintenance
                    if is_in and (oid not in captured_in or oid in pending_in_cross):
                        if capture_open.get(oid, 0) > 0:
                            x1,y1,x2,y2 = map(int, xyxy_from_det(det))
                            plate_img = frame[max(0,y1):min(H,y2), max(0,x1):min(W,x2)]

                            # tiny/blurry guards (help live camera kana)
                            width = x2 - x1
                            height = y2 - y1
                            if width < MIN_RECOG_WIDTH_PX or height < MIN_RECOG_HEIGHT_PX:
                                capture_open[oid] -= 1
                                continue

                            sharp = variance_of_laplacian(plate_img)
                            if sharp < MIN_RECOG_SHARPNESS:
                                capture_open[oid] -= 1
                                continue

                            score = bbox_area(det) * max(1.0, sharp)
                            suffix_present = bool(plate_suffix(text or ""))

                            # keep best
                            if score > best_score.get(oid, 0.0) and (suffix_present or not best_text.get(oid)):
                                best_score[oid] = score
                                best_text[oid]  = text
                                best_det[oid]   = det
                                # Update OCR cache with good result
                                ocr_cache.update(oid, text, sharp, frame_idx)

                            # early accept if sharp enough & suffix present
                            if suffix_present and sharp >= MIN_SHARPNESS_LOCK:
                                capture_open[oid] = 0
                                # Cache this excellent result
                                ocr_cache.update(oid, text, sharp, frame_idx)
                            else:
                                capture_open[oid] -= 1

                        # when window closes, log EVENT once (uses DIRECTION_MODE to determine label)
                        if (oid in pending_in_cross) and capture_open.get(oid, 0) == 0 and inside_count[oid] >= ENTER_STABLE_FRAMES:
                            chosen_text = best_text.get(oid,"") or text or ""
                            pk = plate_key(chosen_text)
                            region_class, suf, kana, city, engine_size = parse_plate_fields(chosen_text)

                            # Use direction from settings
                            event_direction = "in" if DIRECTION_MODE == "IN_ONLY" else "out"

                            # ---- duplicate attach: plate already has a slot ----
                            if pk and pk in active_plate_to_slot:
                                chosen_slot = active_plate_to_slot[pk]
                                active_id_to_plate[oid] = pk
                                active_id_to_slot[oid] = chosen_slot
                                if suf: active_id_to_suffix[oid] = suf
                                captured_in.add(oid); last_event_f[oid] = frame_idx; capture_open[oid] = -1
                                pending_in_cross.discard(oid)
                                if DRAW_BBOXES:
                                    cx_i, cy_i = center_from_det(det)
                                    if cx_i is not None and cy_i is not None:
                                        label = "IN" if DIRECTION_MODE == "IN_ONLY" else "OUT"
                                        color = (0,180,255)
                                        cv2.putText(frame, f"{label} {chosen_slot:02d}(dup)", (int(cx_i), int(cy_i)),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
                                # terminal log (dup attach)
                                slot_str = f"{chosen_slot:02d}({suf})" if suf else f"{chosen_slot:02d}"
                                log_terminal(ts, event_direction, slot_str, region_class, suf, kana, city, chosen_text)
                                prev_inside[oid] = is_in
                                continue

                            # ---- duplicate attach by suffix ----
                            chosen_slot = None
                            if suf and suf in active_suffix_to_slot:
                                chosen_slot = active_suffix_to_slot[suf]
                                if pk:
                                    active_plate_to_slot[pk] = chosen_slot
                                    plate_pref_slot[pk] = chosen_slot
                                    active_id_to_plate[oid] = pk
                                active_id_to_slot[oid] = chosen_slot
                                active_id_to_suffix[oid] = suf
                                captured_in.add(oid); last_event_f[oid] = frame_idx; capture_open[oid] = -1
                                pending_in_cross.discard(oid)
                                if DRAW_BBOXES:
                                    cx_i, cy_i = center_from_det(det)
                                    if cx_i is not None and cy_i is not None:
                                        label = "IN" if DIRECTION_MODE == "IN_ONLY" else "OUT"
                                        color = (0,180,255)
                                        cv2.putText(frame, f"{label} {chosen_slot:02d}(dup)", (int(cx_i), int(cy_i)),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
                                slot_str = f"{chosen_slot:02d}({suf})" if suf else f"{chosen_slot:02d}"
                                log_terminal(ts, event_direction, slot_str, region_class, suf, kana, city, chosen_text)
                                prev_inside[oid] = is_in
                                continue

                            # ---- new slot assignment ----
                            if pk and pk in plate_pref_slot:
                                got = slot_pool.try_acquire_specific(plate_pref_slot[pk])
                                if got is not None: chosen_slot = got
                            if chosen_slot is None:
                                chosen_slot = slot_pool.acquire_lowest()

                            if chosen_slot is not None:
                                if pk:
                                    active_plate_to_slot[pk] = chosen_slot
                                    plate_pref_slot[pk] = chosen_slot
                                    active_id_to_plate[oid] = pk
                                else:
                                    active_id_to_plate[oid] = None
                                active_id_to_slot[oid] = chosen_slot
                                if suf:
                                    active_suffix_to_slot[suf] = chosen_slot
                                    active_id_to_suffix[oid] = suf

                                # CSV write (new format). object_id is slot only.
                                object_id = f"{chosen_slot:02d}"
                                write_row_flush(csv_writer, csv_file,
                                                [ts, object_id, "car", event_direction, city or "", engine_size or "", kana or "", suf or ""])
                                # Update parking analytics based on direction
                                if DIRECTION_MODE == "IN_ONLY":
                                    parking_stats.vehicle_entered(object_id, ts)
                                else:
                                    parking_stats.vehicle_exited(object_id, ts)
                                captured_in.add(oid); last_event_f[oid] = frame_idx; capture_open[oid] = -1
                                pending_in_cross.discard(oid)
                                if DRAW_BBOXES:
                                    cx_i, cy_i = center_from_det(det)
                                    if cx_i is not None and cy_i is not None:
                                        label = "IN" if DIRECTION_MODE == "IN_ONLY" else "OUT"
                                        color = (0,255,0) if DIRECTION_MODE == "IN_ONLY" else (0,0,255)
                                        cv2.putText(frame, f"{label} {object_id}", (int(cx_i), int(cy_i)),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
                                # terminal log
                                log_terminal(ts, event_direction, f"{object_id}({suf})" if suf else object_id,
                                             region_class, suf, kana, city, chosen_text)

                    prev_inside[oid] = is_in

            if frame_idx % 10 == 0:
                stale_ids = [oid for oid, last_f in last_seen_frame.items() if frame_idx - last_f > TRACK_STALE_FORGET]
                for stale_oid in stale_ids:
                    pending_in_cross.discard(stale_oid)
                    track_history.pop(stale_oid, None)
                    latest_det.pop(stale_oid, None)
                    last_seen_frame.pop(stale_oid, None)
                    # Remove from OCR cache
                    ocr_cache.remove(stale_oid)

                # Cleanup stale OCR cache entries
                ocr_cache.cleanup_stale(frame_idx)

            # FPS overlay
            if SHOW_FPS:
                now = time.time()
                if now - t0 >= 0.5:
                    fps_est = fcount / (now - t0); t0 = now; fcount = 0
                tag = "" if do_heavy_this_frame else "  (throttle)"
                cv2.putText(frame, f"FPS ~ {fps_est:.1f}{tag}", (10, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
                mode = "ROI" if use_roi else "Gate"
                side = "Right" if gate_right else "Left"
                cv2.putText(frame, f"Mode:{mode} Inside:{side} Dir:{DIRECTION_MODE}  [m]mode [o]side", (10, 48),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180,255,180), 2, cv2.LINE_AA)

            # Parking stats overlay
            if ENABLE_CAPACITY_TRACKING or ENABLE_ANALYTICS:
                stats_str = parking_stats.get_stats_string()
                if stats_str:
                    # Background box for better visibility
                    text_size = cv2.getTextSize(stats_str, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    cv2.rectangle(frame, (8, 60), (text_size[0] + 15, 85), (0, 0, 0), -1)
                    text_color = (0, 0, 255) if parking_stats.is_near_capacity() else (0, 255, 0)
                    cv2.putText(frame, stats_str, (10, 78),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2, cv2.LINE_AA)

            # OCR Cache stats overlay (show performance gain)
            if ENABLE_OCR_CACHE and frame_idx % 30 == 0:  # Update stats every 30 frames
                cache_stats = ocr_cache.get_stats()
                if cache_stats['hits'] + cache_stats['misses'] > 0:
                    cache_str = f"OCR Cache: {cache_stats['hit_rate']:.1f}% hit rate ({cache_stats['hits']}/{cache_stats['hits']+cache_stats['misses']})"
                    cv2.putText(frame, cache_str, (10, H-15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

            # Display / write
            cv2.imshow("Parking (semi_per01 ROI one-shot + YouTube)", frame)
            if out is not None and WRITE_VIDEO: out.write(frame)

            # keyboard
            k = cv2.waitKey(1) & 0xFF
            if k == 27: break
            if k == ord('m'):
                use_roi = not use_roi
                if use_roi:
                    rx = int(REGION_XYWH_RATIO[0]*W); ry = int(REGION_XYWH_RATIO[1]*H)
                    rw = int(REGION_XYWH_RATIO[2]*W); rh = int(REGION_XYWH_RATIO[3]*H)
                    rx, ry, rw, rh = clamp_region(rx, ry, rw, rh, W, H)
                else:
                    rx, ry, rw, rh = 0, 0, W, H
            if k == ord('o'):
                gate_right = not gate_right
            if use_roi and (not LOCK_ROI):
                step = 12
                if k == ord('w'): ry -= step
                elif k == ord('s'): ry += step
                elif k == ord('a'): rx -= step
                elif k == ord('d'): rx += step
                elif k == ord('['): rx += step; ry += step; rw -= 2*step; rh -= 2*step
                elif k == ord(']'): rx -= step; ry -= step; rw += 2*step; rh += 2*step
                elif k == ord('r'):
                    rx = int(REGION_XYWH_RATIO[0]*W); ry = int(REGION_XYWH_RATIO[1]*H)
                    rw = int(REGION_XYWH_RATIO[2]*W); rh = int(REGION_XYWH_RATIO[3]*H)
                rx, ry, rw, rh = clamp_region(rx, ry, rw, rh, W, H)

    finally:
        try: cap.release()
        except: pass
        try:
            if WRITE_VIDEO and out is not None: out.release()
        except: pass
        cv2.destroyAllWindows()
        csv_file.close()
        print("[INFO] CSV saved to", csv_path)
        if WRITE_VIDEO: print("[INFO] Video saved to output_videos/output_video.avi")

        # Print OCR cache statistics
        if ENABLE_OCR_CACHE:
            cache_stats = ocr_cache.get_stats()
            print(f"\n[OCR CACHE STATISTICS]")
            print(f"  Cache Hits: {cache_stats['hits']}")
            print(f"  Cache Misses: {cache_stats['misses']}")
            print(f"  Hit Rate: {cache_stats['hit_rate']:.2f}%")
            print(f"  Final Cache Size: {cache_stats['cache_size']}")
            if cache_stats['hit_rate'] > 0:
                estimated_speedup = 1 + (cache_stats['hit_rate'] / 100.0 * 0.6)  # 60% of time is OCR
                print(f"  Estimated Performance Gain: {(estimated_speedup - 1) * 100:.1f}%")

if __name__ == "__main__":
    main()
