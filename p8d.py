# -*- coding: utf-8 -*-
# per8_optimized.py — per7_format + intelligent gate flip handling
# - Smart state recalculation when gate orientation changes (press 'o')
# - Prevents duplicate events and tracking confusion during orientation flip
# - Preserves tracking continuity: vehicles still inside remain tracked
# - Auto-exit only vehicles that are now on the wrong side after flip

import os, cv2, csv, re, time, numpy as np
from collections import deque, defaultdict
from datetime import datetime
from licence_plate_detection import LicencePlateDetection
import supervision as sv

# ===================== User settings =====================
# Choose exactly ONE:
USE_CAMERA               = True
USE_FILE                 = False
USE_YOUTUBE              = False

# File source
INPUT_VIDEO_PATH         = "input_videos/Trim03.mp4"

# YouTube source (VOD or live). If enabled, overrides camera/file.
YOUTUBE_URL              = "https://www.youtube.com/watch?v=jqtsC5BYlIk"
YOUTUBE_MAX_HEIGHT       = 480   # progressive MP4 at or below this height

# Terminal logging toggles
PRINT_EVENTS_TO_TERMINAL = True   # <-- print every IN/OUT nicely to terminal
PRINT_RAW_OCR_FOUND      = False  # <-- print per-detection OCR strings when they contain kana/suffix
PRINT_GATE_CROSSINGS     = True   # <-- print gate crossing detections with direction
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
GATE_CAPTURE_MARGIN_PX   = 60     # start capture window when within this many px of gate
DRAW_GATE                = True
GATE_INSIDE_IS_RIGHT     = True   # True: right of line means "inside"; False: left is inside

# Stability & gaps
ENTER_STABLE_FRAMES      = 1
EXIT_STABLE_FRAMES       = 4
MIN_EVENT_GAP_FRAMES     = 10

# One-shot window (fallback in case sharpness never crosses threshold)
CAPTURE_WINDOW_FRAMES    = 6

# While car remains inside, only process heavy every N frames
INSIDE_PROCESS_EVERY_N   = 8
DETECT_EVERY_N           = 1     # run detector every N frames

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

# ByteTrack settings
BYTETRACK_THRESHOLD      = 0.25
BYTETRACK_LOST_BUFFER    = 30
BYTETRACK_MATCH_THRESH   = 0.5
BYTETRACK_MIN_FRAMES     = 1

# Track visualization
TRACK_HISTORY_LEN        = 30
DRAW_TRACK_HISTORY       = True

# Direction detection
DIRECTION_HISTORY_POINTS = 5   # Use last N points to determine direction
LINE_CROSSING_MIN_FRAMES = 2   # Frames LineZone requires on the new side before counting
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

def generate_color_by_id(track_id):
    """Generate unique color for each tracker ID."""
    if track_id is None or track_id < 0:
        return (128, 128, 128)
    np.random.seed(int(track_id) * 17)
    hue = int((track_id * 47) % 180)
    hsv_color = np.uint8([[[hue, 255, 255]]])
    bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
    return tuple(map(int, bgr_color))

def get_trajectory_direction(track_history_deque, n_points=5):
    """
    Analyze the last n_points to determine overall movement direction.
    Returns: 'left_to_right', 'right_to_left', or None
    """
    if len(track_history_deque) < n_points:
        return None

    points = list(track_history_deque)[-n_points:]
    # Calculate average horizontal movement
    dx_sum = 0
    for i in range(len(points) - 1):
        dx_sum += points[i + 1][0] - points[i][0]

    if abs(dx_sum) < 5:  # Not enough horizontal movement
        return None

    return 'left_to_right' if dx_sum > 0 else 'right_to_left'

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

# ---------- main ----------
def main():
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

    # Initialize ByteTrack
    byte_tracker = sv.ByteTrack(
        track_activation_threshold=BYTETRACK_THRESHOLD,
        lost_track_buffer=BYTETRACK_LOST_BUFFER,
        minimum_matching_threshold=BYTETRACK_MATCH_THRESH,
        frame_rate=int(fps),
        minimum_consecutive_frames=BYTETRACK_MIN_FRAMES
    )

    def rebuild_line_zone():
        gx = int(ENTRY_LINE_X_RATIO * W)
        start_pt = sv.Point(x=gx, y=0 if gate_right else H)
        end_pt = sv.Point(x=gx, y=H if gate_right else 0)
        zone = sv.LineZone(
            start=start_pt,
            end=end_pt,
            minimum_crossing_threshold=max(1, LINE_CROSSING_MIN_FRAMES)
        )
        annot = sv.LineZoneAnnotator(
            thickness=2,
            text_thickness=2,
            text_scale=0.6,
            text_color=sv.Color(r=0, g=0, b=0),
            color=sv.Color(r=200, g=200, b=255)
        )
        return gx, zone, annot

    # ROI (or full frame if use_roi=False)
    if use_roi:
        rx = int(REGION_XYWH_RATIO[0]*W); ry = int(REGION_XYWH_RATIO[1]*H)
        rw = int(REGION_XYWH_RATIO[2]*W); rh = int(REGION_XYWH_RATIO[3]*H)
        rx, ry, rw, rh = clamp_region(rx, ry, rw, rh, W, H)
    else:
        rx, ry, rw, rh = 0, 0, W, H

    gate_x = int(ENTRY_LINE_X_RATIO * W)
    if use_roi:
        line_zone = None
        line_zone_annotator = None
    else:
        gate_x, line_zone, line_zone_annotator = rebuild_line_zone()

    # Track state management (using tracker_id instead of custom ID)
    tracker_id_map = {}  # stable_key -> tracker_id mapping (for continuity)
    prev_inside = {}
    inside_count, outside_count, last_event_f = {}, {}, {}
    track_history = defaultdict(lambda: deque(maxlen=TRACK_HISTORY_LEN))

    slot_pool = SlotPool(SLOT_COUNT)
    plate_pref_slot, active_plate_to_slot, active_id_to_plate = {}, {}, {}
    active_suffix_to_slot = {}
    active_id_to_slot = {}
    active_id_to_suffix = {}

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

            raw_dets = []
            raw_texts = []

            if do_heavy_this_frame:
                # Choose detection region: ROI if enabled, else a vertical band around the gate
                if use_roi:
                    d_rx, d_ry, d_rw, d_rh = rx, ry, rw, rh
                else:
                    band_w = max(160, int(W * GATE_DET_BAND_RATIO))
                    d_rx = max(0, min(W-20, gate_x - band_w//2))
                    d_ry = 0
                    d_rw = min(band_w, W - d_rx)
                    d_rh = H
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
                raw_dets = []
                if scale != 1.0:
                    inv_sx = 1.0/scale; inv_sy = 1.0/scale
                    for d in roi_dets:
                        d_scaled = scale_det(d, inv_sx, inv_sy)
                        raw_dets.append(offset_det(d_scaled, d_rx, d_ry))
                else:
                    for d in roi_dets:
                        raw_dets.append(offset_det(d, d_rx, d_ry))
                raw_texts = lp_texts

            # Convert to supervision format and update ByteTrack (even if no detections)
            xyxy_list = []
            for det in raw_dets:
                xy = xyxy_from_det(det)
                if xy:
                    xyxy_list.append(list(xy))

            if xyxy_list:
                det_xyxy = np.array(xyxy_list, dtype=np.float32)
                det_conf = np.ones(len(xyxy_list), dtype=np.float32) * 0.9
                det_texts = np.array(raw_texts, dtype=object)
            else:
                det_xyxy = np.empty((0, 4), dtype=np.float32)
                det_conf = np.empty((0,), dtype=np.float32)
                det_texts = np.empty((0,), dtype=object)

            detections_sv = sv.Detections(
                xyxy=det_xyxy,
                confidence=det_conf,
                data={"ocr_text": det_texts}
            )
            tracked_detections = byte_tracker.update_with_detections(detections_sv)

            tracked_texts = tracked_detections.data.get("ocr_text")
            if tracked_texts is None:
                tracked_texts = np.empty((0,), dtype=object)

            lp_dets_full = []
            lp_texts = []
            for i in range(len(tracked_detections)):
                xy_vals = tracked_detections.xyxy[i].tolist()
                tid_val = int(tracked_detections.tracker_id[i]) if tracked_detections.tracker_id is not None else -1
                det_text = tracked_texts[i] if i < len(tracked_texts) else ""
                if det_text is None:
                    det_text = ""
                if not isinstance(det_text, str):
                    det_text = str(det_text)
                lp_dets_full.append(xy_vals + [tid_val])
                lp_texts.append(det_text)

            crossed_in_mask = np.zeros(len(tracked_detections), dtype=bool)
            crossed_out_mask = np.zeros(len(tracked_detections), dtype=bool)
            if (not use_roi) and line_zone is not None and len(tracked_detections) > 0:
                crossed_in_mask, crossed_out_mask = line_zone.trigger(tracked_detections)

            # Update track history and draw overlays
            if do_heavy_this_frame:
                for det in lp_dets_full:
                    xy = xyxy_from_det(det)
                    if not xy: continue
                    x1,y1,x2,y2 = map(int, xy)
                    # Get tracker_id from det (appended above)
                    tid = det[4] if isinstance(det, list) and len(det) > 4 else -1
                    if tid >= 0:
                        cx, cy = int((x1+x2)/2), int((y1+y2)/2)
                        track_history[tid].append((cx, cy))

                    # Draw bounding box
                    if DRAW_BBOXES:
                        color = generate_color_by_id(tid)
                        cv2.rectangle(frame, (x1,y1), (x2,y2), color, BOX_THICK)
                        # Draw track history
                        if DRAW_TRACK_HISTORY and tid >= 0:
                            pts = list(track_history[tid])
                            if len(pts) >= 2:
                                for j in range(1, len(pts)):
                                    cv2.line(frame, pts[j-1], pts[j], color, 2)
                                cv2.circle(frame, pts[-1], 3, color, -1)
            if DRAW_BBOXES:
                if use_roi:
                    draw_region(frame, rx, ry, rw, rh)
                elif DRAW_GATE:
                    if line_zone is not None and line_zone_annotator is not None:
                        frame = line_zone_annotator.annotate(frame, line_zone)
                    else:
                        cv2.line(frame, (gate_x, 0), (gate_x, H), (200, 200, 255), 2)
                        cv2.putText(frame, "gate", (gate_x+6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,255), 2, cv2.LINE_AA)

            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Process detections with trajectory-based gate crossing
            if do_heavy_this_frame:
                for idx, (det, text) in enumerate(zip(lp_dets_full, lp_texts)):
                    # Get tracker_id from det (appended by ByteTrack above)
                    oid = det[4] if isinstance(det, list) and len(det) > 4 else -1
                    if oid < 0: continue  # Skip untracked detections

                    cx, cy = center_from_det(det)
                    if use_roi:
                        is_in = inside_region(cx, cy, rx, ry, rw, rh)
                    else:
                        zone_state = None
                        if line_zone is not None:
                            hist = line_zone.crossing_state_history.get(oid)
                            if hist and len(hist) > 0:
                                zone_state = bool(hist[-1])
                        if zone_state is None:
                            is_in = (cx is not None) and ((cx >= gate_x) if gate_right else (cx <= gate_x))
                        else:
                            is_in = zone_state

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
                        continue

                    # streaks
                    if is_in:
                        inside_count[oid]  = inside_count.get(oid,0) + 1
                        outside_count[oid] = 0
                    else:
                        outside_count[oid] = outside_count.get(oid,0) + 1
                        inside_count[oid]  = 0

                    # LineZone crossing flags (gate mode)
                    crossed_in_now = (not use_roi) and bool(crossed_in_mask[idx]) if idx < len(crossed_in_mask) else False
                    crossed_out_now = (not use_roi) and bool(crossed_out_mask[idx]) if idx < len(crossed_out_mask) else False
                    handled_exit_via_cross = False

                    if crossed_in_now:
                        if PRINT_GATE_CROSSINGS:
                            print(f"[GATE] tracker_id={oid} crossed → IN, was_captured={'YES' if oid in captured_in else 'NO'}, cx={cx:.0f}, gate_right={gate_right}")
                        if oid in captured_in:
                            pk_now = active_id_to_plate.get(oid)
                            slot_to_free = None
                            if pk_now and pk_now in active_plate_to_slot:
                                slot_to_free = active_plate_to_slot.pop(pk_now, None)
                            if slot_to_free is None:
                                slot_to_free = active_id_to_slot.pop(oid, None)

                            if slot_to_free is not None:
                                chosen_text = best_text.get(oid, "") or text or ""
                                region_class, suf_out, kana, city, engine_size = parse_plate_fields(chosen_text)
                                object_id = f"{slot_to_free:02d}"
                                write_row_flush(csv_writer, csv_file,
                                                [ts, object_id, "car", "out", city or "", engine_size or "", kana or "", suf_out or ""])
                                slot_pool.release(slot_to_free)
                                if DRAW_BBOXES:
                                    cx_o, cy_o = center_from_det(det)
                                    if cx_o is not None and cy_o is not None:
                                        cv2.putText(frame, f"OUT {object_id} (auto)", (int(cx_o), int(cy_o)-20),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2, cv2.LINE_AA)
                                log_terminal(ts, "out", f"{object_id}({suf_out})" if suf_out else object_id,
                                             region_class, suf_out, kana, city, chosen_text)
                                if suf_out and active_suffix_to_slot.get(suf_out) == slot_to_free:
                                    active_suffix_to_slot.pop(suf_out, None)
                            active_id_to_plate.pop(oid, None)
                            active_id_to_suffix.pop(oid, None)
                            captured_in.discard(oid)

                        capture_open[oid] = CAPTURE_WINDOW_FRAMES
                        best_score[oid] = 0.0
                        best_text[oid] = ""
                        best_det[oid] = det
                        last_event_f[oid] = frame_idx

                    if crossed_out_now:
                        if PRINT_GATE_CROSSINGS:
                            print(f"[GATE] tracker_id={oid} crossed → OUT, was_captured={'YES' if oid in captured_in else 'NO'}, cx={cx:.0f}, gate_right={gate_right}")
                        if oid in captured_in:
                            pk_now = active_id_to_plate.get(oid)
                            slot_to_free = None
                            if pk_now and pk_now in active_plate_to_slot:
                                slot_to_free = active_plate_to_slot.pop(pk_now, None)
                            if slot_to_free is None:
                                slot_to_free = active_id_to_slot.pop(oid, None)

                            if slot_to_free is not None:
                                chosen_text = best_text.get(oid, "") or text or ""
                                region_class, suf_out, kana, city, engine_size = parse_plate_fields(chosen_text)
                                object_id = f"{slot_to_free:02d}"
                                write_row_flush(csv_writer, csv_file,
                                                [ts, object_id, "car", "out", city or "", engine_size or "", kana or "", suf_out or ""])
                                slot_pool.release(slot_to_free)
                                last_event_f[oid] = frame_idx
                                if DRAW_BBOXES:
                                    cx_o, cy_o = center_from_det(det)
                                    if cx_o is not None and cy_o is not None:
                                        cv2.putText(frame, f"OUT {object_id}", (int(cx_o), int(cy_o)),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                                log_terminal(ts, "out", f"{object_id}({suf_out})" if suf_out else object_id,
                                             region_class, suf_out, kana, city, chosen_text)
                                if suf_out and active_suffix_to_slot.get(suf_out) == slot_to_free:
                                    active_suffix_to_slot.pop(suf_out, None)

                            active_id_to_plate.pop(oid, None)
                            active_id_to_suffix.pop(oid, None)
                            captured_in.discard(oid)
                            handled_exit_via_cross = True

                        capture_open[oid] = 0

                    # Skip locked captures UNLESS we just crossed (which reopened the window)
                    if capture_open.get(oid, 0) == -1 and not (crossed_in_now or crossed_out_now):
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
                    if is_in and (oid not in captured_in):
                        if capture_open.get(oid, 0) > 0:
                            x1,y1,x2,y2 = map(int, xyxy_from_det(det))
                            plate_img = frame[max(0,y1):min(H,y2), max(0,x1):min(W,x2)]

                            # tiny/blurry guards (help live camera kana)
                            min_side = min(y2-y1, x2-x1)
                            if min_side < 18:
                                capture_open[oid] -= 1
                                continue

                            sharp = variance_of_laplacian(plate_img)
                            if sharp < 25.0:
                                capture_open[oid] -= 1
                                continue

                            score = bbox_area(det) * max(1.0, sharp)

                            # keep best
                            if score > best_score.get(oid, 0.0) and plate_suffix(text or ""):
                                best_score[oid] = score
                                best_text[oid]  = text
                                best_det[oid]   = det

                            # early accept if sharp enough & suffix present
                            if plate_suffix(text or "") and sharp >= MIN_SHARPNESS_LOCK:
                                capture_open[oid] = 0
                            else:
                                capture_open[oid] -= 1

                        # when window closes, log IN once
                        if capture_open.get(oid, 0) == 0 and inside_count[oid] >= ENTER_STABLE_FRAMES:
                            chosen_text = best_text.get(oid,"") or text or ""
                            pk = plate_key(chosen_text)
                            region_class, suf, kana, city, engine_size = parse_plate_fields(chosen_text)

                            # ---- duplicate attach: plate already has a slot ----
                            if pk and pk in active_plate_to_slot:
                                chosen_slot = active_plate_to_slot[pk]
                                active_id_to_plate[oid] = pk
                                active_id_to_slot[oid] = chosen_slot
                                if suf: active_id_to_suffix[oid] = suf
                                captured_in.add(oid); last_event_f[oid] = frame_idx; capture_open[oid] = -1
                                if DRAW_BBOXES:
                                    cx_i, cy_i = center_from_det(det)
                                    if cx_i is not None and cy_i is not None:
                                        cv2.putText(frame, f"IN {chosen_slot:02d}(dup)", (int(cx_i), int(cy_i)),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,180,255), 2, cv2.LINE_AA)
                                # terminal log (dup attach)
                                slot_str = f"{chosen_slot:02d}({suf})" if suf else f"{chosen_slot:02d}"
                                log_terminal(ts, "in", slot_str, region_class, suf, kana, city, chosen_text)
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
                                if DRAW_BBOXES:
                                    cx_i, cy_i = center_from_det(det)
                                    if cx_i is not None and cy_i is not None:
                                        cv2.putText(frame, f"IN {chosen_slot:02d}(dup)", (int(cx_i), int(cy_i)),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,180,255), 2, cv2.LINE_AA)
                                slot_str = f"{chosen_slot:02d}({suf})" if suf else f"{chosen_slot:02d}"
                                log_terminal(ts, "in", slot_str, region_class, suf, kana, city, chosen_text)
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
                                                [ts, object_id, "car", "in", city or "", engine_size or "", kana or "", suf or ""])
                                captured_in.add(oid); last_event_f[oid] = frame_idx; capture_open[oid] = -1
                                if DRAW_BBOXES:
                                    cx_i, cy_i = center_from_det(det)
                                    if cx_i is not None and cy_i is not None:
                                        cv2.putText(frame, f"IN {object_id}", (int(cx_i), int(cy_i)),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv2.LINE_AA)
                                # terminal log
                                log_terminal(ts, "in", f"{object_id}({suf})" if suf else object_id,
                                             region_class, suf, kana, city, chosen_text)

                    # EXIT handling (fallback for ROI mode or when gate crossing wasn't detected)
                    if (not handled_exit_via_cross) and prev_inside[oid] and (not is_in) and outside_count[oid] >= EXIT_STABLE_FRAMES:
                        if (frame_idx - last_event_f.get(oid, -10**9) >= MIN_EVENT_GAP_FRAMES) and (oid in captured_in):
                            pk_now = active_id_to_plate.get(oid)
                            slot_to_free = None
                            if pk_now and pk_now in active_plate_to_slot:
                                slot_to_free = active_plate_to_slot.pop(pk_now, None)
                            if slot_to_free is None:
                                slot_to_free = active_id_to_slot.pop(oid, None)

                            if slot_to_free is not None:
                                chosen_text = best_text.get(oid,"") or text or ""
                                region_class, suf_out, kana, city, engine_size = parse_plate_fields(chosen_text)
                                # CSV write (new format). object_id is slot only.
                                object_id = f"{slot_to_free:02d}"
                                write_row_flush(csv_writer, csv_file,
                                                [ts, object_id, "car", "out", city or "", engine_size or "", kana or "", suf_out or ""])
                                slot_pool.release(slot_to_free)
                                last_event_f[oid] = frame_idx
                                if DRAW_BBOXES:
                                    cx_o, cy_o = center_from_det(det)
                                    if cx_o is not None and cy_o is not None:
                                        cv2.putText(frame, f"OUT {object_id}", (int(cx_o), int(cy_o)),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv2.LINE_AA)
                                # terminal log
                                log_terminal(ts, "out", f"{object_id}({suf_out})" if suf_out else object_id,
                                             region_class, suf_out, kana, city, chosen_text)

                                if suf_out and active_suffix_to_slot.get(suf_out) == slot_to_free:
                                    active_suffix_to_slot.pop(suf_out, None)

                            # clear id mappings/state
                            active_id_to_plate.pop(oid, None)
                            active_id_to_suffix.pop(oid, None)
                            for dct in (capture_open, best_score, best_text, best_det):
                                dct.pop(oid, None)
                            if oid in captured_in: captured_in.remove(oid)

                    prev_inside[oid] = is_in

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
                cv2.putText(frame, f"Mode:{mode} Inside:{side}  [m]mode [o]side", (10, 48),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180,255,180), 2, cv2.LINE_AA)

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
                    line_zone = None
                    line_zone_annotator = None
                else:
                    rx, ry, rw, rh = 0, 0, W, H
                    gate_x, line_zone, line_zone_annotator = rebuild_line_zone()
            if k == ord('o'):
                # Flip gate orientation with intelligent state recalculation
                gate_right = not gate_right
                print(f"\n[MODE CHANGE] Gate orientation flipped. Inside is now: {'RIGHT' if gate_right else 'LEFT'}")
                print("[MODE CHANGE] Recalculating states for tracked vehicles...")

                if use_roi:
                    gate_x = int(ENTRY_LINE_X_RATIO * W)
                    line_zone = None
                    line_zone_annotator = None
                else:
                    gate_x, line_zone, line_zone_annotator = rebuild_line_zone()

                # For each tracked vehicle, determine if it's on the "wrong" side now
                # If a vehicle was "inside" but is now on the "outside" after flip, log OUT
                oids_to_exit = []
                oids_to_keep = []

                for oid in list(captured_in):
                    # Get last known position from track history
                    if oid in track_history and len(track_history[oid]) > 0:
                        cx, cy = track_history[oid][-1]  # last known position

                        # Check if vehicle is on "inside" side with NEW orientation
                        is_inside_now = (cx >= gate_x) if gate_right else (cx <= gate_x)

                        if not is_inside_now:
                            # Vehicle is now on "outside" - it should exit
                            oids_to_exit.append(oid)
                            print(f"  → Vehicle {oid} now on outside (cx={cx:.0f}, gate={gate_x}), auto-exit")
                        else:
                            # Vehicle is still on "inside" - keep it tracked
                            oids_to_keep.append(oid)
                            print(f"  → Vehicle {oid} still inside (cx={cx:.0f}, gate={gate_x}), keeping")
                    else:
                        # No position info, safer to exit
                        oids_to_exit.append(oid)
                        print(f"  → Vehicle {oid} has no position history, auto-exit")

                # Exit vehicles that are now on the wrong side
                for oid in oids_to_exit:
                    pk_now = active_id_to_plate.get(oid)
                    slot_to_free = None
                    if pk_now and pk_now in active_plate_to_slot:
                        slot_to_free = active_plate_to_slot.pop(pk_now, None)
                    if slot_to_free is None:
                        slot_to_free = active_id_to_slot.pop(oid, None)

                    if slot_to_free is not None:
                        chosen_text = best_text.get(oid, "")
                        region_class, suf_out, kana, city, engine_size = parse_plate_fields(chosen_text)
                        object_id = f"{slot_to_free:02d}"
                        write_row_flush(csv_writer, csv_file,
                                        [ts, object_id, "car", "out", city or "", engine_size or "", kana or "", suf_out or ""])
                        slot_pool.release(slot_to_free)
                        print(f"    ✓ Logged exit: slot {object_id}, suffix={suf_out or 'N/A'}")

                        if suf_out and active_suffix_to_slot.get(suf_out) == slot_to_free:
                            active_suffix_to_slot.pop(suf_out, None)

                    # Clear state for exited vehicle
                    active_id_to_plate.pop(oid, None)
                    active_id_to_suffix.pop(oid, None)
                    captured_in.discard(oid)

                    # Clear tracking state for this vehicle
                    for dct in (capture_open, best_score, best_text, best_det,
                                prev_inside, inside_count, outside_count):
                        dct.pop(oid, None)

                # For vehicles still inside, recalculate their inside/outside state
                for oid in oids_to_keep:
                    if oid in track_history and len(track_history[oid]) > 0:
                        cx, cy = track_history[oid][-1]
                        # Update prev_inside based on new orientation
                        is_inside_now = (cx >= gate_x) if gate_right else (cx <= gate_x)
                        prev_inside[oid] = is_inside_now
                        # Reset streaks
                        if is_inside_now:
                            inside_count[oid] = 1
                            outside_count[oid] = 0
                        else:
                            inside_count[oid] = 0
                            outside_count[oid] = 1

                print(f"[MODE CHANGE] State recalculated. {len(oids_to_exit)} exited, {len(oids_to_keep)} kept.\n")
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

if __name__ == "__main__":
    main()
