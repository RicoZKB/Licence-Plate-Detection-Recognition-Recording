# -*- coding: utf-8 -*-
# per5_gate_optimized.py — Gate-based license plate tracking for parking entry/exit
# Optimized for gate-line crossing detection with visual feedback

import os, cv2, csv, re, time
from datetime import datetime
from licence_plate_detection import LicencePlateDetection

# ===================== GATE MODE SETTINGS =====================
# Source selection (choose exactly one)
USE_CAMERA               = True
USE_FILE                 = False
USE_YOUTUBE              = False

INPUT_VIDEO_PATH         = "input_videos/Trim03.mp4"
YOUTUBE_URL              = "https://www.youtube.com/watch?v=jqtsC5BYlIk"
YOUTUBE_MAX_HEIGHT       = 480

# Gate configuration
GATE_MODE                = True   # True = use gate line, False = use ROI box
ENTRY_LINE_X_RATIO       = 0.50   # Position of vertical gate line (0.0-1.0)
GATE_INSIDE_IS_RIGHT     = True   # True = right side is inside, False = left side
GATE_CAPTURE_MARGIN_PX   = 80     # Start capture when within this distance
GATE_DETECTION_BAND      = 0.60   # Width of detection band around gate (0.0-1.0)

# Visual feedback
DRAW_GATE                = True   # Draw the gate line
DRAW_DETECTION_ZONE      = True   # Draw the detection band
GATE_LINE_COLOR          = (0, 255, 255)     # Cyan
INSIDE_ZONE_COLOR        = (100, 255, 100)   # Light green
OUTSIDE_ZONE_COLOR       = (255, 200, 100)   # Light blue
DETECTION_BAND_COLOR     = (200, 150, 255)   # Light purple

# Terminal logging
PRINT_EVENTS_TO_TERMINAL = True
PRINT_RAW_OCR_FOUND      = False
PASS_OCR_DEBUG_TO_DETECT = False

# Event detection tuning
ENTER_STABLE_FRAMES      = 1      # Frames to confirm entry
EXIT_STABLE_FRAMES       = 4      # Frames to confirm exit
MIN_EVENT_GAP_FRAMES     = 10     # Minimum frames between events
CAPTURE_WINDOW_FRAMES    = 8      # Frames to capture best plate reading
MIN_SHARPNESS_LOCK       = 60.0   # Sharpness threshold for instant acceptance

# Performance settings
INSIDE_PROCESS_EVERY_N   = 8      # Process every N frames when car inside
DETECT_EVERY_N           = 1      # Run detector every N frames
TARGET_WIDTH             = 640
ROI_INFER_MAX_W          = 640
WRITE_VIDEO              = False
SHOW_FPS                 = True
BOX_THICK                = 2

# Parking slots
SLOT_COUNT               = 10
START_ID_AT              = 100
# =========================================================

cv2.setUseOptimized(True)
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")

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

# Japanese license plate parsing
CITY_KANA_RE = re.compile(u"^\\s*([^\\s\\d]+)\\s*[0-9]{2,4}.*?([ぁ-ゖァ-ヿA-Za-z])", re.UNICODE)
PLATE_SUFFIX_RE = re.compile(r"(\d{2,3})\s*[-−–—ーｰ~〜－]\s*(\d{2})")
REGION_WITH_NUM_RE = re.compile(u"([ぁ-ゖァ-ヿ一-龯A-Za-z]+)\\s*([0-9]{2,4})", re.UNICODE)

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
    suf = plate_suffix(t)
    if not suf: return None
    city_match = CITY_KANA_RE.search(t)
    city = city_match.group(1) if city_match else '?'
    return f"{city}|{suf}"

def parse_plate_fields(text):
    """Returns: region_class, suffix, kana, city, engine_size"""
    if not text: return None, None, None, None, None
    t = normalize_ocr_text(text)
    suf = plate_suffix(t)
    city = None; region_class = None; engine_size = None
    m = REGION_WITH_NUM_RE.search(t)
    if m:
        city = m.group(1)
        engine_size = m.group(2)
        region_class = f"{city}{engine_size}"
    kana = None; kana_matches = re.findall(u"[ぁ-ゖァ-ヿ]", t)
    if kana_matches: kana = kana_matches[-1]
    return region_class, suf, kana, city, engine_size

def open_daily_csv():
    date_str = datetime.now().strftime("%Y%m%d")
    os.makedirs("logs", exist_ok=True)
    path = os.path.join("logs", f"parking_log_{date_str}.csv")
    new = not os.path.exists(path)
    f = open(path, "a", newline="", encoding="utf-8", buffering=1)
    w = csv.writer(f)
    if new:
        w.writerow(["timestamp","object_id","vehicle_type","direction","city","engine_size","kana","four-digit number"])
        f.flush()
    return f, w, path

def write_row_flush(w, f, row):
    w.writerow(row); f.flush()
    try: os.fsync(f.fileno())
    except Exception: pass

def log_terminal(event_ts, direction, slot_str, region_class, suffix, kana, city, raw_text):
    if not PRINT_EVENTS_TO_TERMINAL:
        return
    kana_disp = kana if kana else ""
    city_disp = city if city else ""
    rc_disp = region_class if region_class else ""
    suf_disp = suffix if suffix else ""
    print(f"[{event_ts}] {direction.upper():3s}  slot={slot_str:>6s}  region_class='{rc_disp}'  suffix='{suf_disp}'  kana='{kana_disp}'  city='{city_disp}'  raw=\"{raw_text}\"")

# Geometry helpers
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

# ID and slot management
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
    t = text if (text and len(text)>=3) else ""
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

def draw_gate_overlay(frame, gate_x, W, H, gate_right):
    """Draw enhanced gate visualization"""
    # Draw gate line
    if DRAW_GATE:
        cv2.line(frame, (gate_x, 0), (gate_x, H), GATE_LINE_COLOR, 3)
        label = "GATE →" if gate_right else "← GATE"
        cv2.putText(frame, label, (gate_x + 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, GATE_LINE_COLOR, 2, cv2.LINE_AA)
    
    # Draw side labels
    if gate_right:
        cv2.putText(frame, "OUTSIDE", (10, H - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, OUTSIDE_ZONE_COLOR, 2, cv2.LINE_AA)
        cv2.putText(frame, "INSIDE", (gate_x + 20, H - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, INSIDE_ZONE_COLOR, 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, "INSIDE", (10, H - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, INSIDE_ZONE_COLOR, 2, cv2.LINE_AA)
        cv2.putText(frame, "OUTSIDE", (gate_x + 20, H - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, OUTSIDE_ZONE_COLOR, 2, cv2.LINE_AA)

def draw_detection_band(frame, gate_x, W, H, band_ratio):
    """Draw semi-transparent detection band around gate"""
    if not DRAW_DETECTION_ZONE:
        return
    band_w = int(W * band_ratio)
    x1 = max(0, gate_x - band_w // 2)
    x2 = min(W, gate_x + band_w // 2)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, 0), (x2, H), DETECTION_BAND_COLOR, -1)
    cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

def draw_lp_boxes(frame, dets, color=(0,255,0)):
    for d in dets:
        xy = xyxy_from_det(d)
        if not xy: continue
        x1,y1,x2,y2 = map(int, xy)
        cv2.rectangle(frame, (x1,y1), (x2,y2), color, BOX_THICK)

def main():
    # Source selection
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
        print("[INFO] Using YouTube stream")
    else:
        src = INPUT_VIDEO_PATH
        if not os.path.exists(src): raise RuntimeError(f"Input file not found: {src}")

    # Runtime flags
    use_gate = GATE_MODE
    gate_right = GATE_INSIDE_IS_RIGHT

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {src}")

    using_camera = (source_name == "Camera")
    if using_camera:
        try: cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except: pass

    try: cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)
    except: pass

    # License plate detector
    lp_det = LicencePlateDetection(model_path="models/best.pt",
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
    print(f"[INFO] Gate mode: {'ENABLED' if use_gate else 'DISABLED'}")
    print(f"[INFO] Gate position: {ENTRY_LINE_X_RATIO*100:.0f}% across frame")
    print(f"[INFO] Inside zone: {'RIGHT' if gate_right else 'LEFT'} of gate line")
    print(f"[INFO] Detection band width: {GATE_DETECTION_BAND*100:.0f}% of frame")
    print("[INFO] Press 'm' to toggle gate/ROI mode, 'o' to flip inside/outside, ESC to quit")

    idbank = IDBank(START_ID_AT)
    prev_inside = {}
    inside_count, outside_count, last_event_f = {}, {}, {}

    slot_pool = SlotPool(SLOT_COUNT)
    plate_pref_slot, active_plate_to_slot, active_id_to_plate = {}, {}, {}
    active_suffix_to_slot = {}
    active_id_to_slot = {}
    active_id_to_suffix = {}

    capture_open = {}
    best_score = {}
    best_text = {}
    best_det = {}
    captured_in = set()

    frame_idx = 0
    t0 = time.time(); fcount = 0; fps_est = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok: break
            frame_idx += 1; fcount += 1

            if using_camera:
                try:
                    for _ in range(2):
                        cap.grab()
                except Exception:
                    pass

            have_locked_inside = any((oid in captured_in) and prev_inside.get(oid, False) for oid in prev_inside.keys())
            do_heavy_this_frame = True
            if have_locked_inside and (frame_idx % INSIDE_PROCESS_EVERY_N != 0):
                do_heavy_this_frame = False
            if frame_idx % DETECT_EVERY_N != 0:
                do_heavy_this_frame = False

            lp_dets_full, lp_texts = [], []
            gate_x = int(ENTRY_LINE_X_RATIO * W)

            if do_heavy_this_frame and use_gate:
                # Detection band around gate
                band_w = max(160, int(W * GATE_DETECTION_BAND))
                d_rx = max(0, min(W-20, gate_x - band_w//2))
                d_ry = 0
                d_rw = min(band_w, W - d_rx)
                d_rh = H
                det_roi = crop_frame(frame, d_rx, d_ry, d_rw, d_rh)

                # Downscale for inference
                inf_roi = det_roi
                scale = 1.0
                if det_roi.shape[1] > ROI_INFER_MAX_W:
                    scale = ROI_INFER_MAX_W / float(det_roi.shape[1])
                    new_w = int(det_roi.shape[1] * scale)
                    new_h = int(det_roi.shape[0] * scale)
                    inf_roi = cv2.resize(det_roi, (new_w, new_h))

                all_lp_dets, all_lp_texts = lp_det.detect_frames([inf_roi])
                roi_dets = all_lp_dets[0]; lp_texts = all_lp_texts[0]

                # Map to full frame
                lp_dets_full = []
                if scale != 1.0:
                    inv_sx = 1.0/scale; inv_sy = 1.0/scale
                    for d in roi_dets:
                        d_scaled = scale_det(d, inv_sx, inv_sy)
                        lp_dets_full.append(offset_det(d_scaled, d_rx, d_ry))
                else:
                    for d in roi_dets:
                        lp_dets_full.append(offset_det(d, d_rx, d_ry))

            # Draw gate overlay
            if use_gate:
                draw_detection_band(frame, gate_x, W, H, GATE_DETECTION_BAND)
                draw_gate_overlay(frame, gate_x, W, H, gate_right)
            
            if do_heavy_this_frame:
                draw_lp_boxes(frame, lp_dets_full, color=(0,255,0))

            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Process detections (same logic as original)
            if do_heavy_this_frame:
                for det, text in zip(lp_dets_full, lp_texts):
                    key = stable_key(text, det)
                    if key is None: continue
                    oid = idbank.get(key)

                    cx, cy = center_from_det(det)
                    if use_gate:
                        is_in = (cx is not None) and ((cx >= gate_x) if gate_right else (cx <= gate_x))
                    else:
                        is_in = False

                    if PRINT_RAW_OCR_FOUND:
                        norm = normalize_ocr_text(text or "")
                        if norm and (re.search(u"[ぁ-ゖァ-ヿ]", norm) or plate_suffix(norm)):
                            print(f"[RAW] text='{norm}'  center=({cx:.0f},{cy:.0f})  {'INSIDE' if is_in else 'OUTSIDE'}")

                    if oid not in prev_inside:
                        prev_inside[oid] = is_in
                        inside_count[oid] = 1 if is_in else 0
                        outside_count[oid] = 0 if is_in else 1
                        if is_in:
                            capture_open[oid] = CAPTURE_WINDOW_FRAMES
                            best_score[oid] = 0.0; best_text[oid] = ""; best_det[oid] = det
                        continue

                    if is_in:
                        inside_count[oid] = inside_count.get(oid,0) + 1
                        outside_count[oid] = 0
                    else:
                        outside_count[oid] = outside_count.get(oid,0) + 1
                        inside_count[oid] = 0

                    if capture_open.get(oid, 0) == -1:
                        prev_inside[oid] = is_in
                        continue

                    # Open capture window on gate crossing
                    if (not prev_inside[oid]) and is_in:
                        capture_open[oid] = CAPTURE_WINDOW_FRAMES
                        best_score[oid] = 0.0; best_text[oid] = ""; best_det[oid] = det

                    # Open when approaching gate
                    if (oid not in captured_in) and (capture_open.get(oid, 0) <= 0):
                        if cx is not None and abs(cx - gate_x) <= GATE_CAPTURE_MARGIN_PX:
                            approaching_from_outside = (cx < gate_x) if gate_right else (cx > gate_x)
                            if approaching_from_outside:
                                capture_open[oid] = CAPTURE_WINDOW_FRAMES
                                best_score[oid] = 0.0; best_text[oid] = ""; best_det[oid] = det

                    # Capture best plate reading
                    if is_in and (oid not in captured_in):
                        if capture_open.get(oid, 0) > 0:
                            x1,y1,x2,y2 = map(int, xyxy_from_det(det))
                            plate_img = frame[max(0,y1):min(H,y2), max(0,x1):min(W,x2)]

                            min_side = min(y2-y1, x2-x1)
                            if min_side < 18:
                                capture_open[oid] -= 1
                                continue

                            sharp = variance_of_laplacian(plate_img)
                            if sharp < 25.0:
                                capture_open[oid] -= 1
                                continue

                            score = bbox_area(det) * max(1.0, sharp)

                            if score > best_score.get(oid, 0.0) and plate_suffix(text or ""):
                                best_score[oid] = score
                                best_text[oid] = text
                                best_det[oid] = det

                            if plate_suffix(text or "") and sharp >= MIN_SHARPNESS_LOCK:
                                capture_open[oid] = 0
                            else:
                                capture_open[oid] -= 1

                        # Log ENTRY event
                        if capture_open.get(oid, 0) == 0 and inside_count[oid] >= ENTER_STABLE_FRAMES:
                            chosen_text = best_text.get(oid,"") or text or ""
                            pk = plate_key(chosen_text)
                            region_class, suf, kana, city, engine_size = parse_plate_fields(chosen_text)

                            # Check for duplicate (same plate already inside)
                            if pk and pk in active_plate_to_slot:
                                chosen_slot = active_plate_to_slot[pk]
                                active_id_to_plate[oid] = pk
                                active_id_to_slot[oid] = chosen_slot
                                if suf: active_id_to_suffix[oid] = suf
                                captured_in.add(oid); last_event_f[oid] = frame_idx; capture_open[oid] = -1
                                cv2.putText(frame, f"IN {chosen_slot:02d}(dup)", (int(cx), int(cy)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,180,255), 2, cv2.LINE_AA)
                                slot_str = f"{chosen_slot:02d}({suf})" if suf else f"{chosen_slot:02d}"
                                log_terminal(ts, "in", slot_str, region_class, suf, kana, city, chosen_text)
                                prev_inside[oid] = is_in
                                continue

                            # Check for duplicate by suffix
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
                                cv2.putText(frame, f"IN {chosen_slot:02d}(dup)", (int(cx), int(cy)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,180,255), 2, cv2.LINE_AA)
                                slot_str = f"{chosen_slot:02d}({suf})" if suf else f"{chosen_slot:02d}"
                                log_terminal(ts, "in", slot_str, region_class, suf, kana, city, chosen_text)
                                prev_inside[oid] = is_in
                                continue

                            # Assign new slot
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

                                object_id = f"{chosen_slot:02d}"
                                write_row_flush(csv_writer, csv_file,
                                                [ts, object_id, "car", "in", city or "", engine_size or "", kana or "", suf or ""])
                                captured_in.add(oid); last_event_f[oid] = frame_idx; capture_open[oid] = -1
                                cv2.putText(frame, f"▶ IN {object_id}", (int(cx), int(cy)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv2.LINE_AA)
                                log_terminal(ts, "in", f"{object_id}({suf})" if suf else object_id,
                                             region_class, suf, kana, city, chosen_text)

                    # EXIT handling
                    if prev_inside[oid] and (not is_in) and outside_count[oid] >= EXIT_STABLE_FRAMES:
                        if frame_idx - last_event_f.get(oid, -10**9) >= MIN_EVENT_GAP_FRAMES:
                            pk_now = active_id_to_plate.get(oid)
                            slot_to_free = None
                            if pk_now and pk_now in active_plate_to_slot:
                                slot_to_free = active_plate_to_slot.pop(pk_now, None)
                            if slot_to_free is None:
                                slot_to_free = active_id_to_slot.pop(oid, None)

                            if slot_to_free is not None:
                                chosen_text = best_text.get(oid,"") or text or ""
                                region_class, suf_out, kana, city, engine_size = parse_plate_fields(chosen_text)
                                object_id = f"{slot_to_free:02d}"
                                write_row_flush(csv_writer, csv_file,
                                                [ts, object_id, "car", "out", city or "", engine_size or "", kana or "", suf_out or ""])
                                slot_pool.release(slot_to_free)
                                last_event_f[oid] = frame_idx
                                cv2.putText(frame, f"◀ OUT {object_id}", (int(cx), int(cy)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv2.LINE_AA)
                                log_terminal(ts, "out", f"{object_id}({suf_out})" if suf_out else object_id,
                                             region_class, suf_out, kana, city, chosen_text)

                                if suf_out and active_suffix_to_slot.get(suf_out) == slot_to_free:
                                    active_suffix_to_slot.pop(suf_out, None)

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
                tag = "" if do_heavy_this_frame else " (throttled)"
                cv2.putText(frame, f"FPS: {fps_est:.1f}{tag}", (10, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
                mode = "GATE" if use_gate else "ROI"
                side = "RIGHT" if gate_right else "LEFT"
                cv2.putText(frame, f"Mode: {mode} | Inside: {side} of line", (10, 52),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180,255,180), 2, cv2.LINE_AA)
                cv2.putText(frame, "[m]Toggle mode  [o]Flip side  [ESC]Quit", (10, 76),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,180,255), 1, cv2.LINE_AA)

            cv2.imshow("Gate-Based Parking Tracker", frame)
            if out is not None and WRITE_VIDEO: out.write(frame)

            # Keyboard controls
            k = cv2.waitKey(1) & 0xFF
            if k == 27: break  # ESC
            if k == ord('m'):
                use_gate = not use_gate
            if k == ord('o'):
                gate_right = not gate_right
                print(f"[INFO] Inside zone now: {'RIGHT' if gate_right else 'LEFT'} of gate")

    finally:
        try: cap.release()
        except: pass
        try:
            if WRITE_VIDEO and out is not None: out.release()
        except: pass
        cv2.destroyAllWindows()
        csv_file.close()
        print(f"\n[INFO] Session complete!")
        print(f"[INFO] CSV log saved to: {csv_path}")
        if WRITE_VIDEO: print("[INFO] Video saved to: output_videos/output_video.avi")

if __name__ == "__main__":
    main()