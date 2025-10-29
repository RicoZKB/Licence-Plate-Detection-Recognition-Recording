# -*- coding: utf-8 -*-
# per5_supervision.py — LicencePlateDetection + Supervision(ByteTrack) integration
# - Source selector: Camera / File / YouTube
# - Plate detect (ROI or Gate band) -> ByteTrack IDs -> gate in/out -> one-shot OCR capture
# - Stable CSV (8 columns) + terminal IN/OUT logs
# - Overlays: gate line / ROI / plate boxes / FPS
#
# pip install supervision ultralytics opencv-python yt-dlp
# (Keeps your own LicencePlateDetection for detection+OCR)

import os, cv2, csv, re, time
from datetime import datetime
import numpy as np

import supervision as sv  # ByteTrack, utils

from licence_plate_detection import LicencePlateDetection

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

# Terminal logging toggles
PRINT_EVENTS_TO_TERMINAL = True
PRINT_RAW_OCR_FOUND      = False
PASS_OCR_DEBUG_TO_DETECT = False
# =========================================================

SLOT_COUNT               = 10
START_ID_AT              = 100  # kept for back-compat (not used; we prefer tracker_id)

# Region as ratios of frame (x, y, w, h)
REGION_XYWH_RATIO        = (0.32, 0.60, 0.36, 0.20)

# Lock ROI from code (disable WASD/[ ])
LOCK_ROI                 = False

# Use a virtual gate line instead of ROI region (detect anywhere)
USE_ROI                  = False
ENTRY_LINE_X_RATIO       = 0.55   # vertical line as fraction of width
GATE_CAPTURE_MARGIN_PX   = 60
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
# =========================================================

cv2.setUseOptimized(True)
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")
os.environ.setdefault("FLAGS_minloglevel", "2")

def _resolve_youtube_opencv_url(url: str, max_height: int = 480) -> str:
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
    if not PRINT_EVENTS_TO_TERMINAL: return
    print(f"[{event_ts}] {direction.upper():3s}  slot={slot_str:>6s}  region_class='{region_class or ''}'  suffix='{suffix or ''}'  kana='{kana or ''}'  city='{city or ''}'  raw=\"{raw_text}\"")

# ---------- geometry helpers ----------
def variance_of_laplacian(img):
    if img is None or img.size == 0: return 0.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape)==3 else img
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def clamp_region(rx, ry, rw, rh, W, H):
    rx = max(0, min(rx, W-10)); ry = max(0, min(ry, H-10))
    rw = max(20, min(rw, W-rx)); rh = max(20, min(rh, H-ry))
    return rx, ry, rw, rh

def crop_frame(frame, rx, ry, rw, rh):
    return frame[ry:ry+rh, rx:rx+rw]

def draw_region(frame, rx, ry, rw, rh):
    cv2.rectangle(frame, (rx, ry), (rx+rw, ry+rh), (0,255,255), BOX_THICK)
    cv2.putText(frame, "ROI (locked)" if LOCK_ROI else "ROI: WASD move, [ ] resize, r reset",
                (rx+6, max(18, ry-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255), 2, cv2.LINE_AA)

def draw_lp_boxes(frame, boxes, color=(0,255,0)):
    for (x1,y1,x2,y2) in boxes:
        cv2.rectangle(frame, (int(x1),int(y1)), (int(x2),int(y2)), color, BOX_THICK)

# ---------- slot management ----------
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

# ===================== main =====================
def main():
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
        print("[INFO] Using YouTube stream.")
    else:
        src = INPUT_VIDEO_PATH
        if not os.path.exists(src): raise RuntimeError(f"Input file not found: {src}")

    # toggles (keyboard)
    use_roi = USE_ROI
    gate_right = GATE_INSIDE_IS_RIGHT

    cap = cv2.VideoCapture(src)
    if not cap.isOpened(): raise RuntimeError(f"Could not open: {src}")

    using_camera = (source_name == "Camera")
    if using_camera:
        try: cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except: pass

    try: cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)
    except: pass

    # Plate detector (keeps your OCR)
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

    # ROI init
    if use_roi:
        rx = int(REGION_XYWH_RATIO[0]*W); ry = int(REGION_XYWH_RATIO[1]*H)
        rw = int(REGION_XYWH_RATIO[2]*W); rh = int(REGION_XYWH_RATIO[3]*H)
        rx, ry, rw, rh = clamp_region(rx, ry, rw, rh, W, H)
    else:
        rx, ry, rw, rh = 0, 0, W, H

    # ===== Supervision: ByteTrack (tracking IDs) =====
    tracker = sv.ByteTrack()

    # Per-track state
    prev_inside   = {}   # track_id -> bool
    inside_count  = {}
    outside_count = {}
    last_event_f  = {}

    # slot assignment & duplications
    slot_pool = SlotPool(SLOT_COUNT)
    plate_pref_slot = {}               # pk -> preferred slot
    active_plate_to_slot = {}          # pk -> slot
    active_suffix_to_slot = {}         # "70-50" -> slot
    active_id_to_slot = {}             # track_id -> slot
    active_id_to_plate = {}            # track_id -> pk
    active_id_to_suffix = {}           # track_id -> suffix

    # one-shot window
    capture_open = {}   # track_id -> frames left (0 close, -1 locked)
    best_score  = {}    # track_id -> area*sharpness
    best_text   = {}    # track_id -> best OCR
    best_box    = {}    # track_id -> (x1,y1,x2,y2)
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
                    for _ in range(2): cap.grab()
                except: pass

            rx, ry, rw, rh = clamp_region(rx, ry, rw, rh, W, H)

            have_locked_inside = any((tid in captured_in) and prev_inside.get(tid, False) for tid in prev_inside.keys())
            do_heavy_this_frame = True
            if have_locked_inside and (frame_idx % INSIDE_PROCESS_EVERY_N != 0):
                do_heavy_this_frame = False
            if frame_idx % DETECT_EVERY_N != 0:
                do_heavy_this_frame = False

            boxes_full = []
            texts_full = []

            # ---------- Detect plates in ROI or gate band ----------
            if do_heavy_this_frame:
                gate_x = int(ENTRY_LINE_X_RATIO * W)
                if use_roi:
                    d_rx, d_ry, d_rw, d_rh = rx, ry, rw, rh
                else:
                    band_w = max(160, int(W * GATE_DET_BAND_RATIO))
                    d_rx = max(0, min(W-20, gate_x - band_w//2))
                    d_ry = 0
                    d_rw = min(band_w, W - d_rx)
                    d_rh = H
                det_roi = crop_frame(frame, d_rx, d_ry, d_rw, d_rh)

                # speed resize
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
                if scale != 1.0:
                    inv = 1.0/scale
                    for d, t in zip(roi_dets, lp_texts):
                        x1,y1,x2,y2 = d[:4]
                        boxes_full.append([x1*inv + d_rx, y1*inv + d_ry, x2*inv + d_rx, y2*inv + d_ry])
                        texts_full.append(t)
                else:
                    for d, t in zip(roi_dets, lp_texts):
                        x1,y1,x2,y2 = d[:4]
                        boxes_full.append([x1 + d_rx, y1 + d_ry, x2 + d_rx, y2 + d_ry])
                        texts_full.append(t)

            # ---------- Convert to Supervision Detections & track ----------
            n = len(boxes_full)
            xyxy_arr = np.asarray(boxes_full, dtype=float).reshape(-1, 4) if n > 0 else np.empty((0, 4), dtype=float)
            conf_arr = np.full((n,), 0.99, dtype=float)  # or use your detector confidences if you have them
            cls_arr  = np.zeros((n,), dtype=int)

            detections = sv.Detections(
                xyxy=xyxy_arr,
                confidence=conf_arr,
                class_id=cls_arr
            )

            detections = tracker.update_with_detections(detections)

            # ---------- Overlays ----------
            if DRAW_BBOXES and do_heavy_this_frame:
                draw_lp_boxes(frame, boxes_full, color=(0,255,0))
            if DRAW_BBOXES:
                if use_roi:
                    draw_region(frame, rx, ry, rw, rh)
                elif DRAW_GATE:
                    gx = int(ENTRY_LINE_X_RATIO * W)
                    cv2.line(frame, (gx, 0), (gx, H), (200, 200, 255), 2)
                    cv2.putText(frame, "gate", (gx+6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,255), 2, cv2.LINE_AA)

            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            gate_x = int(ENTRY_LINE_X_RATIO * W)

            # ---------- Per track IN/OUT + one-shot capture ----------
            if do_heavy_this_frame and detections.xyxy.shape[0] > 0:
                # align texts_full by nearest (IoU) if order differs
                # quick greedy match: detections.xyxy vs boxes_full
                det_boxes = detections.xyxy
                text_map = {}
                for i in range(len(det_boxes)):
                    # pick closest box index
                    db = det_boxes[i]
                    best_j, best_iou = -1, 0.0
                    for j, fb in enumerate(boxes_full):
                        # IoU
                        xA = max(db[0], fb[0]); yA = max(db[1], fb[1])
                        xB = min(db[2], fb[2]); yB = min(db[3], fb[3])
                        inter = max(0, xB-xA) * max(0, yB-yA)
                        area1 = max(0, db[2]-db[0]) * max(0, db[3]-db[1])
                        area2 = max(0, fb[2]-fb[0]) * max(0, fb[3]-fb[1])
                        union = area1 + area2 - inter + 1e-6
                        iou = inter/union
                        if iou > best_iou:
                            best_iou, best_j = iou, j
                    if best_j >= 0:
                        text_map[i] = texts_full[best_j]
                    else:
                        text_map[i] = ""

                for i, tid in enumerate(detections.tracker_id or []):
                    if tid is None: 
                        continue
                    x1,y1,x2,y2 = detections.xyxy[i]
                    cx = 0.5*(x1+x2); cy = 0.5*(y1+y2)
                    text = text_map.get(i, "")

                    # decide inside/outside relative to vertical gate
                    is_in = (cx >= gate_x) if gate_right else (cx <= gate_x)

                    if PRINT_RAW_OCR_FOUND:
                        norm = normalize_ocr_text(text or "")
                        if norm and (re.search(u"[ぁ-ゖァ-ヿ]", norm) or plate_suffix(norm)):
                            print(f"[RAW] id={tid} text='{norm}' center=({int(cx)},{int(cy)})")

                    if tid not in prev_inside:
                        prev_inside[tid]  = is_in
                        inside_count[tid] = 1 if is_in else 0
                        outside_count[tid]= 0 if is_in else 1
                        if is_in:
                            capture_open[tid] = CAPTURE_WINDOW_FRAMES
                            best_score[tid] = 0.0; best_text[tid] = ""; best_box[tid] = (x1,y1,x2,y2)
                        continue

                    # update streaks
                    if is_in:
                        inside_count[tid]  = inside_count.get(tid,0) + 1
                        outside_count[tid] = 0
                    else:
                        outside_count[tid] = outside_count.get(tid,0) + 1
                        inside_count[tid]  = 0

                    if capture_open.get(tid, 0) == -1:
                        prev_inside[tid] = is_in
                        continue

                    # opening capture window when crossing towards inside
                    if (not prev_inside[tid]) and is_in:
                        capture_open[tid] = CAPTURE_WINDOW_FRAMES
                        best_score[tid] = 0.0; best_text[tid] = ""; best_box[tid] = (x1,y1,x2,y2)

                    # approaching margin (pre-open)
                    if (tid not in captured_in) and (capture_open.get(tid, 0) <= 0):
                        if abs(cx - gate_x) <= GATE_CAPTURE_MARGIN_PX:
                            approaching_from_outside = (cx < gate_x) if gate_right else (cx > gate_x)
                            if approaching_from_outside:
                                capture_open[tid] = CAPTURE_WINDOW_FRAMES
                                best_score[tid] = 0.0; best_text[tid] = ""; best_box[tid] = (x1,y1,x2,y2)

                    # one-shot window maintenance
                    if is_in and (tid not in captured_in):
                        if capture_open.get(tid, 0) > 0:
                            x1i,y1i,x2i,y2i = map(int, (x1,y1,x2,y2))
                            x1i=max(0,x1i); y1i=max(0,y1i); x2i=min(W-1,x2i); y2i=min(H-1,y2i)
                            plate_img = frame[y1i:y2i, x1i:x2i]
                            min_side = min(y2i-y1i, x2i-x1i)
                            if min_side < 18:
                                capture_open[tid] -= 1
                                continue
                            sharp = variance_of_laplacian(plate_img)
                            if sharp < 25.0:
                                capture_open[tid] -= 1
                                continue

                            area = max(0,(x2i-x1i)) * max(0,(y2i-y1i))
                            score = area * max(1.0, sharp)

                            if score > best_score.get(tid, 0.0) and plate_suffix(text or ""):
                                best_score[tid] = score
                                best_text[tid]  = text
                                best_box[tid]   = (x1,y1,x2,y2)

                            if plate_suffix(text or "") and sharp >= MIN_SHARPNESS_LOCK:
                                capture_open[tid] = 0
                            else:
                                capture_open[tid] -= 1

                        # IN event when window closes and stable
                        if capture_open.get(tid, 0) == 0 and inside_count[tid] >= ENTER_STABLE_FRAMES:
                            chosen_text = best_text.get(tid,"") or text or ""
                            pk = plate_key(chosen_text)
                            region_class, suf, kana, city, engine_size = parse_plate_fields(chosen_text)

                            # duplicate attach: same plate already has a slot
                            if pk and pk in active_plate_to_slot:
                                chosen_slot = active_plate_to_slot[pk]
                                active_id_to_plate[tid] = pk
                                active_id_to_slot[tid]  = chosen_slot
                                if suf: active_id_to_suffix[tid] = suf
                                captured_in.add(tid); last_event_f[tid] = frame_idx; capture_open[tid] = -1
                                if DRAW_BBOXES:
                                    cv2.putText(frame, f"IN {chosen_slot:02d}(dup)", (int(cx), int(cy)),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,180,255), 2, cv2.LINE_AA)
                                slot_str = f"{chosen_slot:02d}({suf})" if suf else f"{chosen_slot:02d}"
                                log_terminal(ts, "in", slot_str, region_class, suf, kana, city, chosen_text)
                                prev_inside[tid] = is_in
                                continue

                            # duplicate attach by suffix
                            chosen_slot = None
                            if suf and suf in active_suffix_to_slot:
                                chosen_slot = active_suffix_to_slot[suf]
                                if pk:
                                    active_plate_to_slot[pk] = chosen_slot
                                    plate_pref_slot[pk] = chosen_slot
                                    active_id_to_plate[tid] = pk
                                active_id_to_slot[tid] = chosen_slot
                                active_id_to_suffix[tid] = suf
                                captured_in.add(tid); last_event_f[tid] = frame_idx; capture_open[tid] = -1
                                if DRAW_BBOXES:
                                    cv2.putText(frame, f"IN {chosen_slot:02d}(dup)", (int(cx), int(cy)),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,180,255), 2, cv2.LINE_AA)
                                slot_str = f"{chosen_slot:02d}({suf})" if suf else f"{chosen_slot:02d}"
                                log_terminal(ts, "in", slot_str, region_class, suf, kana, city, chosen_text)
                                prev_inside[tid] = is_in
                                continue

                            # new slot assignment
                            if pk and pk in plate_pref_slot:
                                got = slot_pool.try_acquire_specific(plate_pref_slot[pk])
                                if got is not None: chosen_slot = got
                            if chosen_slot is None:
                                chosen_slot = slot_pool.acquire_lowest()

                            if chosen_slot is not None:
                                if pk:
                                    active_plate_to_slot[pk] = chosen_slot
                                    plate_pref_slot[pk] = chosen_slot
                                    active_id_to_plate[tid] = pk
                                else:
                                    active_id_to_plate[tid] = None
                                active_id_to_slot[tid] = chosen_slot
                                if suf:
                                    active_suffix_to_slot[suf] = chosen_slot
                                    active_id_to_suffix[tid] = suf

                                object_id = f"{chosen_slot:02d}"  # <-- slot only (as you wanted)
                                write_row_flush(csv_writer, csv_file,
                                                [ts, object_id, "car", "in", city or "", engine_size or "", kana or "", suf or ""])
                                captured_in.add(tid); last_event_f[tid] = frame_idx; capture_open[tid] = -1
                                if DRAW_BBOXES:
                                    cv2.putText(frame, f"IN {object_id}", (int(cx), int(cy)),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv2.LINE_AA)
                                log_terminal(ts, "in", f"{object_id}({suf})" if suf else object_id,
                                             region_class, suf, kana, city, chosen_text)

                    # OUT event
                    if prev_inside[tid] and (not is_in) and outside_count[tid] >= EXIT_STABLE_FRAMES:
                        if frame_idx - last_event_f.get(tid, -10**9) >= MIN_EVENT_GAP_FRAMES:
                            pk_now = active_id_to_plate.get(tid)
                            slot_to_free = None
                            if pk_now and pk_now in active_plate_to_slot:
                                slot_to_free = active_plate_to_slot.pop(pk_now, None)
                            if slot_to_free is None:
                                slot_to_free = active_id_to_slot.pop(tid, None)

                            if slot_to_free is not None:
                                chosen_text = best_text.get(tid,"") or text or ""
                                region_class, suf_out, kana, city, engine_size = parse_plate_fields(chosen_text)
                                object_id = f"{slot_to_free:02d}"
                                write_row_flush(csv_writer, csv_file,
                                                [ts, object_id, "car", "out", city or "", engine_size or "", kana or "", suf_out or ""])
                                slot_pool.release(slot_to_free)
                                last_event_f[tid] = frame_idx
                                if DRAW_BBOXES:
                                    cv2.putText(frame, f"OUT {object_id}", (int(cx), int(cy)),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv2.LINE_AA)
                                log_terminal(ts, "out", f"{object_id}({suf_out})" if suf_out else object_id,
                                             region_class, suf_out, kana, city, chosen_text)

                                if suf_out and active_suffix_to_slot.get(suf_out) == slot_to_free:
                                    active_suffix_to_slot.pop(suf_out, None)

                            # clear per-id caches
                            active_id_to_plate.pop(tid, None)
                            active_id_to_suffix.pop(tid, None)
                            for dct in (capture_open, best_score, best_text, best_box):
                                dct.pop(tid, None)
                            if tid in captured_in: captured_in.remove(tid)

                    prev_inside[tid] = is_in

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

            cv2.imshow("Parking (per5_supervision)", frame)
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

if __name__ == "__main__":
    main()
