# -*- coding: utf-8 -*-
# per5_f05.py — ROI/Gate + light ByteTrack-style tracking for license plates
# - Tracks license plates so we draw trails and only run one-shot OCR when a track crosses the gate
# - Cuts repeated heavy OCR load while keeping your IN/OUT + CSV logic compatible

import os, cv2, csv, re, time
from datetime import datetime
from licence_plate_detection import LicencePlateDetection

# ===================== User settings =====================
USE_CAMERA               = True
USE_FILE                 = False
USE_YOUTUBE              = False

INPUT_VIDEO_PATH         = "input_videos/Trim03.mp4"
YOUTUBE_URL              = "https://www.youtube.com/watch?v=jqtsC5BYlIk"
YOUTUBE_MAX_HEIGHT       = 480

PRINT_EVENTS_TO_TERMINAL = True
PRINT_RAW_OCR_FOUND      = False
PASS_OCR_DEBUG_TO_DETECT = False
# =========================================================

SLOT_COUNT               = 10

REGION_XYWH_RATIO        = (0.32, 0.60, 0.36, 0.20)
LOCK_ROI                 = False

USE_ROI                  = False
ENTRY_LINE_X_RATIO       = 0.55
GATE_CAPTURE_MARGIN_PX   = 60
DRAW_GATE                = True
GATE_INSIDE_IS_RIGHT     = True

ENTER_STABLE_FRAMES      = 1
EXIT_STABLE_FRAMES       = 4
MIN_EVENT_GAP_FRAMES     = 10

CAPTURE_WINDOW_FRAMES    = 6
INSIDE_PROCESS_EVERY_N   = 8
DETECT_EVERY_N           = 1

PLATE_SUFFIX_RE          = re.compile(r"(\d{2,3})\s*[-−–—ーｰ~〜－]\s*(\d{2})")
MIN_SHARPNESS_LOCK       = 60.0

WRITE_VIDEO              = False
DRAW_BBOXES              = True
DRAW_TRACKS              = True
SHOW_FPS                 = True
TARGET_WIDTH             = 640
BOX_THICK                = 1
ROI_INFER_MAX_W          = 640
GATE_DET_BAND_RATIO      = 0.50

# Tracker tuning
TRACK_IOU_THRESH         = 0.3
TRACK_MAX_AGE            = 20
TRACK_MIN_HITS           = 2
TRACK_TAIL               = 24
# =========================================================

cv2.setUseOptimized(True)
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")
os.environ.setdefault("FLAGS_minloglevel", "2")

# ---------- YouTube helper ----------
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

# ---------- text / OCR helpers ----------
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
        engine_size = m.group(2)
        region_class = f"{city}{engine_size}"
    kana = None; kana_matches = re.findall(u"[ぁ-ゖァ-ヿ]", t)
    if kana_matches: kana = kana_matches[-1]
    return region_class, suf, kana, city, engine_size

# ---------- CSV ----------
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
    kana_disp = kana if kana else ""
    city_disp = city if city else ""
    rc_disp = region_class if region_class else ""
    suf_disp = suffix if suffix else ""
    print(f"[{event_ts}] {direction.upper():3s}  slot={slot_str:>6s}  region_class='{rc_disp}'  suffix='{suf_disp}'  kana='{kana_disp}'  city='{city_disp}'  raw=\"{raw_text}\"")

# ---------- geometry ----------
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

def center_from_xyxy(x1,y1,x2,y2):
    return (x1+x2)/2.0, (y1+y2)/2.0

def variance_of_laplacian(img):
    if img is None or img.size == 0: return 0.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape)==3 else img
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def crop_frame(frame, rx, ry, rw, rh):
    return frame[ry:ry+rh, rx:rx+rw]

# ---------- region helpers ----------
def clamp_region(rx, ry, rw, rh, W, H):
    """Clamp ROI rectangle to frame bounds and ensure a minimum size."""
    rx = max(0, min(int(rx), max(0, W - 10)))
    ry = max(0, min(int(ry), max(0, H - 10)))
    rw = max(20, min(int(rw), max(20, W - rx)))
    rh = max(20, min(int(rh), max(20, H - ry)))
    return rx, ry, rw, rh

def inside_region(cx, cy, rx, ry, rw, rh):
    return (cx is not None) and (cy is not None) and (rx <= cx <= rx+rw) and (ry <= cy <= ry+rh)

# ---------- Slot Pool ----------
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

# ---------- ByteTrack-lite ----------
def iou(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2-ix1), max(0.0, iy2-iy1)
    inter = iw*ih
    ua = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter/ua if ua>0 else 0.0

class BTTrack(object):
    __slots__ = ("tid","bbox","age","hits","time_since_update","history")
    def __init__(self, tid, bbox):
        self.tid = tid
        self.bbox = bbox
        self.age = 0
        self.hits = 1
        self.time_since_update = 0
        self.history = []  # list of (cx,cy)

class ByteTrackLite(object):
    def __init__(self, iou_thresh=0.3, max_age=20, min_hits=2, tail=24):
        self.iou_thr = iou_thresh
        self.max_age = max_age
        self.min_hits = min_hits
        self.tail = tail
        self._next_tid = 1
        self.tracks = []

    def update(self, dets):
        # dets: list of (x1,y1,x2,y2,conf)
        assigned = set()
        for tr in self.tracks:
            tr.age += 1
            tr.time_since_update += 1

        # greedy IoU assign
        for tr in self.tracks:
            best_iou = 0.0; best_j = -1
            for j, d in enumerate(dets):
                if j in assigned: continue
                i = iou(tr.bbox, d[:4])
                if i > best_iou:
                    best_iou = i; best_j = j
            if best_j >= 0 and best_iou >= self.iou_thr:
                x1,y1,x2,y2,conf = dets[best_j]
                tr.bbox = (x1,y1,x2,y2)
                tr.hits += 1
                tr.time_since_update = 0
                cx, cy = center_from_xyxy(x1,y1,x2,y2)
                tr.history.append((int(cx), int(cy)))
                if len(tr.history) > self.tail: tr.history.pop(0)
                assigned.add(best_j)

        # births
        for j, d in enumerate(dets):
            if j in assigned: continue
            x1,y1,x2,y2,conf = d
            t = BTTrack(self._next_tid, (x1,y1,x2,y2))
            self._next_tid += 1
            cx, cy = center_from_xyxy(x1,y1,x2,y2)
            t.history.append((int(cx), int(cy)))
            self.tracks.append(t)

        # prune
        self.tracks = [tr for tr in self.tracks if tr.time_since_update <= self.max_age]

        return [{
            "track_id": tr.tid,
            "bbox": tr.bbox,
            "confirmed": (tr.hits >= self.min_hits),
            "history": list(tr.history)
        } for tr in self.tracks]

# ---------- main ----------
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
        short = (src[:80] + '...') if isinstance(src, str) and len(src) > 80 else src
        print("[INFO] Using YouTube stream:", short)
    else:
        src = INPUT_VIDEO_PATH
        if not os.path.exists(src): raise RuntimeError(f"Input file not found: {src}")

    use_roi = USE_ROI
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

    if use_roi:
        rx = int(REGION_XYWH_RATIO[0]*W); ry = int(REGION_XYWH_RATIO[1]*H)
        rw = int(REGION_XYWH_RATIO[2]*W); rh = int(REGION_XYWH_RATIO[3]*H)
        rx, ry, rw, rh = clamp_region(rx, ry, rw, rh, W, H)
    else:
        rx, ry, rw, rh = 0, 0, W, H

    tracker = ByteTrackLite(TRACK_IOU_THRESH, TRACK_MAX_AGE, TRACK_MIN_HITS, TRACK_TAIL)
    slot_pool = SlotPool(SLOT_COUNT)

    # plate ↔ slot state
    plate_pref_slot = {}       # pk -> sticky preferred slot
    active_plate_to_slot = {}  # pk -> current slot
    active_suffix_to_slot = {} # 'nn-nn' -> current slot

    # per-track state
    prev_inside     = {}       # tid -> bool
    inside_count    = {}
    outside_count   = {}
    last_event_f    = {}
    captured_in     = set()
    capture_open    = {}       # tid -> frames left (0 close, -1 locked)
    best_score      = {}       # tid -> area*sharpness
    best_text       = {}       # tid -> best OCR
    best_det        = {}       # tid -> best bbox
    track_slot      = {}       # tid -> assigned slot number (for OUT)

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

            rx, ry, rw, rh = clamp_region(rx, ry, rw, rh, W, H)
            gate_x = int(ENTRY_LINE_X_RATIO * W)

            # throttle heavy work
            have_locked_inside = any((tid in captured_in) and prev_inside.get(tid, False) for tid in list(prev_inside.keys()))
            do_heavy_this_frame = True
            if have_locked_inside and (frame_idx % INSIDE_PROCESS_EVERY_N != 0):
                do_heavy_this_frame = False
            if frame_idx % DETECT_EVERY_N != 0:
                do_heavy_this_frame = False

            # detection region
            lp_dets_full, lp_texts = [], []
            if do_heavy_this_frame:
                if use_roi:
                    d_rx, d_ry, d_rw, d_rh = rx, ry, rw, rh
                else:
                    band_w = max(160, int(W * GATE_DET_BAND_RATIO))
                    d_rx = max(0, min(W-20, gate_x - band_w//2))
                    d_ry = 0
                    d_rw = min(band_w, W - d_rx)
                    d_rh = H
                det_roi = crop_frame(frame, d_rx, d_ry, d_rw, d_rh)

                inf_roi = det_roi
                scale = 1.0
                if det_roi.shape[1] > ROI_INFER_MAX_W:
                    scale = ROI_INFER_MAX_W / float(det_roi.shape[1])
                    new_w = int(det_roi.shape[1] * scale)
                    new_h = int(det_roi.shape[0] * scale)
                    inf_roi = cv2.resize(det_roi, (new_w, new_h))

                all_lp_dets, all_lp_texts = lp_det.detect_frames([inf_roi])
                roi_dets = all_lp_dets[0]; lp_texts = all_lp_texts[0]

                # map back
                lp_dets_full = []
                if scale != 1.0:
                    inv_sx = 1.0/scale; inv_sy = 1.0/scale
                    for d in roi_dets:
                        d_scaled = scale_det(d, inv_sx, inv_sy)
                        lp_dets_full.append(offset_det(d_scaled, d_rx, d_ry))
                else:
                    for d in roi_dets:
                        lp_dets_full.append(offset_det(d, d_rx, d_ry))

            # --- update tracker ---
            dets_for_tracker = []
            for det in lp_dets_full:
                xy = xyxy_from_det(det)
                if not xy: continue
                x1,y1,x2,y2 = xy
                conf = 1.0
                if isinstance(det, dict):
                    conf = float(det.get("conf", 1.0))
                dets_for_tracker.append((x1,y1,x2,y2,conf))
            tracks = tracker.update(dets_for_tracker)

            # drawing: boxes + gate/roi
            if DRAW_BBOXES and do_heavy_this_frame:
                for det in lp_dets_full:
                    xy = xyxy_from_det(det)
                    if not xy: continue
                    x1,y1,x2,y2 = map(int, xy)
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), BOX_THICK)
            if DRAW_BBOXES:
                if use_roi:
                    cv2.rectangle(frame, (rx, ry), (rx+rw, ry+rh), (0,255,255), BOX_THICK)
                    cv2.putText(frame, "ROI (locked)" if LOCK_ROI else "ROI: WASD move, [ ] resize, r reset",
                                (rx+6, max(18, ry-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255), 2, cv2.LINE_AA)
                elif DRAW_GATE:
                    gx = gate_x
                    cv2.line(frame, (gx, 0), (gx, H), (200, 200, 255), 2)
                    cv2.putText(frame, "gate", (gx+6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,255), 2, cv2.LINE_AA)

            # trails + track ids
            if DRAW_TRACKS:
                for tr in tracks:
                    tid = tr["track_id"]
                    hist = tr["history"]
                    if len(hist) >= 2:
                        for i in range(1, len(hist)):
                            cv2.line(frame, hist[i-1], hist[i], (255, 200, 0), 2)
                    if len(hist) > 0:
                        lx, ly = hist[-1]
                        cv2.putText(frame, f"ID {tid}", (lx+6, ly-6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,200,0), 2, cv2.LINE_AA)

            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # pair current dets with texts
            det_boxes = []
            for det, txt in zip(lp_dets_full, lp_texts):
                xy = xyxy_from_det(det)
                if xy:
                    det_boxes.append((xy, txt, det))

            def nearest_text_for_bbox(bbox):
                if not det_boxes: return "", None
                x1,y1,x2,y2 = bbox
                cx = (x1+x2)/2.0; cy = (y1+y2)/2.0
                best_d = 1e18; best = ("", None)
                for (bx1,by1,bx2,by2), t, det in det_boxes:
                    pcx = (bx1+bx2)/2.0; pcy = (by1+by2)/2.0
                    d = (pcx-cx)**2 + (pcy-cy)**2
                    if d < best_d:
                        best_d = d; best = (t, det)
                return best

            # ==== per-track gate logic ====
            for tr in tracks:
                tid = tr["track_id"]
                x1,y1,x2,y2 = tr["bbox"]
                cx = (x1+x2)/2.0; cy = (y1+y2)/2.0

                # which side is "inside"
                if use_roi:
                    is_in = inside_region(cx, cy, rx, ry, rw, rh)
                else:
                    is_in = (cx >= gate_x) if gate_right else (cx <= gate_x)

                # init streaks
                if tid not in prev_inside:
                    prev_inside[tid] = is_in
                    inside_count[tid]  = 1 if is_in else 0
                    outside_count[tid] = 0 if is_in else 1
                    if is_in:
                        capture_open[tid] = CAPTURE_WINDOW_FRAMES
                        best_score[tid] = 0.0; best_text[tid] = ""; best_det[tid] = (x1,y1,x2,y2)
                    continue

                # streak updates
                if is_in:
                    inside_count[tid]  = inside_count.get(tid,0) + 1
                    outside_count[tid] = 0
                else:
                    outside_count[tid] = outside_count.get(tid,0) + 1
                    inside_count[tid]  = 0

                # open capture when crossing into inside
                crossed_in = (not prev_inside[tid]) and is_in
                if (not use_roi) and crossed_in:
                    capture_open[tid] = CAPTURE_WINDOW_FRAMES
                    best_score[tid] = 0.0; best_text[tid] = ""; best_det[tid] = (x1,y1,x2,y2)

                # also open if approaching gate from outside side
                if (not use_roi) and (tid not in captured_in) and (capture_open.get(tid, 0) <= 0):
                    if abs(cx - gate_x) <= GATE_CAPTURE_MARGIN_PX:
                        approaching_from_outside = (cx < gate_x) if gate_right else (cx > gate_x)
                        if approaching_from_outside:
                            capture_open[tid] = CAPTURE_WINDOW_FRAMES
                            best_score[tid] = 0.0; best_text[tid] = ""; best_det[tid] = (x1,y1,x2,y2)

                # one-shot capture maintenance
                if is_in and (tid not in captured_in):
                    if capture_open.get(tid, 0) > 0:
                        text, _ = nearest_text_for_bbox((x1,y1,x2,y2))
                        if PRINT_RAW_OCR_FOUND:
                            norm = normalize_ocr_text(text or "")
                            if norm and (re.search(u"[ぁ-ゖァ-ヿ]", norm) or plate_suffix(norm)):
                                print(f"[RAW] text='{norm}'  center=({cx:.1f},{cy:.1f})")

                        X1,Y1,X2,Y2 = map(int, (x1,y1,x2,y2))
                        X1 = max(0, X1); Y1 = max(0, Y1); X2 = min(W-1, X2); Y2 = min(H-1, Y2)
                        plate_img = frame[Y1:Y2, X1:X2]
                        min_side = min(max(0, Y2-Y1), max(0, X2-X1))
                        if min_side < 18:
                            capture_open[tid] -= 1
                        else:
                            sharp = variance_of_laplacian(plate_img)
                            if sharp < 25.0:
                                capture_open[tid] -= 1
                            else:
                                area = max(0.0, (x2-x1)) * max(0.0, (y2-y1))
                                score = area * max(1.0, sharp)
                                if text and plate_suffix(text or "") and score > best_score.get(tid, 0.0):
                                    best_score[tid] = score
                                    best_text[tid]  = text
                                    best_det[tid]   = (x1,y1,x2,y2)
                                if plate_suffix(text or "") and sharp >= MIN_SHARPNESS_LOCK:
                                    capture_open[tid] = 0
                                else:
                                    capture_open[tid] -= 1

                    # when window closes -> log IN once
                    if capture_open.get(tid, 0) == 0 and inside_count[tid] >= ENTER_STABLE_FRAMES:
                        chosen_text = best_text.get(tid,"") or ""
                        if not chosen_text:
                            t_now, _ = nearest_text_for_bbox((x1,y1,x2,y2))
                            chosen_text = t_now or ""
                        pk = plate_key(chosen_text)
                        region_class, suf, kana, city, engine_size = parse_plate_fields(chosen_text)

                        # duplicate attach by plate key
                        if pk and pk in active_plate_to_slot:
                            chosen_slot = active_plate_to_slot[pk]
                            captured_in.add(tid); last_event_f[tid] = frame_idx; capture_open[tid] = -1
                            track_slot[tid] = chosen_slot
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
                            captured_in.add(tid); last_event_f[tid] = frame_idx; capture_open[tid] = -1
                            track_slot[tid] = chosen_slot
                            if DRAW_BBOXES:
                                cv2.putText(frame, f"IN {chosen_slot:02d}(dup)", (int(cx), int(cy)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,180,255), 2, cv2.LINE_AA)
                            slot_str = f"{chosen_slot:02d}({suf})" if suf else f"{chosen_slot:02d}"
                            log_terminal(ts, "in", slot_str, region_class, suf, kana, city, chosen_text)
                            prev_inside[tid] = is_in
                            continue

                        # new slot
                        if pk and pk in plate_pref_slot:
                            got = slot_pool.try_acquire_specific(plate_pref_slot[pk])
                            if got is not None: chosen_slot = got
                        if chosen_slot is None:
                            chosen_slot = slot_pool.acquire_lowest()

                        if chosen_slot is not None:
                            if pk:
                                active_plate_to_slot[pk] = chosen_slot
                                plate_pref_slot[pk] = chosen_slot
                            if suf:
                                active_suffix_to_slot[suf] = chosen_slot

                            object_id = f"{chosen_slot:02d}"
                            write_row_flush(csv_writer, csv_file,
                                            [ts, object_id, "car", "in", city or "", engine_size or "", kana or "", suf or ""])
                            captured_in.add(tid); last_event_f[tid] = frame_idx; capture_open[tid] = -1
                            track_slot[tid] = chosen_slot
                            if DRAW_BBOXES:
                                cv2.putText(frame, f"IN {object_id}", (int(cx), int(cy)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv2.LINE_AA)
                            log_terminal(ts, "in", f"{object_id}({suf})" if suf else object_id,
                                         region_class, suf, kana, city, chosen_text)

                # exit handling (inside -> outside with stability)
                crossed_out = prev_inside[tid] and (not is_in) and outside_count[tid] >= EXIT_STABLE_FRAMES
                if crossed_out and (frame_idx - last_event_f.get(tid, -10**9) >= MIN_EVENT_GAP_FRAMES):
                    chosen_text = best_text.get(tid,"") or ""
                    if not chosen_text:
                        t_now, _ = nearest_text_for_bbox((x1,y1,x2,y2))
                        chosen_text = t_now or ""
                    region_class, suf_out, kana, city, engine_size = parse_plate_fields(chosen_text)

                    slot_to_free = track_slot.get(tid)
                    if slot_to_free is None:
                        if suf_out and suf_out in active_suffix_to_slot:
                            slot_to_free = active_suffix_to_slot.get(suf_out)
                        if slot_to_free is None:
                            pk_now = plate_key(chosen_text)
                            if pk_now and pk_now in active_plate_to_slot:
                                slot_to_free = active_plate_to_slot.get(pk_now)

                    if slot_to_free is not None:
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
                        # also clear active_plate_to_slot if it points to this slot
                        pk_now = plate_key(chosen_text)
                        if pk_now and active_plate_to_slot.get(pk_now) == slot_to_free:
                            active_plate_to_slot.pop(pk_now, None)

                    # clear per-track buffers
                    for dct in (capture_open, best_score, best_text, best_det, track_slot):
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

            # Display / write
            cv2.imshow("Parking (ByteTrack-lite + ROI/Gate one-shot)", frame)
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
