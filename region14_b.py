# -*- coding: utf-8 -*-
# region14.py — ROI one-shot plate capture with fast path + robust parsing + dedupe
# CSV header: timestamp,object_id,vehicle_type,direction,city,engine_size,kana

import os, cv2, csv, re, time
from datetime import datetime
from detections import CarDetection, LicencePlateDetection

# ===================== User settings =====================
USE_CAMERA               = False
INPUT_VIDEO_PATH         = "input_videos/Trim03.mp4"

SLOT_COUNT               = 10
START_ID_AT              = 100

# ROI as ratios (x, y, w, h)
REGION_XYWH_RATIO        = (0.32, 0.60, 0.36, 0.20)
LOCK_ROI                 = False

# Stability & gaps
ENTER_STABLE_FRAMES      = 1
EXIT_STABLE_FRAMES       = 4
MIN_EVENT_GAP_FRAMES     = 10

# One-shot window + grace window
CAPTURE_WINDOW_FRAMES    = 6
FILL_GRACE_FRAMES        = 6   # keep trying OCR briefly after capture window

# Heavy detection cadence and idle throttling
DETECT_EVERY_N           = 2   # run full plate detector every N frames
INSIDE_PROCESS_EVERY_N   = 8   # when cars idle inside and no capture pending

# One-shot accept rules
MIN_SHARPNESS_LOCK       = 60.0

# Gating: open capture only at a good moment
ENTER_DELAY_FRAMES       = 3
REQUIRE_FULLY_INSIDE     = True
FULLY_INSIDE_MARGIN_FRAC = 0.05
CENTER_BAND_FRAC         = 0.40
MIN_PLATE_AREA_FRAC      = 0.015

# Perf/UX
USE_CAR_DETECTOR         = False
WRITE_VIDEO              = False
DRAW_BBOXES              = True
SHOW_FPS                 = True
TARGET_WIDTH             = 640
BOX_THICK                = 1
OVERLAY_PERSIST_FRAMES   = 12
AUTO_ROTATE_CSV_HEADER_MISMATCH = True
# =========================================================

# OpenCV housekeeping
cv2.setUseOptimized(True)
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")
os.environ.setdefault("FLAGS_minloglevel", "2")

# ---- tolerant parsers -------------------------------------------------------
_DASHES = "-–—ー‐−~"
_PLATE_SUFFIX_FLEX = re.compile(rf"(\d{{2,3}})\s*[-{_DASHES}]\s*(\d{{2}})")

# City + class number (2–4 digits) + optional kana (single char)
CITY_CLASS_KANA_RE = re.compile(
    r"([ぁ-ゖァ-ヿ一-龯A-Za-z]+)\s*([0-9]{2,4})\s*([ぁ-ゖァ-ヿA-Za-z])?",
    re.UNICODE
)

def _normalize_text(t: str) -> str:
    if not t: return ""
    trans = str.maketrans({c: "-" for c in _DASHES})
    t = t.translate(trans).replace("—", "-").replace("–", "-")
    return " ".join(t.split())

def _normalize_for_digits(t: str) -> str:
    if not t: return ""
    repl = str.maketrans({
        'O':'0','o':'0','D':'0',
        'I':'1','l':'1','L':'1','i':'1',
        'S':'5','s':'5','B':'8',
        'Z':'2','z':'2','q':'9','Q':'9',
    })
    return t.translate(repl)

def parse_city_class_kana(text):
    """Return (city, class_number, kana) e.g., ('神戸', '310', 'ふ')."""
    t = _normalize_text(text or "")
    if not t: return None, None, None
    # Prefer JP side if OCR merged "JP | EN"
    if " | " in t:
        t = t.split(" | ", 1)[0]
    m = CITY_CLASS_KANA_RE.search(t)
    if not m: return None, None, None
    city = m.group(1)
    klass = m.group(2)
    kana = m.group(3) if len(m.groups()) >= 3 else None
    return city, klass, kana

def plate_suffix(text):
    t = _normalize_for_digits(_normalize_text(text or ""))
    m = _PLATE_SUFFIX_FLEX.search(t)
    return f"{m.group(1)}-{m.group(2)}" if m else None

def plate_key(text):
    """Stable identity across frames: city? + numeric suffix."""
    suf = plate_suffix(text or "");  city, _, _ = parse_city_class_kana(text or "")
    if not suf: return None
    return f"{city if city else '?'}|{suf}"

# ---------- CSV I/O ----------
def open_daily_csv():
    date_str = datetime.now().strftime("%Y%m%d")
    os.makedirs("logs", exist_ok=True)
    path = os.path.join("logs", f"parking_log_{date_str}.csv")
    new = not os.path.exists(path)

    if not new:
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as rf:
                first = (rf.readline() or "").strip()
            expected = "timestamp,object_id,vehicle_type,direction,city,engine_size,kana"
            def normalize(h): return ",".join(x.strip() for x in h.split(",")) if h else ""
            if normalize(first) != expected and AUTO_ROTATE_CSV_HEADER_MISMATCH:
                base, ext = os.path.splitext(path)
                bak = f"{base}_old_{datetime.now().strftime('%H%M%S')}{ext}"
                os.replace(path, bak)
                print(f"[INFO] Rotated CSV with mismatched header -> {bak}")
                new = True
        except Exception as e:
            print("[WARN] Could not inspect/rotate CSV header:", e)

    f = open(path, "a", newline="", encoding="utf-8", buffering=1)  # line-buffered
    w = csv.writer(f)
    if new:
        w.writerow(["timestamp","object_id","vehicle_type","direction","city","engine_size","kana"])
        f.flush()
    return f, w, path

def write_row_flush(w, f, row):
    w.writerow(row); f.flush()
    try: os.fsync(f.fileno())
    except Exception: pass

# ---------- geometry helpers ----------
def xyxy_from_det(det):
    if isinstance(det, (list, tuple)) and len(det) >= 4:
        return float(det[0]), float(det[1]), float(det[2]), float(det[3])
    if isinstance(det, dict):
        box = det.get("bbox") or det.get("xyxy")
        if box and len(box) >= 4:
            return float(box[0]), float(box[1]), float(box[2]), float(box[3])
    return None

def offset_det(det, ox, oy):
    if isinstance(det, (list, tuple)) and len(det) >= 4:
        x1,y1,x2,y2 = det[:4]
        return [x1+ox, y1+oy, x2+ox, y2+oy]
    if isinstance(det, dict):
        box = det.get("bbox") or det.get("xyxy")
        if box and len(box) >= 4:
            box = [box[0]+ox, box[1]+oy, box[2]+ox, box[3]+oy]
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

def iou_xyxy(a, b):
    if not a or not b: return 0.0
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2-ix1), max(0.0, iy2-iy1)
    inter = iw * ih
    if inter <= 0: return 0.0
    a_area = max(0.0, (ax2-ax1)) * max(0.0, (ay2-ay1))
    b_area = max(0.0, (bx2-bx1)) * max(0.0, (by2-by1))
    denom = a_area + b_area - inter
    return inter/denom if denom > 0 else 0.0

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
    cv2.putText(frame, "ROI: WASD move, [ ] resize, r reset" if not LOCK_ROI else "ROI (locked)",
                (rx+6, max(18, ry-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255), 2, cv2.LINE_AA)

def draw_lp_boxes(frame, dets, color=(0,255,0)):
    for d in dets:
        xy = xyxy_from_det(d)
        if not xy: continue
        x1,y1,x2,y2 = map(int, xy)
        cv2.rectangle(frame, (x1,y1), (x2,y2), color, BOX_THICK)

def _ascii_safe(s: str) -> str:
    if not s: return ""
    return "".join(ch for ch in s if 32 <= ord(ch) < 127)

def plate_overlay_label(text: str) -> str:
    suf = plate_suffix(text or "")
    if suf: return suf
    t = _normalize_text(text or ""); t = _ascii_safe(t)
    return t[:24] if t else ""

# ---------- gate helpers ----------
def fully_inside_roi(det, rx, ry, rw, rh, m):
    xy = xyxy_from_det(det)
    if not xy: return False
    x1,y1,x2,y2 = xy
    return (x1 >= rx + m) and (y1 >= ry + m) and (x2 <= rx + rw - m) and (y2 <= ry + rh - m)

def center_in_vertical_band(det, ry, rh, band_frac):
    _, cy = center_from_det(det)
    if cy is None: return False
    mid = ry + rh*0.5
    half_band = (rh * band_frac) * 0.5
    return (mid - half_band) <= cy <= (mid + half_band)

def area_ok_in_roi(det, rw, rh, min_frac):
    a = bbox_area(det); roi_a = float(rw * rh)
    return roi_a > 0 and (a / roi_a) >= min_frac

# ---------- main ----------
def main():
    src = 0 if USE_CAMERA else INPUT_VIDEO_PATH
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {src}")

    if USE_CAMERA:
        try: cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except: pass

    try: cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)
    except: pass

    car_det = CarDetection(model_path="yolo11n.pt") if USE_CAR_DETECTOR else None
    lp_det  = LicencePlateDetection(model_path="models/best.pt")

    os.makedirs("output_videos", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or TARGET_WIDTH)
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or int(TARGET_WIDTH*3/5))
    out = cv2.VideoWriter("output_videos/output_video.avi", fourcc, fps, (W, H)) if WRITE_VIDEO else None

    csv_file, csv_writer, csv_path = open_daily_csv()
    print("[INFO] Logging to:", csv_path)

    # ROI
    rx = int(REGION_XYWH_RATIO[0]*W); ry = int(REGION_XYWH_RATIO[1]*H)
    rw = int(REGION_XYWH_RATIO[2]*W); rh = int(REGION_XYWH_RATIO[3]*H)
    rx, ry, rw, rh = clamp_region(rx, ry, rw, rh, W, H)

    idbank = IDBank(START_ID_AT)
    prev_inside = {}
    inside_count, outside_count, last_event_f = {}, {}, {}

    slot_pool = SlotPool(SLOT_COUNT)
    plate_pref_slot, active_plate_to_slot, active_id_to_plate = {}, {}, {}
    active_id_to_slot = {}

    # one-shot bookkeeping
    capture_open = {}    # oid -> frames left (0 close, -1 locked, -2 pending write)
    best_score  = {}     # oid -> area*sharpness
    best_text   = {}     # oid -> best OCR so far
    best_det    = {}     # oid -> det for best
    captured_in = set()

    # pending write / grace
    grace_left = {}      # oid -> frames left in grace
    pending_in = set()

    # gate state
    gate_ready      = {}
    gate_delay_left = {}

    # last-seen caches
    last_suf  = {}
    last_city = {}
    last_kana = {}
    last_class = {}      # class number cache

    # recent dedupe: (city|suffix) -> last frame index written
    recent_writes = {}
    DEDUP_MIN_GAP_FRAMES = 90   # ~3s at 30fps

    # overlay cache: oid -> {bbox, label, ttl}
    overlay_cache = {}

    # IoU association memory
    last_det = {}

    frame_idx = 0
    t0 = time.time(); fcount = 0; fps_est = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok: break
            frame_idx += 1; fcount += 1

            # ROI crop
            rx, ry, rw, rh = clamp_region(rx, ry, rw, rh, W, H)
            roi = crop_frame(frame, rx, ry, rw, rh)

            # --- fast/slow path decision ---
            periodic_heavy = (frame_idx % DETECT_EVERY_N == 0)
            have_locked_inside = any((oid in captured_in) and prev_inside.get(oid, False)
                                     for oid in prev_inside.keys())
            capture_active = any((v is not None) and (v != -1) and (v > 0)
                                 for v in capture_open.values()) or bool(pending_in)

            do_heavy_this_frame = periodic_heavy or capture_active
            if have_locked_inside and (not capture_active) and (frame_idx % INSIDE_PROCESS_EVERY_N != 0):
                do_heavy_this_frame = False

            lp_dets_full, lp_texts = [], []

            # Run detectors on heavy frames
            if do_heavy_this_frame:
                all_lp_dets, all_lp_texts = lp_det.detect_frames([roi])
                roi_dets = all_lp_dets[0]; lp_texts = all_lp_texts[0]
                lp_dets_full = [offset_det(d, rx, ry) for d in roi_dets]

            # Optional car boxes
            car_dets_full = []
            if USE_CAR_DETECTOR and do_heavy_this_frame:
                car_dets = car_det.detect_frames([roi], read_from_stub=False)[0]
                car_dets_full = [offset_det(d, rx, ry) for d in car_dets]

            # Draw overlays when we have new dets
            if DRAW_BBOXES and do_heavy_this_frame:
                draw_lp_boxes(frame, lp_dets_full, color=(0,255,0))
                for det, text in zip(lp_dets_full, lp_texts):
                    xy = xyxy_from_det(det)
                    if not xy: continue
                    x1,y1,x2,y2 = map(int, xy)
                    label = plate_overlay_label(text)
                    if not label: continue
                    tx, ty = x1 + 2, max(16, y1 - 6)
                    cv2.putText(frame, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 3, cv2.LINE_AA)
                    cv2.putText(frame, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255), 1, cv2.LINE_AA)
                    skey = stable_key(text, det)
                    if skey:
                        oid_tmp = idbank.get(skey)
                        overlay_cache[oid_tmp] = {"bbox": (x1,y1,x2,y2), "label": label, "ttl": OVERLAY_PERSIST_FRAMES}
                if USE_CAR_DETECTOR:
                    draw_lp_boxes(frame, car_dets_full, color=(255,200,0))

            # Persist cached overlays on throttled frames
            if DRAW_BBOXES:
                to_del = []
                for oid_tmp, entry in overlay_cache.items():
                    ttl = entry.get("ttl", 0)
                    if ttl <= 0: to_del.append(oid_tmp); continue
                    x1,y1,x2,y2 = entry.get("bbox", (0,0,0,0))
                    label = entry.get("label", "")
                    if label:
                        tx, ty = int(x1) + 2, max(16, int(y1) - 6)
                        cv2.putText(frame, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 3, cv2.LINE_AA)
                        cv2.putText(frame, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255), 1, cv2.LINE_AA)
                    entry["ttl"] = ttl - 1
                for kdel in to_del:
                    overlay_cache.pop(kdel, None)
                draw_region(frame, rx, ry, rw, rh)

            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # ---------------- Process detections ----------------
            if do_heavy_this_frame:
                matched_oids = set()
                for det, text in zip(lp_dets_full, lp_texts):
                    # Associate with existing OID by IoU first
                    xy_now = xyxy_from_det(det)
                    chosen_oid = None; best_iou = 0.0
                    if xy_now is not None:
                        for cand, xy_prev in last_det.items():
                            if cand in matched_oids: continue
                            i = iou_xyxy(xy_now, xy_prev)
                            if i > best_iou:
                                best_iou = i; chosen_oid = cand
                    if chosen_oid is not None and best_iou >= 0.45:
                        oid = chosen_oid
                    else:
                        skey = stable_key(text, det)
                        if skey is None: continue
                        oid = idbank.get(skey)
                    matched_oids.add(oid)

                    cx, cy = center_from_det(det)
                    is_in = inside_region(cx, cy, rx, ry, rw, rh)

                    # update last-seen caches
                    suf_now = plate_suffix(text or "")
                    if suf_now: last_suf[oid] = suf_now
                    c_now, klass_now, k_now = parse_city_class_kana(text or "")
                    if c_now:      last_city[oid]  = c_now
                    if klass_now:  last_class[oid] = klass_now
                    if k_now:      last_kana[oid]  = k_now

                    # init
                    if oid not in prev_inside:
                        prev_inside[oid] = is_in
                        inside_count[oid]  = 1 if is_in else 0
                        outside_count[oid] = 0 if is_in else 1
                        gate_ready[oid]      = False
                        gate_delay_left[oid] = 0
                        continue

                    # streaks
                    if is_in:
                        inside_count[oid]  = inside_count.get(oid,0) + 1
                        outside_count[oid] = 0
                    else:
                        outside_count[oid] = outside_count.get(oid,0) + 1
                        inside_count[oid]  = 0

                    # already locked? skip
                    if capture_open.get(oid, 0) == -1:
                        prev_inside[oid] = is_in
                        continue

                    # ----- One-shot capture gating -----
                    if is_in and (oid not in captured_in):
                        margin = int(min(rw, rh) * FULLY_INSIDE_MARGIN_FRAC)
                        gate_fully  = (not REQUIRE_FULLY_INSIDE) or fully_inside_roi(det, rx, ry, rw, rh, margin)
                        gate_center = center_in_vertical_band(det, ry, rh, CENTER_BAND_FRAC)
                        gate_area   = area_ok_in_roi(det, rw, rh, MIN_PLATE_AREA_FRAC)
                        gate_pass   = gate_fully and gate_center and gate_area

                        if gate_pass and not gate_ready.get(oid, False) and capture_open.get(oid, None) is None:
                            gate_ready[oid] = True
                            gate_delay_left[oid] = ENTER_DELAY_FRAMES

                        if gate_ready.get(oid, False) and capture_open.get(oid, None) is None:
                            if gate_delay_left.get(oid, 0) > 0:
                                gate_delay_left[oid] -= 1
                            else:
                                capture_open[oid] = CAPTURE_WINDOW_FRAMES
                                best_score[oid] = 0.0; best_text[oid] = ""; best_det[oid] = det

                        # inside window: track best sharp suffix frame
                        if capture_open.get(oid, 0) and capture_open[oid] > 0:
                            x1,y1,x2,y2 = map(int, xyxy_from_det(det))
                            y1c,y2c = max(0,y1), min(int(H),y2)
                            x1c,x2c = max(0,x1), min(int(W),x2)
                            plate_img = frame[y1c:y2c, x1c:x2c]
                            sharp = variance_of_laplacian(plate_img)
                            score = bbox_area(det) * max(1.0, sharp)
                            if score > best_score.get(oid, 0.0) and plate_suffix(text or ""):
                                best_score[oid] = score; best_text[oid]  = text; best_det[oid] = det
                            if plate_suffix(text or "") and sharp >= MIN_SHARPNESS_LOCK:
                                capture_open[oid] = 0
                            else:
                                capture_open[oid] -= 1

                        # window closed: begin short grace fill before writing IN
                        if capture_open.get(oid, None) == 0 and inside_count[oid] >= ENTER_STABLE_FRAMES:
                            pending_in.add(oid)
                            grace_left[oid] = FILL_GRACE_FRAMES
                            capture_open[oid] = -2  # pending write

                    # ----- EXIT handling -----
                    if prev_inside[oid] and (not is_in) and outside_count[oid] >= EXIT_STABLE_FRAMES:
                        # force IN if still pending
                        if oid in pending_in:
                            chosen_text = best_text.get(oid,"")
                            suf = plate_suffix(chosen_text) or last_suf.get(oid)
                            city = parse_city_class_kana(chosen_text)[0] or last_city.get(oid)
                            klass = parse_city_class_kana(chosen_text)[1] or last_class.get(oid)
                            kana = parse_city_class_kana(chosen_text)[2] or last_kana.get(oid)
                            pk = plate_key(chosen_text) if suf else None
                            if pk and (frame_idx - recent_writes.get(pk, -10**9) < DEDUP_MIN_GAP_FRAMES):
                                pass  # skip duplicate IN
                            else:
                                chosen_slot = active_id_to_slot.get(oid) or slot_pool.acquire_lowest() or 1
                                if pk:
                                    active_plate_to_slot[pk] = chosen_slot
                                    plate_pref_slot[pk] = chosen_slot
                                    active_id_to_plate[oid] = pk
                                active_id_to_slot[oid] = chosen_slot
                                object_id = f"{chosen_slot:02d}({suf})" if suf else f"{chosen_slot:02d}"
                                write_row_flush(csv_writer, csv_file,
                                                [ts, object_id, "car", "in", city or "", klass or "", kana or ""])
                                if pk: recent_writes[pk] = frame_idx
                            captured_in.add(oid)
                            pending_in.discard(oid)
                            capture_open[oid] = -1

                        if frame_idx - last_event_f.get(oid, -10**9) >= MIN_EVENT_GAP_FRAMES:
                            pk_now = active_id_to_plate.get(oid)
                            slot_to_free = None
                            if pk_now and pk_now in active_plate_to_slot:
                                slot_to_free = active_plate_to_slot.pop(pk_now, None)
                            if slot_to_free is None:
                                slot_to_free = active_id_to_slot.pop(oid, None)

                            if slot_to_free is not None:
                                chosen_text = best_text.get(oid,"")
                                suf_out = plate_suffix(chosen_text) or last_suf.get(oid)
                                city = parse_city_class_kana(chosen_text)[0] or last_city.get(oid)
                                klass_out = parse_city_class_kana(chosen_text)[1] or last_class.get(oid)
                                kana = parse_city_class_kana(chosen_text)[2] or last_kana.get(oid)

                                label_for_id = plate_overlay_label(chosen_text) or overlay_cache.get(oid, {}).get("label", "")
                                obj_tail = (suf_out or label_for_id)
                                object_id = f"{slot_to_free:02d}({obj_tail})" if obj_tail else f"{slot_to_free:02d}"

                                pk = plate_key(chosen_text) if suf_out else None
                                if not pk or (frame_idx - recent_writes.get(pk, -10**9) >= DEDUP_MIN_GAP_FRAMES):
                                    write_row_flush(csv_writer, csv_file,
                                                    [ts, object_id, "car", "out", city or "", klass_out or "", kana or ""])
                                    if pk: recent_writes[pk] = frame_idx

                                slot_pool.release(slot_to_free)
                                last_event_f[oid] = frame_idx

                            # clear state
                            for dct in (capture_open, best_score, best_text, best_det,
                                        gate_ready, gate_delay_left, grace_left):
                                dct.pop(oid, None)
                            pending_in.discard(oid)
                            if oid in captured_in: captured_in.remove(oid)
                            last_det.pop(oid, None)

                    prev_inside[oid] = is_in
                    if xy_now is not None:
                        last_det[oid] = xy_now

            # ----- GRACE FILL: keep trying briefly, then write IN once -----
            if do_heavy_this_frame and pending_in:
                to_write = []
                for oid in list(pending_in):
                    chosen_text = best_text.get(oid, "")
                    suf = plate_suffix(chosen_text) or last_suf.get(oid)
                    city, klass, kana = parse_city_class_kana(chosen_text)
                    if not city:  city  = last_city.get(oid)
                    if not klass: klass = last_class.get(oid)
                    if not kana:  kana  = last_kana.get(oid)

                    if suf or klass or city or (grace_left.get(oid, 0) <= 0):
                        to_write.append((oid, chosen_text, suf, city, klass, kana))
                    else:
                        grace_left[oid] = grace_left.get(oid, FILL_GRACE_FRAMES) - 1

                for oid, chosen_text, suf, city, klass, kana in to_write:
                    pk = plate_key(chosen_text) if suf else None
                    if pk and (frame_idx - recent_writes.get(pk, -10**9) < DEDUP_MIN_GAP_FRAMES):
                        pending_in.discard(oid)
                        capture_open[oid] = -1
                        captured_in.add(oid)
                        last_event_f[oid] = frame_idx
                        continue

                    # allocate slot
                    chosen_slot = None
                    if pk and pk in plate_pref_slot:
                        chosen_slot = slot_pool.try_acquire_specific(plate_pref_slot[pk])
                    if chosen_slot is None:
                        chosen_slot = active_id_to_slot.get(oid) or slot_pool.acquire_lowest()
                    if chosen_slot is None: chosen_slot = 1

                    if pk:
                        active_plate_to_slot[pk] = chosen_slot
                        plate_pref_slot[pk] = chosen_slot
                        active_id_to_plate[oid] = pk
                    else:
                        active_id_to_plate[oid] = None
                    active_id_to_slot[oid] = chosen_slot

                    label_for_id = plate_overlay_label(chosen_text) or overlay_cache.get(oid, {}).get("label", "")
                    obj_tail = (suf or label_for_id)
                    object_id = f"{chosen_slot:02d}({obj_tail})" if obj_tail else f"{chosen_slot:02d}"

                    write_row_flush(csv_writer, csv_file,
                                    [ts, object_id, "car", "in", city or "", klass or "", kana or ""])
                    if pk: recent_writes[pk] = frame_idx
                    captured_in.add(oid); pending_in.discard(oid)
                    capture_open[oid] = -1
                    last_event_f[oid] = frame_idx

            # FPS overlay
            if SHOW_FPS:
                now = time.time()
                if now - t0 >= 0.5:
                    fps_est = fcount / (now - t0); t0 = now; fcount = 0
                tag = "" if do_heavy_this_frame else "  (throttle)"
                cv2.putText(frame, f"FPS ~ {fps_est:.1f}{tag}", (10, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

            # Display / write
            cv2.imshow("Parking (region14 fast+robust)", frame)
            if out is not None and WRITE_VIDEO: out.write(frame)

            # optional ROI keyboard control
            k = cv2.waitKey(1) & 0xFF
            if k == 27: break
            if not LOCK_ROI:
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
