# -*- coding: utf-8 -*-
# region13.py — ROI-cropped, one-shot plate capture with "perfect-moment" gate
# Logs plate data robustly using tolerant parsing + last-seen fallback + engine size

import os, cv2, csv, re, time
from datetime import datetime
from detections import CarDetection, LicencePlateDetection

# ===================== User settings =====================
USE_CAMERA               = True
INPUT_VIDEO_PATH         = "input_videos/Trim03.mp4"

SLOT_COUNT               = 10
START_ID_AT              = 100

# Region as ratios of frame (x, y, w, h)
REGION_XYWH_RATIO        = (0.32, 0.60, 0.36, 0.20)

# ROI lock (False = user can move/resize with keyboard)
LOCK_ROI                 = False

# Stability & gaps
ENTER_STABLE_FRAMES      = 1
EXIT_STABLE_FRAMES       = 4
MIN_EVENT_GAP_FRAMES     = 10

# One-shot window (fallback if sharpness never crosses threshold)
CAPTURE_WINDOW_FRAMES    = 6

# While a car remains inside and already captured, only process heavy every N frames
INSIDE_PROCESS_EVERY_N   = 8

# One-shot accept rules
MIN_SHARPNESS_LOCK       = 60.0   # Laplacian variance to accept immediately

# ── One-shot gating: wait for a "perfect" moment ────────────────────────────
ENTER_DELAY_FRAMES       = 3
REQUIRE_FULLY_INSIDE     = True
FULLY_INSIDE_MARGIN_FRAC = 0.05
CENTER_BAND_FRAC         = 0.40
MIN_PLATE_AREA_FRAC      = 0.015

# Performance toggles
USE_CAR_DETECTOR         = False
WRITE_VIDEO              = False
DRAW_BBOXES              = True
SHOW_FPS                 = True
TARGET_WIDTH             = 640
BOX_THICK                = 1
# Persist overlays across throttled frames
OVERLAY_PERSIST_FRAMES   = 12
AUTO_ROTATE_CSV_HEADER_MISMATCH = True
# =========================================================

# Reduce OpenCV overhead a bit
cv2.setUseOptimized(True)
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")
os.environ.setdefault("FLAGS_minloglevel", "2")

# ---- tolerant parsers -------------------------------------------------------
# Accept many dash variants and ignore stray symbols around numbers
_DASHES = "-–—ー‐−~"  # common unicode dashes
_PLATE_SUFFIX_FLEX = re.compile(rf"(\d{{2,3}})\s*[-{_DASHES}]\s*(\d{{2}})")
CITY_KANA_RE = re.compile(u"^\\s*([^\\s\\d]+)\\s+[0-9]{3,4}\\s*([ぁ-ゖァ-ヿ一-龯A-Za-z])", re.UNICODE)

def _normalize_text(t: str) -> str:
    if not t: return ""
    # unify dashes to hyphen, strip non-printing, collapse spaces
    trans = str.maketrans({c: "-" for c in _DASHES})
    t = t.translate(trans)
    t = t.replace("—", "-").replace("–", "-")
    return " ".join(t.split())

def _normalize_for_digits(t: str) -> str:
    """Fix common OCR confusions for numeric groups in plates."""
    if not t: return ""
    repl = str.maketrans({
        'O': '0', 'o': '0', 'D': '0',
        'I': '1', 'l': '1', 'L': '1', 'i': '1',
        'S': '5', 's': '5',
        'B': '8',
        'Z': '2', 'z': '2',
        'q': '9', 'Q': '9',
    })
    return t.translate(repl)

def parse_city_kana(text):
    t = _normalize_text(text or "")
    if not t: return None, None
    m = CITY_KANA_RE.search(t)
    return (m.group(1), m.group(2)) if m else (None, None)

def plate_suffix(text):
    t = _normalize_text(text or "")
    t = _normalize_for_digits(t)
    m = _PLATE_SUFFIX_FLEX.search(t)
    if not m: return None
    return f"{m.group(1)}-{m.group(2)}"

def plate_key(text):
    suf = plate_suffix(text or "");  city, _ = parse_city_kana(text or "")
    if not suf: return None
    return f"{city if city else '?'}|{suf}"

def engine_size_from_suffix(suf: str):
    """Return left part of NN-NN (e.g., '70-50' -> '70')."""
    if not suf: return ""
    try:
        return str(suf).split("-")[0]
    except Exception:
        return ""

def open_daily_csv():
    date_str = datetime.now().strftime("%Y%m%d")
    os.makedirs("logs", exist_ok=True)
    path = os.path.join("logs", f"parking_log_{date_str}.csv")
    new = not os.path.exists(path)

    # If file exists, check header and optionally rotate
    if not new:
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as rf:
                first = rf.readline()
            required = ["object_id","plate_text","vehicle_type","direction","city","engine_size","kana"]
            if first and any(h not in first for h in required) and AUTO_ROTATE_CSV_HEADER_MISMATCH:
                base, ext = os.path.splitext(path)
                bak = f"{base}_old_{datetime.now().strftime('%H%M%S')}{ext}"
                os.replace(path, bak)
                print(f"[INFO] Rotated old CSV without engine_size -> {bak}")
                new = True
        except Exception as e:
            print("[WARN] Could not inspect/rotate CSV header:", e)

    f = open(path, "a", newline="", encoding="utf-8", buffering=1)  # line-buffered
    w = csv.writer(f)
    if new:
        # include plate_text and engine_size columns
        w.writerow(["timestamp","object_id","plate_text","vehicle_type","direction","city","engine_size","kana"])
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
    """Prefer numeric plate suffix; fallback to normalized ASCII text."""
    suf = plate_suffix(text or "")
    if suf:
        return suf
    t = _normalize_text(text or "")
    t = _ascii_safe(t)
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
    capture_open = {}    # oid -> frames left (0 close, -1 locked)
    best_score  = {}     # oid -> area*sharpness
    best_text   = {}     # oid -> best OCR so far
    best_det    = {}     # oid -> det for best
    captured_in = set()

    # gate state
    gate_ready      = {}
    gate_delay_left = {}

    # NEW: last-seen cache (robust CSV fill even if window missed the best tokens)
    last_suf  = {}
    last_city = {}
    last_kana = {}

    # overlay cache: oid -> {bbox:(x1,y1,x2,y2), label:str, ttl:int}
    overlay_cache = {}

    # IoU association memory: last bbox per oid
    last_det = {}

    frame_idx = 0
    t0 = time.time(); fcount = 0; fps_est = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok: break
            frame_idx += 1; fcount += 1

            # --- ROI crop for inference only ---
            rx, ry, rw, rh = clamp_region(rx, ry, rw, rh, W, H)
            roi = crop_frame(frame, rx, ry, rw, rh)

            # throttle heavy work if any captured car is still inside
            have_locked_inside = any((oid in captured_in) and prev_inside.get(oid, False) for oid in prev_inside.keys())
            do_heavy_this_frame = True
            if have_locked_inside and (frame_idx % INSIDE_PROCESS_EVERY_N != 0):
                do_heavy_this_frame = False

            lp_dets_full, lp_texts = [], []

            if do_heavy_this_frame:
                all_lp_dets, all_lp_texts = lp_det.detect_frames([roi])
                roi_dets = all_lp_dets[0]; lp_texts = all_lp_texts[0]
                lp_dets_full = [offset_det(d, rx, ry) for d in roi_dets]

            # Optional car boxes
            car_dets_full = []
            if USE_CAR_DETECTOR and do_heavy_this_frame:
                car_dets = car_det.detect_frames([roi], read_from_stub=False)[0]
                for d in car_dets:
                    car_dets_full.append(offset_det(d, rx, ry))

            # Draw overlays
            if DRAW_BBOXES and do_heavy_this_frame:
                draw_lp_boxes(frame, lp_dets_full, color=(0,255,0))
                # Overlay plate text/suffix near each detected plate box
                for det, text in zip(lp_dets_full, lp_texts):
                    xy = xyxy_from_det(det)
                    if not xy: continue
                    x1,y1,x2,y2 = map(int, xy)
                    label = plate_overlay_label(text)
                    if not label: continue
                    tx, ty = x1 + 2, max(16, y1 - 6)
                    # outline for readability
                    cv2.putText(frame, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 3, cv2.LINE_AA)
                    cv2.putText(frame, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255), 1, cv2.LINE_AA)
                    # cache for persistence
                    oid_tmp = idbank.get(stable_key(text, det)) if stable_key(text, det) else None
                    if oid_tmp is not None:
                        overlay_cache[oid_tmp] = {"bbox": (x1,y1,x2,y2), "label": label, "ttl": OVERLAY_PERSIST_FRAMES}
                if USE_CAR_DETECTOR:
                    draw_lp_boxes(frame, car_dets_full, color=(255,200,0))
            
            # Persist cached overlays on any frame
            if DRAW_BBOXES:
                # decay and draw
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

            # Process detections
            if do_heavy_this_frame:
                matched_oids = set()
                for det, text in zip(lp_dets_full, lp_texts):
                    # Associate with existing OID by IoU first
                    xy_now = xyxy_from_det(det)
                    chosen_oid = None
                    best_i = 0.0
                    if xy_now is not None:
                        for cand_oid, xy_prev in last_det.items():
                            if cand_oid in matched_oids: continue
                            i = iou_xyxy(xy_now, xy_prev)
                            if i > best_i:
                                best_i = i; chosen_oid = cand_oid
                    if chosen_oid is not None and best_i >= 0.45:
                        oid = chosen_oid
                    else:
                        key = stable_key(text, det)
                        if key is None: continue
                        oid = idbank.get(key)
                    matched_oids.add(oid)

                    cx, cy = center_from_det(det)
                    is_in = inside_region(cx, cy, rx, ry, rw, rh)

                    # update last-seen cache from any text we get this frame
                    suf_now = plate_suffix(text or "")
                    if suf_now: last_suf[oid] = suf_now
                    c_now, k_now = parse_city_kana(text or "")
                    if c_now: last_city[oid] = c_now
                    if k_now: last_kana[oid] = k_now

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

                    # skip already locked
                    if capture_open.get(oid, 0) == -1:
                        prev_inside[oid] = is_in
                        continue

                    # gated one-shot
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

                        if capture_open.get(oid, 0) and capture_open[oid] > 0:
                            x1,y1,x2,y2 = map(int, xyxy_from_det(det))
                            y1c,y2c = max(0,y1), min(int(H),y2)
                            x1c,x2c = max(0,x1), min(int(W),x2)
                            plate_img = frame[y1c:y2c, x1c:x2c]
                            sharp = variance_of_laplacian(plate_img)
                            score = bbox_area(det) * max(1.0, sharp)

                            if score > best_score.get(oid, 0.0) and plate_suffix(text or ""):
                                best_score[oid] = score
                                best_text[oid]  = text
                                best_det[oid]   = det

                            if plate_suffix(text or "") and sharp >= MIN_SHARPNESS_LOCK:
                                capture_open[oid] = 0
                            else:
                                capture_open[oid] -= 1

                        if capture_open.get(oid, 0) == 0 and inside_count[oid] >= ENTER_STABLE_FRAMES:
                            # build final fields with robust fallbacks
                            chosen_text = best_text.get(oid,"") or text or ""
                            suf = plate_suffix(chosen_text) or last_suf.get(oid)
                            city, kana = parse_city_kana(chosen_text)
                            if not city: city = last_city.get(oid)
                            if not kana: kana = last_kana.get(oid)

                            # allocate slot
                            pk = plate_key(chosen_text) if suf else None
                            chosen_slot = None
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

                                label_for_id = plate_overlay_label(chosen_text) or overlay_cache.get(oid, {}).get("label", "")
                                obj_tail = suf or label_for_id
                                object_id = f"{chosen_slot:02d}({obj_tail})" if obj_tail else f"{chosen_slot:02d}"
                                engine_size = engine_size_from_suffix(suf)
                                plate_text_col = (chosen_text or "").strip() or label_for_id or (suf or "")
                                write_row_flush(csv_writer, csv_file,
                                                [ts, object_id, plate_text_col, "car", "in", city or "", engine_size, kana or ""])
                                captured_in.add(oid)
                                last_event_f[oid] = frame_idx
                                capture_open[oid] = -1  # lock
                                if DRAW_BBOXES:
                                    cv2.putText(frame, f"IN {object_id}", (int(cx), int(cy)),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv2.LINE_AA)

                    # EXIT handling
                    if prev_inside[oid] and (not is_in) and outside_count[oid] >= EXIT_STABLE_FRAMES:
                        if frame_idx - last_event_f.get(oid, -10**9) >= MIN_EVENT_GAP_FRAMES:
                            pk_now = active_id_to_plate.get(oid)
                            slot_to_free = None
                            if pk_now and pk_now in active_plate_to_slot:
                                slot_to_free = active_plate_to_slot.pop(pk_now, None)
                            # Fallback: free by oid if we never had a stable plate key
                            if slot_to_free is None:
                                slot_to_free = active_id_to_slot.pop(oid, None)

                            if slot_to_free is not None:
                                chosen_text = best_text.get(oid,"") or text or ""
                                suf_out = plate_suffix(chosen_text) or last_suf.get(oid)
                                city, kana = parse_city_kana(chosen_text)
                                if not city: city = last_city.get(oid)
                                if not kana: kana = last_kana.get(oid)

                                label_for_id = plate_overlay_label(chosen_text) or overlay_cache.get(oid, {}).get("label", "")
                                obj_tail = suf_out or label_for_id
                                object_id = f"{slot_to_free:02d}({obj_tail})" if obj_tail else f"{slot_to_free:02d}"
                                engine_size_out = engine_size_from_suffix(suf_out)
                                plate_text_col = (chosen_text or "").strip() or label_for_id or (suf_out or "")
                                write_row_flush(csv_writer, csv_file,
                                                [ts, object_id, plate_text_col, "car", "out", city or "", engine_size_out, kana or ""])
                                slot_pool.release(slot_to_free)
                                last_event_f[oid] = frame_idx
                                if DRAW_BBOXES:
                                    cv2.putText(frame, f"OUT {object_id}", (int(cx), int(cy)),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv2.LINE_AA)

                            # clear state so future re-entry will capture again
                            for dct in (capture_open, best_score, best_text, best_det, gate_ready, gate_delay_left):
                                dct.pop(oid, None)
                            if oid in captured_in: captured_in.remove(oid)
                            last_det.pop(oid, None)

                    prev_inside[oid] = is_in
                    # update association memory
                    if xy_now is not None:
                        last_det[oid] = xy_now

            # FPS overlay
            if SHOW_FPS:
                now = time.time()
                if now - t0 >= 0.5:
                    fps_est = fcount / (now - t0); t0 = now; fcount = 0
                tag = "" if do_heavy_this_frame else "  (throttle)"
                cv2.putText(frame, f"FPS ~ {fps_est:.1f}{tag}", (10, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

            # Display / write
            cv2.imshow("Parking (region13 ROI one-shot, gated)", frame)
            if out is not None and WRITE_VIDEO: out.write(frame)

            # optional ROI keyboard control if not locked
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
