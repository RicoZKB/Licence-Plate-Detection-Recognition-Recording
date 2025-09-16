# -*- coding: utf-8 -*-
# segion08.py — ROI-cropped, one-shot plate capture + throttled OCR while inside
# - Runs detector/OCR only INSIDE the yellow ROI (cropped inference = faster)
# - Logs once on ENTER as soon as a "good" frame appears (sharp + valid suffix)
# - Skips further OCR for that object until EXIT (lightweight while inside)
# - Immediate CSV flush on every write

import os, cv2, csv, re, time
from datetime import datetime
from detections import CarDetection, LicencePlateDetection

# ===================== User settings =====================
USE_CAMERA               = False
INPUT_VIDEO_PATH         = "input_videos/Trim02.mp4"

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
ENTER_STABLE_FRAMES      = 1     # we lock early anyway
EXIT_STABLE_FRAMES       = 4
MIN_EVENT_GAP_FRAMES     = 10

# One-shot window (fallback in case sharpness never crosses threshold)
CAPTURE_WINDOW_FRAMES    = 6

# While car remains inside, only process heavy every N frames
INSIDE_PROCESS_EVERY_N   = 8
DETECT_EVERY_N           = 1     # run detector every N frames (global throttle)

# One-shot accept rules
# Accept hyphen variants: - − – — ー ｰ ~ 〜 and optional spaces
PLATE_SUFFIX_RE          = re.compile(r"(\d{2,3})\s*[-−–—ーｰ~〜－]\s*(\d{2})")
MIN_SHARPNESS_LOCK       = 60.0   # Laplacian variance to accept immediately

# Performance toggles
USE_CAR_DETECTOR         = False  # more load if True
WRITE_VIDEO              = False
DRAW_BBOXES              = True
SHOW_FPS                 = True
TARGET_WIDTH             = 640    # small input for speed
BOX_THICK                = 1      # thin overlays
# If ROI is wider than this, downscale before detection for speed
ROI_INFER_MAX_W          = 640
GATE_DET_BAND_RATIO      = 0.50  # width of detection band around gate (0..1 of W)
# =========================================================

# Reduce OpenCV overhead a bit
cv2.setUseOptimized(True)
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")
# Silence some noisy frameworks if present
os.environ.setdefault("FLAGS_minloglevel", "2")

CITY_KANA_RE = re.compile(
    u"^\\s*([^\\s\\d]+)\\s*[0-9]{2,4}.*?([ぁ-ゖァ-ヿA-Za-z])",
    re.UNICODE,
)

def parse_city_kana(text):
    if not text: return None, None
    m = CITY_KANA_RE.search(text)
    return (m.group(1), m.group(2)) if m else (None, None)

def normalize_ocr_text(text: str) -> str:
    if not text:
        return ""
    # Convert full-width digits to ASCII
    fw_digits = "０１２３４５６７８９"
    ascii_digits = "0123456789"
    trans = str.maketrans({fw: ascii for fw, ascii in zip(fw_digits, ascii_digits)})
    t = text.translate(trans)
    # Normalize hyphen-like characters to '-'
    t = re.sub(r"[−–—ーｰ~〜－]", "-", t)
    # Collapse whitespace
    t = re.sub(r"\s+", " ", t).strip()
    return t

def plate_suffix(text):
    if not text: return None
    t = normalize_ocr_text(text)
    m = PLATE_SUFFIX_RE.search(t)
    if not m:
        return None
    # Return normalized form with ASCII hyphen
    return f"{m.group(1)}-{m.group(2)}"

def plate_key(text):
    t = normalize_ocr_text(text or "")
    suf = plate_suffix(t);  city, _ = parse_city_kana(t)
    if not suf: return None
    return f"{city if city else '?'}|{suf}"

# More robust extraction for fields like: "世田谷310 70-50 ふ"
REGION_WITH_NUM_RE = re.compile(u"([ぁ-ゖァ-ヿ一-龯A-Za-z]+)\\s*([0-9]{2,4})", re.UNICODE)

def parse_plate_fields(text):
    """Extract (region_class, suffix, kana, city) from OCR text.
    - region_class: e.g. "世田谷310"
    - suffix: e.g. "70-50"
    - kana: e.g. "ふ"
    - city: e.g. "世田谷"
    """
    if not text:
        return None, None, None, None
    # suffix
    t = normalize_ocr_text(text)
    suf = plate_suffix(t)
    # region + class
    city = None; region_class = None
    m = REGION_WITH_NUM_RE.search(t)
    if m:
        city = m.group(1)
        region_class = f"{m.group(1)}{m.group(2)}"
    # kana: choose last kana char if multiple
    kana = None
    kana_matches = re.findall(u"[ぁ-ゖァ-ヿ]", t)
    if kana_matches:
        kana = kana_matches[-1]
    return region_class, suf, kana, city

def open_daily_csv():
    date_str = datetime.now().strftime("%Y%m%d")
    os.makedirs("logs", exist_ok=True)
    path = os.path.join("logs", f"parking_log_{date_str}.csv")
    new = not os.path.exists(path)
    f = open(path, "a", newline="", encoding="utf-8", buffering=1)  # line-buffered
    w = csv.writer(f)
    if new:
        w.writerow([
            "timestamp", "object_id", "vehicle_type", "direction",
            "region_class", "suffix", "kana", "city"
        ])
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
    """Shift a detection back to full-frame coords after ROI crop."""
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

def scale_det(det, sx, sy):
    """Scale a detection box by (sx, sy) in-place for dicts, or return scaled list."""
    if isinstance(det, (list, tuple)) and len(det) >= 4:
        x1,y1,x2,y2 = det[:4]
        return [x1*sx, y1*sy, x2*sx, y2*sy]
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

# ---------- main ----------
def main():
    src = 0 if USE_CAMERA else INPUT_VIDEO_PATH
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {src}")

    if USE_CAMERA:
        # keep the camera buffer tiny; prefer fresh frames
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

    # ROI (or full frame if USE_ROI=False)
    if USE_ROI:
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
    # Additional mappings for robust dedup and release
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

            # --- ROI bookkeeping for overlays (and optional keyboard control) ---
            rx, ry, rw, rh = clamp_region(rx, ry, rw, rh, W, H)
            roi = crop_frame(frame, rx, ry, rw, rh)

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
                if USE_ROI:
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

            # Optional car boxes (also cropped for speed)
            car_dets_full = []
            if USE_CAR_DETECTOR and do_heavy_this_frame:
                # keep car detector in sync with detection ROI scaling
                gate_x = int(ENTRY_LINE_X_RATIO * W)
                if USE_ROI:
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
                car_dets = car_det.detect_frames([inf_roi], read_from_stub=False)[0]
                if scale != 1.0:
                    inv_sx = 1.0/scale; inv_sy = 1.0/scale
                    for d in car_dets:
                        d_scaled = scale_det(d, inv_sx, inv_sy)
                        car_dets_full.append(offset_det(d_scaled, d_rx, d_ry))
                else:
                    for d in car_dets:
                        car_dets_full.append(offset_det(d, d_rx, d_ry))

            # Draw overlays
            if DRAW_BBOXES and do_heavy_this_frame:
                # draw our thin LP boxes (faster than calling detector drawer)
                draw_lp_boxes(frame, lp_dets_full, color=(0,255,0))
                if USE_CAR_DETECTOR:
                    draw_lp_boxes(frame, car_dets_full, color=(255,200,0))
            if DRAW_BBOXES:
                if USE_ROI:
                    draw_region(frame, rx, ry, rw, rh)
                elif DRAW_GATE:
                    gx = int(ENTRY_LINE_X_RATIO * W)
                    cv2.line(frame, (gx, 0), (gx, H), (200, 200, 255), 2)
                    cv2.putText(frame, "gate", (gx+6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,255), 2, cv2.LINE_AA)

            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # gate x for no-ROI mode
            gate_x = int(ENTRY_LINE_X_RATIO * W)

            # Process detections (only when we actually ran them)
            if do_heavy_this_frame:
                for det, text in zip(lp_dets_full, lp_texts):
                    key = stable_key(text, det)
                    if key is None: continue
                    oid = idbank.get(key)

                    cx, cy = center_from_det(det)
                    if USE_ROI:
                        is_in = inside_region(cx, cy, rx, ry, rw, rh)
                    else:
                        # gate orientation: inside can be right or left of the line
                        if cx is None:
                            is_in = False
                        else:
                            is_in = (cx >= gate_x) if GATE_INSIDE_IS_RIGHT else (cx <= gate_x)

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

                    # skip anything already locked (-1)
                    if capture_open.get(oid, 0) == -1:
                        prev_inside[oid] = is_in
                        continue

                    # Open capture window on transition across gate (no-ROI),
                    # or when entering ROI (ROI mode)
                    if (not USE_ROI) and (not prev_inside[oid]) and is_in:
                        capture_open[oid] = CAPTURE_WINDOW_FRAMES
                        best_score[oid] = 0.0; best_text[oid] = ""; best_det[oid] = det

                    # Also open when approaching gate within margin (no-ROI), from the outside side
                    if (not USE_ROI) and (oid not in captured_in) and (capture_open.get(oid, 0) <= 0):
                        if cx is not None and abs(cx - gate_x) <= GATE_CAPTURE_MARGIN_PX:
                            # Only trigger if approaching from the outside direction
                            approaching_from_outside = (cx < gate_x) if GATE_INSIDE_IS_RIGHT else (cx > gate_x)
                            if not approaching_from_outside:
                                pass
                            else:
                                capture_open[oid] = CAPTURE_WINDOW_FRAMES
                                best_score[oid] = 0.0; best_text[oid] = ""; best_det[oid] = det

                    # one-shot capture maintenance
                    if is_in and (oid not in captured_in):
                        plate_img = None
                        # rank by area*sharpness as fallback
                        if capture_open.get(oid, 0) > 0:
                            x1,y1,x2,y2 = map(int, xyxy_from_det(det))
                            plate_img = frame[max(0,y1):min(H,y2), max(0,x1):min(W,x2)]
                            sharp = variance_of_laplacian(plate_img)
                            score = bbox_area(det) * max(1.0, sharp)

                            # keep best
                            if score > best_score.get(oid, 0.0) and plate_suffix(text or ""):
                                best_score[oid] = score
                                best_text[oid]  = text
                                best_det[oid]   = det

                            # early accept if sharp enough & suffix present
                            if plate_suffix(text or "") and sharp >= MIN_SHARPNESS_LOCK:
                                capture_open[oid] = 0  # close the window now
                            else:
                                capture_open[oid] -= 1

                        # when window closes (by countdown or early accept), log IN once
                        if capture_open.get(oid, 0) == 0 and inside_count[oid] >= ENTER_STABLE_FRAMES:
                            chosen_text = best_text.get(oid,"") or text or ""
                            pk = plate_key(chosen_text)
                            region_class, suf, kana, city = parse_plate_fields(chosen_text)

                            chosen_slot = None
                            # If this plate is already active, attach to existing slot and suppress duplicate IN
                            if pk and pk in active_plate_to_slot:
                                chosen_slot = active_plate_to_slot[pk]
                                active_id_to_plate[oid] = pk
                                active_id_to_slot[oid] = chosen_slot
                                if suf:
                                    active_id_to_suffix[oid] = suf
                                captured_in.add(oid)
                                last_event_f[oid] = frame_idx
                                capture_open[oid] = -1
                                # Do not write a duplicate CSV row
                                if DRAW_BBOXES:
                                    cv2.putText(frame, f"IN {chosen_slot:02d}(dup)", (int(cx), int(cy)),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,180,255), 2, cv2.LINE_AA)
                                # done for this oid
                                prev_inside[oid] = is_in
                                continue

                            # Fallback dedup: reuse slot by suffix if already active
                            if suf and suf in active_suffix_to_slot:
                                chosen_slot = active_suffix_to_slot[suf]
                                if pk:
                                    active_plate_to_slot[pk] = chosen_slot
                                    plate_pref_slot[pk] = chosen_slot
                                    active_id_to_plate[oid] = pk
                                active_id_to_slot[oid] = chosen_slot
                                active_id_to_suffix[oid] = suf
                                captured_in.add(oid)
                                last_event_f[oid] = frame_idx
                                capture_open[oid] = -1
                                if DRAW_BBOXES:
                                    cv2.putText(frame, f"IN {chosen_slot:02d}(dup)", (int(cx), int(cy)),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,180,255), 2, cv2.LINE_AA)
                                prev_inside[oid] = is_in
                                continue

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
                                # Always track id->slot to guarantee release even without pk
                                active_id_to_slot[oid] = chosen_slot
                                if suf:
                                    active_suffix_to_slot[suf] = chosen_slot
                                    active_id_to_suffix[oid] = suf

                                object_id = f"{chosen_slot:02d}({suf})" if suf else f"{chosen_slot:02d}"
                                write_row_flush(
                                    csv_writer, csv_file,
                                    [ts, object_id, "car", "in", region_class or "", suf or "", kana or "", city or ""]
                                )
                                captured_in.add(oid)
                                last_event_f[oid] = frame_idx
                                capture_open[oid] = -1  # LOCKED: skip future OCR until exit
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
                            # fallback: free by id->slot mapping when pk is missing
                            if slot_to_free is None:
                                slot_to_free = active_id_to_slot.pop(oid, None)

                            if slot_to_free is not None:
                                chosen_text = best_text.get(oid,"") or text or ""
                                region_class, suf_out, kana, city = parse_plate_fields(chosen_text)
                                object_id = f"{slot_to_free:02d}({suf_out})" if suf_out else f"{slot_to_free:02d}"
                                write_row_flush(
                                    csv_writer, csv_file,
                                    [ts, object_id, "car", "out", region_class or "", suf_out or "", kana or "", city or ""]
                                )
                                slot_pool.release(slot_to_free)
                                last_event_f[oid] = frame_idx
                                if DRAW_BBOXES:
                                    cv2.putText(frame, f"OUT {object_id}", (int(cx), int(cy)),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv2.LINE_AA)

                                # clear suffix mapping if it points to this slot
                                if suf_out and active_suffix_to_slot.get(suf_out) == slot_to_free:
                                    active_suffix_to_slot.pop(suf_out, None)

                            # always clear id->plate on exit
                            active_id_to_plate.pop(oid, None)
                            suf_for_oid = active_id_to_suffix.pop(oid, None)

                            # clear state so future re-entry will capture again
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

            # Display / write
            cv2.imshow("Parking (segion08 ROI one-shot)", frame)
            if out is not None and WRITE_VIDEO: out.write(frame)

            # optional ROI keyboard control if not locked (ROI mode only)
            k = cv2.waitKey(1) & 0xFF
            if k == 27: break
            if USE_ROI and (not LOCK_ROI):
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
