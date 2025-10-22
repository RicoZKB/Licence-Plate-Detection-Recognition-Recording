# -*- coding: utf-8 -*-
# per5_format.py — ROI/Gate + one-shot OCR + IN/OUT by direction (stable IDs)
# - Tiny centroid tracker to keep one oid per car (fixes OUT not firing)
# - Direction logic: crossing the vertical gate line decides IN/OUT
# - "Inside" is on the RIGHT side of the line (as you asked)
# - CSV stays at 8 columns

import os, cv2, csv, re, time
from collections import deque
from datetime import datetime
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
YOUTUBE_MAX_HEIGHT       = 480

# Terminal logging toggles
PRINT_EVENTS_TO_TERMINAL = True
PRINT_RAW_OCR_FOUND      = False
PASS_OCR_DEBUG_TO_DETECT = False
# =========================================================

SLOT_COUNT               = 10
START_ID_AT              = 100

# Region as ratios of frame (x, y, w, h)
REGION_XYWH_RATIO        = (0.32, 0.60, 0.36, 0.20)

LOCK_ROI                 = False

# Gate mode
USE_ROI                  = False
ENTRY_LINE_X_RATIO       = 0.50     # center line (use 0.55 if you prefer)
GATE_CAPTURE_MARGIN_PX   = 60
DRAW_GATE                = True
GATE_INSIDE_IS_RIGHT     = True     # Right of the line = inside (as requested)

# Stability & gaps
ENTER_STABLE_FRAMES      = 1
EXIT_STABLE_FRAMES       = 3
MIN_EVENT_GAP_FRAMES     = 6

# One-shot window (fallback)
CAPTURE_WINDOW_FRAMES    = 6

# Throughput
INSIDE_PROCESS_EVERY_N   = 1
DETECT_EVERY_N           = 1

# OCR accept rules
PLATE_SUFFIX_RE          = re.compile(r"(\d{2,3})\s*[-−–—ーｰ~〜－]\s*(\d{2})")
MIN_SHARPNESS_LOCK       = 60.0

# Draw/perf
WRITE_VIDEO              = False
DRAW_BBOXES              = True
SHOW_FPS                 = True
TARGET_WIDTH             = 640
BOX_THICK                = 1
ROI_INFER_MAX_W          = 640
GATE_DET_BAND_RATIO      = 0.50

# Reduce OpenCV overhead
cv2.setUseOptimized(True)
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")
os.environ.setdefault("FLAGS_minloglevel", "2")

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
    print(f"[{event_ts}] {direction.upper():3s}  slot={slot_str:>6s}  region_class='{region_class or ''}'  "
          f"suffix='{suffix or ''}'  kana='{kana or ''}'  city='{city or ''}'  raw=\"{raw_text}\"")

# ---------- OCR helpers ----------
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

# ---------- geometry ----------
def xyxy_from_det(det):
    if isinstance(det, (list, tuple)) and len(det) >= 4:
        return float(det[0]), float(det[1]), float(det[2]), float(det[3])
    if isinstance(det, dict):
        box = det.get("bbox") or det.get("xyxy")
        if box and len(box) >= 4: return float(box[0]), float(box[1]), float(box[2]), float(box[3])
    return None

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

# ---------- IDs / slots ----------
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

# ---------- Tiny centroid tracker ----------
class SimpleTracker:
    """
    Greedy nearest-neighbor tracker; keeps one stable oid per car.
    This replaces the old 'stable_key' (which changed while moving).
    """
    def __init__(self, start_id=100, max_distance=70, max_missed=12):
        self.next_id = int(start_id)
        self.max_distance = float(max_distance)
        self.max_missed = int(max_missed)
        self.tracks = {}          # oid -> {'xy':(x,y), 'missed':0, 'last_frame':0,
                                  #        'side':'L'/'R', 'side_streak':0}
    def _new_id(self):
        oid = self.next_id; self.next_id += 1; return oid

    def update(self, detections, frame_idx, gate_x):
        """
        detections: list of dicts {'det':det, 'text':text, 'cx':cx, 'cy':cy}
        returns: list of dicts with 'oid' added
        """
        # mark all unmatched by default
        for tr in self.tracks.values():
            tr['missed'] += 1

        used = set()
        # greedy match by distance
        for d in detections:
            cx, cy = d['cx'], d['cy']
            best_oid, best_dist = None, 1e9
            for oid, tr in self.tracks.items():
                if tr['missed'] > self.max_missed: continue
                tx, ty = tr['xy']
                dist = abs(cx - tx) + abs(cy - ty)  # L1 is fast & ok here
                if dist < best_dist and dist <= self.max_distance and oid not in used:
                    best_dist, best_oid = dist, oid
            if best_oid is None:
                best_oid = self._new_id()
                self.tracks[best_oid] = {'xy':(cx,cy), 'missed':0, 'last_frame':frame_idx,
                                         'side': ('R' if cx >= gate_x else 'L'), 'side_streak':1}
            else:
                tr = self.tracks[best_oid]
                tr['xy'] = (cx, cy)
                tr['missed'] = 0
                tr['last_frame'] = frame_idx
                side_now = 'R' if cx >= gate_x else 'L'
                if side_now == tr['side']:
                    tr['side_streak'] += 1
                else:
                    tr['side'] = side_now
                    tr['side_streak'] = 1
            used.add(best_oid)
            d['oid'] = best_oid

        # drop very stale tracks
        drop = [oid for oid,tr in self.tracks.items() if tr['missed'] > self.max_missed]
        for oid in drop: self.tracks.pop(oid, None)

        return detections

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
        try:
            import yt_dlp
            src = _resolve_youtube_opencv_url(YOUTUBE_URL, max_height=YOUTUBE_MAX_HEIGHT)
        except Exception:
            src = None
        if not src: raise RuntimeError("Failed to resolve YouTube stream.")
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

    # ROI / gate
    if use_roi:
        rx = int(REGION_XYWH_RATIO[0]*W); ry = int(REGION_XYWH_RATIO[1]*H)
        rw = int(REGION_XYWH_RATIO[2]*W); rh = int(REGION_XYWH_RATIO[3]*H)
        rx = max(0, min(rx, W-10)); ry = max(0, min(ry, H-10))
        rw = max(20, min(rw, W-rx)); rh = max(20, min(rh, H-ry))
    else:
        rx, ry, rw, rh = 0, 0, W, H

    # slots/IDs & states
    tracker = SimpleTracker(start_id=START_ID_AT, max_distance=70, max_missed=12)
    slot_pool = SlotPool(SLOT_COUNT)

    plate_pref_slot, active_plate_to_slot, active_id_to_plate = {}, {}, {}
    active_suffix_to_slot = {}
    active_id_to_slot = {}
    active_id_to_suffix = {}

    capture_open = {}    # oid -> frames left (0 close, -1 locked)
    best_score  = {}     # oid -> area*sharpness
    best_text   = {}     # oid -> best OCR so far
    best_det    = {}     # oid -> det for best
    captured_in = set()
    last_event_f = {}

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
                except Exception: pass

            # ROI clamp (if used)
            rx = max(0, min(rx, W-10)); ry = max(0, min(ry, H-10))
            rw = max(20, min(rw, W-rx)); rh = max(20, min(rh, H-ry))

            have_locked_inside = any((oid in captured_in) for oid in captured_in)
            do_heavy_this_frame = True
            if have_locked_inside and (frame_idx % INSIDE_PROCESS_EVERY_N != 0): do_heavy_this_frame = False
            if frame_idx % DETECT_EVERY_N != 0: do_heavy_this_frame = False

            lp_dets_full, lp_texts = [], []

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
                if scale != 1.0:
                    inv_sx = 1.0/scale; inv_sy = 1.0/scale
                    for d in roi_dets:
                        x1,y1,x2,y2 = xyxy_from_det(d); x1*=inv_sx; y1*=inv_sy; x2*=inv_sx; y2*=inv_sy
                        lp_dets_full.append({'xyxy':(x1+d_rx,y1+d_ry,x2+d_rx,y2+d_ry)})
                else:
                    for d in roi_dets:
                        x1,y1,x2,y2 = xyxy_from_det(d)
                        lp_dets_full.append({'xyxy':(x1+d_rx,y1+d_ry,x2+d_rx,y2+d_ry)})

            # draw
            if DRAW_BBOXES and do_heavy_this_frame:
                for d in lp_dets_full:
                    x1,y1,x2,y2 = map(int, d['xyxy'])
                    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),BOX_THICK)
            if DRAW_BBOXES and not use_roi and DRAW_GATE:
                gx = int(ENTRY_LINE_X_RATIO * W)
                cv2.line(frame,(gx,0),(gx,H),(200,200,255),2)
                cv2.putText(frame, "gate", (gx+6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,255), 2, cv2.LINE_AA)

            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            gate_x = int(ENTRY_LINE_X_RATIO * W)

            # build det list with centers + text (paired order)
            det_list = []
            for i, d in enumerate(lp_dets_full):
                cx, cy = center_from_det(d)
                det_list.append({'det': d, 'text': (lp_texts[i] if i < len(lp_texts) else ""), 'cx': cx or 0.0, 'cy': cy or 0.0})

            # track assignment (stable oids)
            det_list = tracker.update(det_list, frame_idx, gate_x)

            # process each tracked plate
            for item in det_list:
                oid = item['oid']; det = item['det']; text = item['text']
                cx, cy = item['cx'], item['cy']

                side_now = 'R' if cx >= gate_x else 'L'
                tr = tracker.tracks.get(oid, None)
                side_streak = tr['side_streak'] if tr else 1

                # Raw OCR debug
                if PRINT_RAW_OCR_FOUND:
                    norm = normalize_ocr_text(text or "")
                    if norm and (re.search(u"[ぁ-ゖァ-ヿ]", norm) or plate_suffix(norm)):
                        print(f"[RAW] oid={oid} text='{norm}'  center=({cx:.1f},{cy:.1f})")

                # ------------------ one-shot OCR capture (IN) ------------------
                # Open when crossing from outside -> inside (right side is inside)
                # OR when approaching the gate from the outside side
                if oid not in captured_in:
                    open_now = False
                    if (side_now == ('R' if gate_right else 'L')) and side_streak >= 1:
                        # just arrived to inside side (we rely on direction later)
                        if capture_open.get(oid, 0) <= 0:
                            open_now = True
                    else:
                        if abs(cx - gate_x) <= GATE_CAPTURE_MARGIN_PX:
                            open_now = True
                    if open_now:
                        capture_open[oid] = CAPTURE_WINDOW_FRAMES
                        best_score[oid] = 0.0; best_text[oid] = ""; best_det[oid] = det

                if oid not in captured_in and capture_open.get(oid, 0) > 0:
                    x1,y1,x2,y2 = map(int, det['xyxy'])
                    x1=max(0,x1); y1=max(0,y1); x2=min(W,x2); y2=min(H,y2)
                    plate_img = frame[y1:y2, x1:x2]
                    min_side = min(y2-y1, x2-x1)
                    if min_side >= 18:
                        sharp = variance_of_laplacian(plate_img)
                        if sharp >= 25.0:
                            score = bbox_area(det) * max(1.0, sharp)
                            if score > best_score.get(oid,0.0) and plate_suffix(text or ""):
                                best_score[oid] = score; best_text[oid] = text; best_det[oid] = det
                            if plate_suffix(text or "") and sharp >= MIN_SHARPNESS_LOCK:
                                capture_open[oid] = 0
                            else:
                                capture_open[oid] -= 1
                        else:
                            capture_open[oid] -= 1
                    else:
                        capture_open[oid] -= 1

                # When window closes, decide direction by last side flip and log IN
                if oid not in captured_in and capture_open.get(oid, 0) == 0:
                    chosen_text = best_text.get(oid,"") or text or ""
                    pk = plate_key(chosen_text)
                    region_class, suf, kana, city, engine_size = parse_plate_fields(chosen_text)

                    # slot reuse if known
                    chosen_slot = None
                    if pk and pk in plate_pref_slot:
                        got = slot_pool.try_acquire_specific(plate_pref_slot[pk])
                        if got is not None: chosen_slot = got
                    if chosen_slot is None:
                        chosen_slot = slot_pool.acquire_lowest()

                    if chosen_slot is None:
                        # lot full: just don't spam
                        capture_open[oid] = -1
                        continue

                    # book-keeping
                    if pk:
                        active_plate_to_slot[pk] = chosen_slot
                        plate_pref_slot[pk] = chosen_slot
                        active_id_to_plate[oid] = pk
                    active_id_to_slot[oid] = chosen_slot
                    if suf:
                        active_suffix_to_slot[suf] = chosen_slot
                        active_id_to_suffix[oid] = suf

                    object_id = f"{chosen_slot:02d}"
                    write_row_flush(csv_writer, csv_file,
                                    [ts, object_id, "car", "in", city or "", engine_size or "", kana or "", suf or ""])
                    captured_in.add(oid); last_event_f[oid] = frame_idx; capture_open[oid] = -1
                    if DRAW_BBOXES:
                        cv2.putText(frame, f"IN {object_id}", (int(cx), int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv2.LINE_AA)
                    log_terminal(ts, "in", f"{object_id}({suf})" if suf else object_id, region_class, suf, kana, city, chosen_text)

                # ------------------ OUT by direction flip ------------------
                # If we have a slot and the side flips to "outside" consistently, write OUT.
                if (oid in captured_in) and (oid in active_id_to_slot):
                    # "outside" is the opposite side of 'inside'
                    outside_side = 'L' if gate_right else 'R'
                    if tracker.tracks[oid]['side'] == outside_side and tracker.tracks[oid]['side_streak'] >= EXIT_STABLE_FRAMES:
                        # prevent double firing
                        if frame_idx - last_event_f.get(oid, -10**9) >= MIN_EVENT_GAP_FRAMES:
                            slot_to_free = active_id_to_slot.pop(oid, None)
                            if slot_to_free is None:
                                pk_now = active_id_to_plate.get(oid)
                                if pk_now and pk_now in active_plate_to_slot:
                                    slot_to_free = active_plate_to_slot.pop(pk_now, None)

                            if slot_to_free is not None:
                                chosen_text = best_text.get(oid,"") or text or ""
                                region_class, suf_out, kana, city, engine_size = parse_plate_fields(chosen_text)
                                object_id = f"{slot_to_free:02d}"
                                write_row_flush(csv_writer, csv_file,
                                                [ts, object_id, "car", "out", city or "", engine_size or "", kana or "", suf_out or ""])
                                slot_pool.release(slot_to_free)
                                last_event_f[oid] = frame_idx
                                if DRAW_BBOXES:
                                    cv2.putText(frame, f"OUT {object_id}", (int(cx), int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv2.LINE_AA)
                                log_terminal(ts, "out", f"{object_id}({suf_out})" if suf_out else object_id,
                                             region_class, suf_out, kana, city, chosen_text)
                                # clear maps
                                if suf_out and active_suffix_to_slot.get(suf_out) == slot_to_free:
                                    active_suffix_to_slot.pop(suf_out, None)
                                active_id_to_plate.pop(oid, None)
                                active_id_to_suffix.pop(oid, None)
                                captured_in.discard(oid)
                                capture_open.pop(oid, None); best_score.pop(oid, None); best_text.pop(oid, None); best_det.pop(oid, None)

            # FPS overlay
            if SHOW_FPS:
                now = time.time()
                if now - t0 >= 0.5:
                    fps_est = fcount / (now - t0); t0 = now; fcount = 0
                cv2.putText(frame, f"FPS ~ {fps_est:.1f}", (10, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
                mode = "ROI" if use_roi else "Gate(center)"
                side = "Right" if gate_right else "Left"
                cv2.putText(frame, f"Mode:{mode} Inside:{side}  [m]mode [o]side", (10, 48),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180,255,180), 2, cv2.LINE_AA)

            cv2.imshow("Parking (direction-based IN/OUT)", frame)
            if out is not None and WRITE_VIDEO: out.write(frame)

            # keyboard
            k = cv2.waitKey(1) & 0xFF
            if k == 27: break
            if k == ord('m'): use_roi = not use_roi
            if k == ord('o'): gate_right = not gate_right

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

# --------- YouTube resolver (kept last to keep the file tidy) ----------
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

if __name__ == "__main__":
    main()
