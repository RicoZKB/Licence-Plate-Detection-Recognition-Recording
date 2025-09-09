# -*- coding: utf-8 -*-
# region05.py — Region logger: one-shot OCR per entry + suspend OCR while inside
# - Logs once on ENTER (leases slot 01..10, or recalls previous slot)
# - Logs once on EXIT (frees slot)
# - Picks "best" entry frame by bbox_area × sharpness (Laplacian)
# - After logging IN, HALTS heavy detection/OCR until the car exits (photo-capture behavior)
# - CSV rows are flushed immediately

import os
# Quieter backends (set BEFORE other imports that may init loggers)
os.environ.setdefault("FLAGS_minloglevel", "2")
os.environ.setdefault("GLOG_minloglevel", "2")

import cv2, csv, re, time, io
from contextlib import redirect_stdout
from datetime import datetime
from detections import CarDetection, LicencePlateDetection

# ===================== User settings =====================
USE_CAMERA            = False
INPUT_VIDEO_PATH      = "input_videos/Trim03.mp4"

SLOT_COUNT            = 10
START_ID_AT           = 100

# Smaller ROI (x, y, w, h) as ratios of frame; move with WASD, resize with [ ]
REGION_XYWH_RATIO     = (0.32, 0.60, 0.36, 0.22)

# Stability & gaps
ENTER_STABLE_FRAMES   = 2
EXIT_STABLE_FRAMES    = 4
MIN_EVENT_GAP_FRAMES  = 10

# One-shot capture window (frames after first entry to choose best)
CAPTURE_WINDOW_FRAMES = 8

# After an IN is logged, STOP heavy detection/OCR while the car stays inside
SUSPEND_WHEN_CAPTURED = True

# Optional global detector skip (set to 1 for "every frame"; 2 = every 2nd frame, etc.)
DETECT_GLOBAL_EVERY   = 1

# Performance toggles
USE_CAR_DETECTOR      = False     # more load if True
WRITE_VIDEO           = False
DRAW_BBOXES           = True
SHOW_FPS              = True
TARGET_WIDTH          = 640       # keep frames small

# Silence noisy OCR debug prints coming from python-level prints
SILENCE_OCR_DEBUG     = True
# =========================================================

cv2.setUseOptimized(True)

CITY_KANA_RE = re.compile(u"^\\s*([^\\s\\d]+)\\s+[0-9]{3,4}\\s*([ぁ-ゖァ-ヿ一-龯A-Za-z])", re.UNICODE)
PLATE_SUFFIX_RE = re.compile(r"([0-9]{2,3}-[0-9]{2})")

def parse_city_kana(text):
    if not text: return None, None
    m = CITY_KANA_RE.search(text);  return (m.group(1), m.group(2)) if m else (None, None)

def plate_suffix(text):
    if not text: return None
    m = PLATE_SUFFIX_RE.search(text);  return m.group(1) if m else None

def plate_key(text):
    suf = plate_suffix(text or "");  city, _ = parse_city_kana(text or "")
    if not suf: return None
    return f"{city if city else '?'}|{suf}"

def open_daily_csv():
    date_str = datetime.now().strftime("%Y%m%d")
    os.makedirs("logs", exist_ok=True)
    path = os.path.join("logs", f"parking_log_{date_str}.csv")
    new = not os.path.exists(path)
    f = open(path, "a", newline="", encoding="utf-8", buffering=1)  # line-buffered
    w = csv.writer(f)
    if new:
        w.writerow(["timestamp","object_id","vehicle_type","direction","city","kana"])
        f.flush()
    return f, w, path

def write_row_flush(w, f, row):
    w.writerow(row); f.flush()
    try: os.fsync(f.fileno())
    except Exception: pass

def xyxy_from_det(det):
    if isinstance(det, (list, tuple)) and len(det) >= 4:
        return float(det[0]), float(det[1]), float(det[2]), float(det[3])
    if isinstance(det, dict):
        box = det.get("bbox") or det.get("xyxy")
        if box and len(box) >= 4:
            return float(box[0]), float(box[1]), float(box[2]), float(box[3])
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

def crop(frame, det):
    xy = xyxy_from_det(det)
    if not xy: return None
    x1,y1,x2,y2 = map(int, xy)
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(frame.shape[1]-1, x2); y2 = min(frame.shape[0]-1, y2)
    return frame[y1:y2, x1:x2]

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

def inside_region(cx, cy, rx, ry, rw, rh):
    return (cx is not None) and (cy is not None) and (rx <= cx <= rx+rw) and (ry <= cy <= ry+rh)

def clamp_region(rx, ry, rw, rh, W, H):
    rx = max(0, min(rx, W-10)); ry = max(0, min(ry, H-10))
    rw = max(20, min(rw, W-rx)); rh = max(20, min(rh, H-ry))
    return rx, ry, rw, rh

def draw_region(frame, rx, ry, rw, rh):
    cv2.rectangle(frame, (rx, ry), (rx+rw, ry+rh), (0,255,255), 1)  # thinner line
    cv2.putText(frame, "ROI: WASD move, [ ] resize, r reset",
                (rx+6, max(18, ry-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255), 1, cv2.LINE_AA)

def silent_detect(lp_det, frames):
    """Call lp_det.detect_frames(frames) while swallowing python-level prints."""
    if not SILENCE_OCR_DEBUG:
        return lp_det.detect_frames(frames)
    buf = io.StringIO()
    with redirect_stdout(buf):
        return lp_det.detect_frames(frames)

def main():
    # Source
    src = 0 if USE_CAMERA else INPUT_VIDEO_PATH
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {src}")

    # Keep it small for speed
    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)
        if USE_CAMERA:
            # live-cam niceties
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except: pass

    # Detectors
    car_det = CarDetection(model_path="yolo11n.pt") if USE_CAR_DETECTOR else None
    lp_det  = LicencePlateDetection(model_path="models/best.pt")

    # IO
    os.makedirs("output_videos", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or TARGET_WIDTH)
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or int(TARGET_WIDTH*3/5))
    out = cv2.VideoWriter("output_videos/output_video.avi", fourcc, fps, (W, H)) if WRITE_VIDEO else None

    csv_file, csv_writer, csv_path = open_daily_csv()
    print("[INFO] Logging to:", csv_path)

    # Region
    rx = int(REGION_XYWH_RATIO[0]*W); ry = int(REGION_XYWH_RATIO[1]*H)
    rw = int(REGION_XYWH_RATIO[2]*W); rh = int(REGION_XYWH_RATIO[3]*H)
    rx, ry, rw, rh = clamp_region(rx, ry, rw, rh, W, H)

    # State
    idbank = IDBank(START_ID_AT)
    prev_inside = {}
    inside_count, outside_count, last_event_f = {}, {}, {}

    slot_pool = SlotPool(SLOT_COUNT)
    plate_pref_slot = {}         # pk -> preferred slot (last used)
    active_plate_to_slot = {}    # pk -> slot currently in use
    active_id_to_plate = {}      # oid -> pk

    # One-shot capture bookkeeping per oid
    capture_open = {}            # oid -> frames left in capture window
    best_score = {}              # oid -> best area*sharpness
    best_text  = {}              # oid -> best OCR text so far
    best_det   = {}              # oid -> det for best text
    captured_in = set()          # oids that already logged IN

    frame_idx = 0
    t0 = time.time(); fcount = 0; fps_est = 0.0

    try:
        while True:
            ok = cap.grab() if USE_CAMERA else True
            if USE_CAMERA and not ok:
                break
            if USE_CAMERA:
                ok, frame = cap.retrieve()
            else:
                ok, frame = cap.read()
                if not ok: break

            frame_idx += 1; fcount += 1

            # Decide if we run heavy detection/OCR this frame
            have_inside_captured = any(oid in captured_in and prev_inside.get(oid, False)
                                       for oid in prev_inside.keys())
            if SUSPEND_WHEN_CAPTURED and have_inside_captured:
                do_heavy_this_frame = False
            else:
                do_heavy_this_frame = True
                if DETECT_GLOBAL_EVERY > 1 and (frame_idx % DETECT_GLOBAL_EVERY != 0):
                    do_heavy_this_frame = False

            # Run LP detection (and OCR) only if needed this frame
            lp_dets, lp_texts = [], []
            if do_heavy_this_frame:
                all_lp_dets, all_lp_texts = silent_detect(lp_det, [frame])
                lp_dets, lp_texts = all_lp_dets[0], all_lp_texts[0]

            # Optional cars
            car_dets = []
            if USE_CAR_DETECTOR and do_heavy_this_frame:
                car_dets = car_det.detect_frames([frame], read_from_stub=False)[0]

            # Draw overlays
            if DRAW_BBOXES and do_heavy_this_frame:
                if USE_CAR_DETECTOR:
                    frame = car_det.draw_bboxes([frame], [car_dets])[0]
                frame = lp_det.draw_bboxes([frame], [lp_dets], [lp_texts])[0]
            if DRAW_BBOXES:
                draw_region(frame, rx, ry, rw, rh)

            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Process detections (only when we actually ran them)
            if do_heavy_this_frame:
                for det, text in zip(lp_dets, lp_texts):
                    key = stable_key(text, det)
                    if key is None: continue
                    oid = idbank.get(key)
                    cx, cy = center_from_det(det)
                    is_in = inside_region(cx, cy, rx, ry, rw, rh)

                    # init counters
                    if oid not in prev_inside:
                        prev_inside[oid] = is_in
                        inside_count[oid]  = 1 if is_in else 0
                        outside_count[oid] = 0 if is_in else 1
                        if is_in:
                            capture_open[oid] = CAPTURE_WINDOW_FRAMES
                            best_score[oid] = 0.0; best_text[oid] = ""; best_det[oid] = det
                        continue

                    # inside/outside streak counters
                    if is_in:
                        inside_count[oid]  = inside_count.get(oid,0) + 1
                        outside_count[oid] = 0
                    else:
                        outside_count[oid] = outside_count.get(oid,0) + 1
                        inside_count[oid]  = 0

                    # maintain capture window while inside and not yet captured
                    if is_in and (oid not in captured_in):
                        if capture_open.get(oid, 0) > 0:
                            # choose best frame by area * sharpness
                            plate_img = crop(frame, det)
                            score = bbox_area(det) * max(1.0, variance_of_laplacian(plate_img))
                            if score > best_score.get(oid, 0.0) and plate_suffix(text or ""):
                                best_score[oid] = score
                                best_text[oid]  = text
                                best_det[oid]   = det
                            capture_open[oid] -= 1

                        # When window closes (or we’ve seen enough) log IN once
                        if capture_open.get(oid, 0) == 0 and inside_count[oid] >= ENTER_STABLE_FRAMES:
                            pk = plate_key(best_text.get(oid,"") or text or "")
                            suf = plate_suffix(best_text.get(oid,"") or text or "")
                            city, kana = parse_city_kana(best_text.get(oid,"") or text or "")

                            chosen = None
                            if pk and pk in plate_pref_slot:
                                got = slot_pool.try_acquire_specific(plate_pref_slot[pk])
                                if got is not None: chosen = got
                            if chosen is None:
                                chosen = slot_pool.acquire_lowest()

                            if chosen is not None:
                                if pk:
                                    active_plate_to_slot[pk] = chosen
                                    plate_pref_slot[pk] = chosen
                                    active_id_to_plate[oid] = pk
                                else:
                                    active_id_to_plate[oid] = None

                                object_id = f"{chosen:02d}({suf})" if suf else f"{chosen:02d}"
                                write_row_flush(csv_writer, csv_file, [ts, object_id, "car", "in", city or "", kana or ""])
                                captured_in.add(oid)  # one-shot lock
                                last_event_f[oid] = frame_idx
                                if DRAW_BBOXES:
                                    cv2.putText(frame, f"IN {object_id}", (int(cx), int(cy)),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv2.LINE_AA)

                    # Exit handling
                    if prev_inside[oid] and (not is_in) and outside_count[oid] >= EXIT_STABLE_FRAMES:
                        if frame_idx - last_event_f.get(oid, -10**9) >= MIN_EVENT_GAP_FRAMES:
                            pk_now = active_id_to_plate.get(oid) or plate_key(best_text.get(oid,"") or text or "")
                            slot_to_free = None
                            if pk_now and pk_now in active_plate_to_slot:
                                slot_to_free = active_plate_to_slot.pop(pk_now, None)

                            # If we know slot, log OUT
                            if slot_to_free is not None:
                                suf_out = plate_suffix(best_text.get(oid,"") or text or "")
                                city, kana = parse_city_kana(best_text.get(oid,"") or text or "")
                                object_id = f"{slot_to_free:02d}({suf_out})" if suf_out else f"{slot_to_free:02d}"
                                write_row_flush(csv_writer, csv_file, [ts, object_id, "car", "out", city or "", kana or ""])
                                slot_pool.release(slot_to_free)
                                last_event_f[oid] = frame_idx
                                if DRAW_BBOXES:
                                    cv2.putText(frame, f"OUT {object_id}", (int(cx), int(cy)),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv2.LINE_AA)

                            # reset one-shot state so a future entry can capture again
                            capture_open.pop(oid, None)
                            best_score.pop(oid, None); best_text.pop(oid, None); best_det.pop(oid, None)
                            if oid in captured_in: captured_in.remove(oid)

                    prev_inside[oid] = is_in

            # FPS overlay
            if SHOW_FPS:
                now = time.time()
                if now - t0 >= 0.5:
                    fps_est = fcount / (now - t0); t0 = now; fcount = 0
                tag = "" if do_heavy_this_frame else "  (suspended)"
                cv2.putText(frame, f"FPS ~ {fps_est:.1f}{tag}", (10, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

            # Display / write
            cv2.imshow("Parking (region05 one-shot)", frame)
            if out is not None and WRITE_VIDEO: out.write(frame)

            # ROI controls
            k = cv2.waitKey(1) & 0xFF
            if k == 27: break
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
