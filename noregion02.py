# -*- coding: utf-8 -*-
# Realtime-feel plate logger: latest-frame capture + OCR frame-skipping + slot leasing.
# Logs once on ENTER (leases slot 01..10, or recalls previous slot) and once on EXIT (frees).

import os, cv2, csv, re, time, threading
from datetime import datetime
from detections import CarDetection, LicencePlateDetection

# ===================== User settings =====================
USE_CAMERA         = True
INPUT_VIDEO_PATH   = "input_videos/Trim03.mp4"

SLOT_COUNT         = 10
START_ID_AT        = 100

# ROI as ratios (x,y,w,h). Move with WASD, resize with [ ]
REGION_XYWH_RATIO  = (0.20, 0.55, 0.60, 0.30)

ENTER_STABLE_FRAMES = 2
EXIT_STABLE_FRAMES  = 4
MIN_EVENT_GAP_FRAMES= 10

# Performance toggles
USE_CAR_DETECTOR    = False   # car boxes are optional
WRITE_VIDEO         = False   # AVI writing off by default
DRAW_BBOXES         = True
SHOW_FPS            = True
OCR_EVERY           = 3       # run LP OCR every N frames (e.g. 3); reuse between ticks

# =========================================================

# ---------- JP plate parsing ----------
CITY_KANA_RE   = re.compile(u"^\\s*([^\\s\\d]+)\\s+[0-9]{3,4}\\s*([ぁ-ゖァ-ヿ一-龯A-Za-z])", re.UNICODE)
PLATE_SUFFIX_RE= re.compile(r"([0-9]{2,3}-[0-9]{2})")

def parse_city_kana(text):
    if not text: return None, None
    m = CITY_KANA_RE.search(text)
    return (m.group(1), m.group(2)) if m else (None, None)

def plate_suffix(text):
    if not text: return None
    m = PLATE_SUFFIX_RE.search(text)
    return m.group(1) if m else None

def plate_key(text):
    suf = plate_suffix(text or "")
    if not suf: return None
    city, _ = parse_city_kana(text or "")
    return f"{city if city else '?'}|{suf}"

# ---------- CSV ----------
def open_daily_csv():
    date_str = datetime.now().strftime("%Y%m%d")
    os.makedirs("logs", exist_ok=True)
    path = os.path.join("logs", f"parking_log_{date_str}.csv")
    new = not os.path.exists(path)
    f = open(path, "a", newline="", encoding="utf-8")
    w = csv.writer(f)
    if new:
        w.writerow(["timestamp","object_id","vehicle_type","direction","city","kana"])
        f.flush()
    return f, w, path

# ---------- Geometry ----------
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

# ---------- ID bank ----------
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

# ---------- Slot pool ----------
class SlotPool:
    def __init__(self, n):
        self.free = list(range(1, n+1)); self.used=set()
    def acquire_lowest(self):
        if not self.free: return None
        s = self.free.pop(0); self.used.add(s); return s
    def try_acquire_specific(self, s):
        if s in self.free:
            self.free.remove(s); self.used.add(s); return s
        return None
    def release(self, s):
        if s in self.used:
            self.used.remove(s); self.free.append(s); self.free.sort()

# ---------- Region helpers ----------
def inside_region(cx, cy, rx, ry, rw, rh):
    return (cx is not None) and (cy is not None) and (rx <= cx <= rx+rw) and (ry <= cy <= ry+rh)

def clamp_region(rx, ry, rw, rh, W, H):
    rx = max(0, min(rx, W-10)); ry = max(0, min(ry, H-10))
    rw = max(20, min(rw, W-rx)); rh = max(20, min(rh, H-ry))
    return rx, ry, rw, rh

def draw_region(frame, rx, ry, rw, rh):
    cv2.rectangle(frame, (rx, ry), (rx+rw, ry+rh), (0,255,255), 2)
    cv2.putText(frame, "ROI: WASD move  [ ] resize", (rx+6, max(18, ry-6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255), 2, cv2.LINE_AA)

# ---------- Latest-frame capture thread ----------
class FrameGrabber:
    def __init__(self, src, width=640, height=384):
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video source: {src}")
        # small frame for speed
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.lock = threading.Lock()
        self.latest = None
        self.stopped = False
        self.t = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while not self.stopped:
            ok = self.cap.grab()
            if not ok: time.sleep(0.005); continue
            ok, frame = self.cap.retrieve()
            if not ok: continue
            with self.lock:
                self.latest = frame

    def start(self):
        self.t.start()
        # wait for first frame
        for _ in range(100):
            with self.lock:
                if self.latest is not None: return
            time.sleep(0.01)
        raise RuntimeError("No frames from source")

    def read(self):
        with self.lock:
            if self.latest is None: return None
            return self.latest.copy()

    def release(self):
        self.stopped = True
        try:
            self.t.join(timeout=0.2)
        except: pass
        try:
            self.cap.release()
        except: pass

def main():
    # Speed tweak: avoid OpenCV oversubscription
    try: cv2.setNumThreads(1)
    except: pass

    src = 0 if USE_CAMERA else INPUT_VIDEO_PATH
    grab = FrameGrabber(src)
    grab.start()

    # Get frame size
    first = grab.read()
    H, W = first.shape[:2]

    # Detectors
    car_det = CarDetection(model_path="yolo11n.pt") if USE_CAR_DETECTOR else None
    lp_det  = LicencePlateDetection(model_path="models/best.pt")

    # Video writer (optional)
    os.makedirs("output_videos", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    fps = 20.0
    out = cv2.VideoWriter("output_videos/output_video.avi", fourcc, fps, (W, H)) if WRITE_VIDEO else None

    # CSV
    csv_file, csv_writer, csv_path = open_daily_csv()
    print("[INFO] Logging to:", csv_path)

    # ROI
    rx = int(REGION_XYWH_RATIO[0]*W); ry = int(REGION_XYWH_RATIO[1]*H)
    rw = int(REGION_XYWH_RATIO[2]*W); rh = int(REGION_XYWH_RATIO[3]*H)
    rx, ry, rw, rh = clamp_region(rx, ry, rw, rh, W, H)

    # State
    idbank = IDBank(START_ID_AT)
    prev_inside = {}
    inside_count, outside_count, last_event_f = {}, {}, {}
    slot_pool = SlotPool(SLOT_COUNT)
    plate_pref_slot, active_plate_to_slot, active_id_to_plate = {}, {}, {}

    frame_idx = 0
    last_lp_dets, last_lp_texts = [], []

    t0 = time.time(); f_counter = 0; fps_est = 0.0

    try:
        while True:
            frame = grab.read()
            if frame is None:
                time.sleep(0.005)
                continue

            frame_idx += 1; f_counter += 1
            run_ocr = (frame_idx % OCR_EVERY == 1)

            # plate detection (latest frame only)
            if run_ocr:
                all_lp_dets, all_lp_texts = lp_det.detect_frames([frame])
                last_lp_dets, last_lp_texts = all_lp_dets[0], all_lp_texts[0]
            lp_dets, lp_texts = last_lp_dets, last_lp_texts

            # optional car det
            car_dets = []
            if USE_CAR_DETECTOR and run_ocr:
                car_dets = car_det.detect_frames([frame], read_from_stub=False)[0]

            if DRAW_BBOXES:
                if USE_CAR_DETECTOR and car_dets:
                    frame = car_det.draw_bboxes([frame], [car_dets])[0]
                if lp_dets:
                    frame = lp_det.draw_bboxes([frame], [lp_dets], [lp_texts])[0]
                draw_region(frame, rx, ry, rw, rh)

            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # events
            for det, text in zip(lp_dets, lp_texts):
                key = stable_key(text, det)
                if key is None: continue
                oid = idbank.get(key)

                cx, cy = center_from_det(det)
                is_in = inside_region(cx, cy, rx, ry, rw, rh)

                if oid not in prev_inside:
                    prev_inside[oid] = is_in
                    inside_count[oid]  = 1 if is_in else 0
                    outside_count[oid] = 0 if is_in else 1
                    continue

                if is_in:
                    inside_count[oid]  = inside_count.get(oid,0) + 1
                    outside_count[oid] = 0
                else:
                    outside_count[oid] = outside_count.get(oid,0) + 1
                    inside_count[oid]  = 0

                if frame_idx - last_event_f.get(oid, -10**9) < MIN_EVENT_GAP_FRAMES:
                    prev_inside[oid] = is_in
                    continue

                pk   = plate_key(text or "")
                suf  = plate_suffix(text or "")
                city, kana = parse_city_kana(text or "")

                # ENTER
                if (not prev_inside[oid]) and is_in and inside_count[oid] >= ENTER_STABLE_FRAMES:
                    chosen = None
                    if pk and pk in plate_pref_slot:
                        got = slot_pool.try_acquire_specific(plate_pref_slot[pk])
                        if got is not None: chosen = got
                    if chosen is None: chosen = slot_pool.acquire_lowest()
                    if chosen is not None:
                        if pk:
                            active_plate_to_slot[pk] = chosen
                            plate_pref_slot[pk] = chosen
                            active_id_to_plate[oid] = pk
                        else:
                            active_id_to_plate[oid] = None
                        object_id = f"{chosen:02d}({suf})" if suf else f"{chosen:02d}"
                        csv_writer.writerow([ts, object_id, "car", "in", city or "", kana or ""])
                        csv_file.flush()
                        last_event_f[oid] = frame_idx
                        if DRAW_BBOXES:
                            cv2.putText(frame, f"IN {object_id}", (int(cx), int(cy)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)

                # EXIT
                if prev_inside[oid] and (not is_in) and outside_count[oid] >= EXIT_STABLE_FRAMES:
                    slot_to_free = None
                    if pk and pk in active_plate_to_slot:
                        slot_to_free = active_plate_to_slot.pop(pk, None)
                    else:
                        pk2 = active_id_to_plate.get(oid)
                        if pk2 and pk2 in active_plate_to_slot:
                            slot_to_free = active_plate_to_slot.pop(pk2, None)

                    if slot_to_free is not None:
                        object_id = f"{slot_to_free:02d}({suf})" if suf else f"{slot_to_free:02d}"
                        csv_writer.writerow([ts, object_id, "car", "out", city or "", kana or ""])
                        csv_file.flush()
                        last_event_f[oid] = frame_idx
                        slot_pool.release(slot_to_free)
                        if DRAW_BBOXES:
                            cv2.putText(frame, f"OUT {object_id}", (int(cx), int(cy)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2, cv2.LINE_AA)

                prev_inside[oid] = is_in

            # FPS (viewer stays near realtime because we never backlog frames)
            if SHOW_FPS:
                now = time.time()
                if now - t0 >= 0.5:
                    fps_est = f_counter / (now - t0)
                    t0 = now; f_counter = 0
                cv2.putText(frame, f"FPS(view) ~ {fps_est:.1f} | OCR x{OCR_EVERY}",
                            (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

            # Display / write
            cv2.imshow("Parking (latest-frame, OCR-skip)", frame)
            if WRITE_VIDEO and out is not None: out.write(frame)

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
        grab.release()
        try:
            if WRITE_VIDEO and out is not None: out.release()
        except: pass
        cv2.destroyAllWindows()
        csv_file.close()
        print("[INFO] CSV saved to", csv_path)
        if WRITE_VIDEO: print("[INFO] Video saved to output_videos/output_video.avi")

if __name__ == "__main__":
    main()
