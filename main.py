# -*- coding: utf-8 -*-
# One-entry/one-exit CSV logger with "slot leasing" (01..10) — SAFE FIX
# - Keeps your original behavior (preview + output AVI)
# - Single VideoCapture (no threading, no second handle)
# - Light speed-up: PROCESS_EVERY_N frames for detectors

import os, csv, re, cv2, collections, time
from datetime import datetime
from detections import CarDetection, LicencePlateDetection

# ===================== USER SETTINGS =====================
USE_CAMERA = False
INPUT_VIDEO_PATH = "input_videos/Trim20.mp4"

SLOT_COUNT = 10
START_ID_AT = 100
ID_BY = "bbox"                # "bbox" or "text"

AREA_WINDOW = 8
MIN_REL_GROW = 0.15
MIN_REL_SHRINK = -0.12
MISS_GRACE_FRAMES = 10

# ---- Light speed knob (keep small to preserve accuracy) ----
PROCESS_EVERY_N = 2           # 1 = run every frame (exactly like original)
# ============================================================

# ---------- CSV ----------
def open_daily_csv():
    date_str = datetime.now().strftime("%Y%m%d")
    os.makedirs("logs", exist_ok=True)
    path = os.path.join("logs", "parking_log_%s.csv" % date_str)
    new = not os.path.exists(path)
    f = open(path, "a", newline="", encoding="utf-8")
    w = csv.writer(f)
    if new:
        w.writerow(["timestamp","object_id","vehicle_type","direction","city","engine_size","kana"])
    return f, w, path

# ---------- Simple ID bank ----------
class IDBank:
    def __init__(self, start_at=100):
        self._map = {}
        self._next = int(start_at)
    def get(self, key):
        if key not in self._map:
            self._map[key] = self._next
            self._next += 1
        return self._map[key]

# ---------- SLOT POOL ----------
class SlotPool:
    def __init__(self, n):
        self.free = list(range(1, n+1))
        self.used = set()
    def acquire(self):
        if not self.free: return None
        s = self.free.pop(0)
        self.used.add(s)
        return s
    def release(self, s):
        if s in self.used:
            self.used.remove(s)
            self.free.append(s)
            self.free.sort()

# ---------- JP plate parsing ----------
CITY_KANA_RE = re.compile(r"^\s*([^\s\d]+)\s+[0-9]{3,4}\s*([ぁ-ゖァ-ヿ一-龯A-Za-z])")
ENGINE_RE    = re.compile(r"([0-9]{3,4})")
NUM_SUFFIX_RE= re.compile(r"([0-9]{2,3}-[0-9]{2})")

def parse_city_kana(text):
    if not text: return (None, None)
    m = CITY_KANA_RE.search(text)
    return (m.group(1), m.group(2)) if m else (None, None)

def parse_engine_size(text):
    if not text: return None
    m = ENGINE_RE.search(text)
    if not m: return None
    digits = m.group(1)
    return digits[0] if digits else None

def parse_number_suffix(text):
    if not text: return None
    m = NUM_SUFFIX_RE.search(text)
    return m.group(1) if m else None

# ---------- Geometry ----------
def xyxy_from_det(det):
    if isinstance(det, (list, tuple)) and len(det) >= 4:
        return float(det[0]), float(det[1]), float(det[2]), float(det[3])
    if isinstance(det, dict):
        box = det.get("bbox") or det.get("xyxy")
        if box and len(box) >= 4:
            return float(box[0]), float(box[1]), float(box[2]), float(box[3])
    return None

def area_from_det(det):
    xy = xyxy_from_det(det)
    if not xy: return None
    x1,y1,x2,y2 = xy
    return max(0.0, (x2-x1)) * max(0.0, (y2-y1))

# ---------- Keys ----------
def key_text(text):
    if not text: return None
    t = re.sub(r"[^\wぁ-ゖァ-ヿ一-龯\-]+", "", text)
    return t if len(t) >= 3 else None

def key_bbox(det):
    xy = xyxy_from_det(det)
    if not xy: return None
    x1,y1,x2,y2 = xy
    cx = (x1+x2)/2.0; cy = (y1+y2)/2.0
    return "bb-%d-%d" % (int(cx//20), int(cy//20))

def make_key(text, det):
    if ID_BY == "text":
        return key_text(text) or key_bbox(det)
    return key_bbox(det) or key_text(text)

# ---------- Trend ----------
def rel_change(seq):
    if len(seq) < 3: return 0.0
    a0, a1 = seq[0], seq[-1]
    if a0 <= 0: return 0.0
    return (a1 - a0) / max(a0, 1e-6)

# ---------- Track ----------
class Track:
    def __init__(self, oid):
        self.oid = oid
        self.slot = None
        self.best_text = ""
        self.areas = collections.deque(maxlen=AREA_WINDOW)
        self.last_seen_f = 0
        self.state = "new"
    def update_text(self, t):
        if not t: return
        cand_len = len(t)
        cur_len  = len(self.best_text)
        if (cand_len > cur_len) or (cand_len == cur_len and parse_number_suffix(t) and not parse_number_suffix(self.best_text)):
            self.best_text = t
    def observe(self, area, frame_idx):
        if area is not None:
            self.areas.append(area)
        self.last_seen_f = frame_idx
    def decide_in(self):
        return rel_change(list(self.areas)) >= MIN_REL_GROW
    def decide_out(self):
        return rel_change(list(self.areas)) <= MIN_REL_SHRINK

def main():
    # ---- single capture (no second handle) ----
    src = 0 if USE_CAMERA else INPUT_VIDEO_PATH
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera/video source: %s" % str(src))

    # ---- output video ----
    os.makedirs("output_videos", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1: fps = 20.0
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    out = cv2.VideoWriter("output_videos/output_video.avi", fourcc, fps, (w, h))

    # ---- CSV ----
    csv_file, csv_writer, csv_path = open_daily_csv()
    print("[INFO] Logging to: %s" % csv_path)

    # ---- detectors ----
    car_det = CarDetection(model_path="yolo11n.pt")
    lp_det  = LicencePlateDetection(model_path="models/best.pt")

    # ---- state ----
    idbank = IDBank(START_ID_AT)
    tracks = {}
    slot_pool = SlotPool(SLOT_COUNT)
    frame_idx = 0

    # cache last detections to reuse on skipped frames
    last_car_dets, last_lp_dets, last_lp_texts = [], [], []

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1

            run_detectors = (frame_idx % PROCESS_EVERY_N == 1)

            if run_detectors:
                car_dets = car_det.detect_frames([frame], read_from_stub=False)[0]
                if len(car_dets) == 0:
                    lp_dets, lp_texts = [], []
                else:
                    all_lp_dets, all_lp_texts = lp_det.detect_frames([frame])
                    lp_dets, lp_texts = all_lp_dets[0], all_lp_texts[0]
                last_car_dets, last_lp_dets, last_lp_texts = car_dets, lp_dets, lp_texts
            else:
                car_dets, lp_dets, lp_texts = last_car_dets, last_lp_dets, last_lp_texts

            # Draw overlays (exactly like your original)
            disp = car_det.draw_bboxes([frame], [car_dets])[0]
            disp = lp_det.draw_bboxes([disp], [lp_dets], [lp_texts])[0]

            # --- tracking / slot logic (only if LPs exist) ---
            for det, text in zip(lp_dets, lp_texts):
                key = make_key(text, det)
                if key is None: 
                    continue
                oid = idbank.get(key)
                tr = tracks.get(oid)
                if tr is None:
                    tr = Track(oid)
                    tracks[oid] = tr
                tr.update_text(text or "")
                area = area_from_det(det)
                tr.observe(area, frame_idx)

                if tr.slot is None and tr.decide_in():
                    slot = slot_pool.acquire()
                    if slot is not None:
                        tr.slot = slot
                        tr.state = "in"
                        suffix = parse_number_suffix(tr.best_text) or ""
                        object_id = "%02d(%s)" % (slot, suffix) if suffix else "%02d" % slot
                        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        city, kana = parse_city_kana(tr.best_text)
                        engine = parse_engine_size(tr.best_text)
                        csv_writer.writerow([ts, object_id, "car", "in", city or "", engine or "", kana or ""])
                        print("[IN ] %s object_id=%s text='%s'" % (ts, object_id, tr.best_text))

                if tr.slot is not None and tr.state == "in" and tr.decide_out():
                    tr.state = "leaving"

            # finalize outs
            for oid, tr in list(tracks.items()):
                if tr.slot is None:
                    continue
                missed = frame_idx - tr.last_seen_f
                if tr.state == "leaving" and missed >= MISS_GRACE_FRAMES:
                    suffix = parse_number_suffix(tr.best_text) or ""
                    object_id = "%02d(%s)" % (tr.slot, suffix) if suffix else "%02d" % tr.slot
                    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    city, kana = parse_city_kana(tr.best_text)
                    engine = parse_engine_size(tr.best_text)
                    csv_writer.writerow([ts, object_id, "car", "out", city or "", engine or "", kana or ""])
                    print("[OUT] %s object_id=%s text='%s'" % (ts, object_id, tr.best_text))
                    slot_pool.release(tr.slot)
                    del tracks[oid]

            # small on-screen slot labels
            y = 20
            for oid, tr in tracks.items():
                if tr.slot is None: 
                    continue
                cv2.putText(disp, "%02d" % tr.slot, (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2, cv2.LINE_AA)
                y += 20

            # show & save
            cv2.imshow("Parking (slot leasing) [safe]", disp)
            out.write(disp)

            # file vs camera pacing
            if USE_CAMERA:
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            else:
                # keep UI responsive but don't throttle too much
                if cv2.waitKey(1) & 0xFF == 27:
                    break

    except KeyboardInterrupt:
        print("[INFO] Interrupted by user.")
    finally:
        try: cap.release()
        except: pass
        try: out.release()
        except: pass
        cv2.destroyAllWindows()
        csv_file.close()
        print("[INFO] Video saved to output_videos/output_video.avi")
        print("[INFO] CSV saved to %s" % csv_path)

if __name__ == "__main__":
    main()