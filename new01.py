# -*- coding: utf-8 -*-
# One-entry/one-exit CSV logger with "slot leasing" (01..10).
# - Logs exactly once on 'in' (approach) and once on 'out' (leave+disappear)
# - object_id format: "SS(nn-nn)" e.g. "01(70-50)"
# - Frees slot on 'out' so it can be reused.
#
# CSV columns: timestamp,object_id,vehicle_type,direction,city,engine_size,kana
# Video saved as: output_videos/output_video.avi (MJPG/AVI)
#
# Requires your CarDetection and LicencePlateDetection classes.

import os, csv, re, cv2, collections, time
from datetime import datetime
from detections import CarDetection, LicencePlateDetection

# ===================== USER SETTINGS =====================
USE_CAMERA = False                         # True: webcam(0), False: use INPUT_VIDEO_PATH
INPUT_VIDEO_PATH = "input_videos/Trim20.mp4"  # used when USE_CAMERA=False

SLOT_COUNT = 10               # number of "lease IDs": 01..SLOT_COUNT
START_ID_AT = 100             # internal monotonic ID seed (not shown in CSV)
ID_BY = "bbox"                # "bbox" (robust) or "text" for tracking key

AREA_WINDOW = 8               # frames used to judge growth/shrink
MIN_REL_GROW = 0.15           # >= +15% across window => approaching
MIN_REL_SHRINK = -0.12        # <= -12% across window => leaving
MISS_GRACE_FRAMES = 10        # how many frames absent before we finalize OUT
# ==========================================================

# ---------- CSV ----------
def open_daily_csv():
    date_str = datetime.now().strftime("%Y%m%d")
    os.makedirs("logs", exist_ok=True)
    path = os.path.join("logs", f"parking_log_{date_str}.csv")
    new = not os.path.exists(path)
    f = open(path, "a", newline="", encoding="utf-8")
    w = csv.writer(f)
    if new:
        w.writerow(["timestamp","object_id","vehicle_type","direction","city","engine_size","kana"])
    return f, w, path

# ---------- Simple ID bank for internal objects ----------
class IDBank:
    def __init__(self, start_at=100):
        self._map = {}
        self._next = int(start_at)
    def get(self, key):
        if key not in self._map:
            self._map[key] = self._next
            self._next += 1
        return self._map[key]

# ---------- SLOT POOL (01..SLOT_COUNT) ----------
class SlotPool:
    def __init__(self, n):
        self.free = list(range(1, n+1))   # 1..n
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
# text samples: '世田谷310  わ 70-50', '神戸 500 わ 12-34'
CITY_KANA_RE = re.compile(r"^\s*([^\s\d]+)\s+[0-9]{3,4}\s*([ぁ-ゖァ-ヿ一-龯A-Za-z])")
ENGINE_RE    = re.compile(r"([0-9]{3,4})")            # pick first 3/4-digit number -> first digit is engine size
NUM_SUFFIX_RE= re.compile(r"([0-9]{2,3}-[0-9]{2})")   # 70-50, 123-45, etc.

def parse_city_kana(text):
    if not text: return None, None
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

# ---------- Geometry / areas ----------
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

def area_from_det(det):
    xy = xyxy_from_det(det)
    if not xy: return None
    x1,y1,x2,y2 = xy
    return max(0.0, (x2-x1)) * max(0.0, (y2-y1))

# ---------- Stable keys for tracking ----------
def key_text(text):
    if not text: return None
    t = re.sub(r"[^\wぁ-ゖァ-ヿ一-龯\-]+", "", text)
    return t if len(t) >= 3 else None

def key_bbox(det):
    xy = xyxy_from_det(det)
    if not xy: return None
    x1,y1,x2,y2 = xy
    cx = (x1+x2)/2.0; cy = (y1+y2)/2.0
    return f"bb-{int(cx//20)}-{int(cy//20)}"  # quantize center to 20px buckets

def make_key(text, det):
    if ID_BY == "text":
        return key_text(text) or key_bbox(det)
    return key_bbox(det) or key_text(text)

# ---------- Trend helpers ----------
def rel_change(seq):
    if len(seq) < 3: return 0.0
    a0, a1 = seq[0], seq[-1]
    if a0 <= 0: return 0.0
    return (a1 - a0) / max(a0, 1e-6)

# ---------- Track object ----------
class Track:
    def __init__(self, oid):
        self.oid = oid
        self.slot = None            # leased slot (1..SLOT_COUNT)
        self.best_text = ""         # best OCR seen
        self.areas = collections.deque(maxlen=AREA_WINDOW)
        self.last_seen_f = 0        # frame index last seen
        self.state = "new"          # new -> in -> leaving -> gone

    def update_text(self, t):
        # prefer longer string; if equal, prefer one that contains a numeric suffix
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
        # Enough growth across window?
        rc = rel_change(list(self.areas))
        return rc >= MIN_REL_GROW

    def decide_out(self):
        rc = rel_change(list(self.areas))
        return rc <= MIN_REL_SHRINK

def main():
    # video source
    cap = cv2.VideoCapture(0 if USE_CAMERA else INPUT_VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera/video source")

    # detectors
    car_det = CarDetection(model_path="yolo11n.pt")
    lp_det  = LicencePlateDetection(model_path="models/best.pt")

    # output video (MJPG/AVI)
    os.makedirs("output_videos", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter("output_videos/output_video.avi", fourcc, fps, (w, h))

    # CSV
    csv_file, csv_writer, csv_path = open_daily_csv()
    print("[INFO] Logging to:", csv_path)

    # state
    idbank = IDBank(START_ID_AT)
    tracks = {}                # internal oid -> Track
    slot_pool = SlotPool(SLOT_COUNT)
    frame_idx = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok: break
            frame_idx += 1

            car_dets = car_det.detect_frames([frame], read_from_stub=False)[0]
            all_lp_dets, all_lp_texts = lp_det.detect_frames([frame])
            lp_dets, lp_texts = all_lp_dets[0], all_lp_texts[0]

            # Draw overlays
            frame = car_det.draw_bboxes([frame], [car_dets])[0]
            frame = lp_det.draw_bboxes([frame], [lp_dets], [lp_texts])[0]

            # --- update tracks for current detections ---
            seen_oids = set()
            for det, text in zip(lp_dets, lp_texts):
                key = make_key(text, det)
                if key is None:
                    continue
                oid = idbank.get(key)
                tr = tracks.get(oid)
                if tr is None:
                    tr = tracks[oid] = Track(oid)
                tr.update_text(text or "")
                area = area_from_det(det)
                tr.observe(area, frame_idx)
                seen_oids.add(oid)

                # ENTRY: car approaching & no slot yet & slot available
                if tr.slot is None and tr.decide_in():
                    slot = slot_pool.acquire()
                    if slot is not None:
                        tr.slot = slot
                        tr.state = "in"
                        # Build object_id string like "01(70-50)"
                        suffix = parse_number_suffix(tr.best_text) or ""
                        object_id = f"{slot:02d}({suffix})" if suffix else f"{slot:02d}"
                        # CSV
                        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        city, kana = parse_city_kana(tr.best_text)
                        engine = parse_engine_size(tr.best_text)
                        csv_writer.writerow([ts, object_id, "car", "in", city or "", engine or "", kana or ""])
                        print(f"[IN ] {ts} object_id={object_id} text='{tr.best_text}'")

                # If already 'in' and trend says leaving, mark 'leaving'
                if tr.slot is not None and tr.state == "in" and tr.decide_out():
                    tr.state = "leaving"

            # --- process exits for tracks NOT seen this frame (or already leaving) ---
            for oid, tr in list(tracks.items()):
                if tr.slot is None:
                    continue
                # mark missing frames
                missed = frame_idx - tr.last_seen_f
                if tr.state == "leaving" and missed >= MISS_GRACE_FRAMES:
                    # finalize OUT, free slot
                    suffix = parse_number_suffix(tr.best_text) or ""
                    object_id = f"{tr.slot:02d}({suffix})" if suffix else f"{tr.slot:02d}"
                    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    city, kana = parse_city_kana(tr.best_text)
                    engine = parse_engine_size(tr.best_text)
                    csv_writer.writerow([ts, object_id, "car", "out", city or "", engine or "", kana or ""])
                    print(f"[OUT] {ts} object_id={object_id} text='{tr.best_text}'")
                    slot_pool.release(tr.slot)
                    del tracks[oid]   # remove the whole track

            # --- show & save ---
            # draw slot labels on screen for fun
            for oid, tr in tracks.items():
                if tr.slot is None: continue
                # try to draw near last bbox center (approx via last area trend is tricky;
                # we skip exact position drawing for brevity)
                cv2.putText(frame, f"{tr.slot:02d}", (10, 30 + 20*(tr.slot%20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2, cv2.LINE_AA)

            cv2.imshow("Parking (slot leasing)", frame)
            out.write(frame)
            wait = 1 if USE_CAMERA else 20
            if cv2.waitKey(wait) & 0xFF == 27:
                break

    except KeyboardInterrupt:
        print("[INFO] Interrupted by user.")
    finally:
        cap.release(); out.release(); csv_file.close()
        cv2.destroyAllWindows()
        print("[INFO] Video saved to output_videos/output_video.avi")
        print("[INFO] CSV saved to", csv_path)

if __name__ == "__main__":
    main()
