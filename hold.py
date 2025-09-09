# -*- coding: utf-8 -*-
# Car + License Plate detection with CSV logging.
# Modes:
#   DIRECTION_MODE="gate"  -> log only when plate crosses a virtual gate line (in/out)
#   DIRECTION_MODE="first" -> log once on first sighting (direction preset)
#
# Creates: logs/parking_log_YYYYMMDD.csv
# Saves video: output_videos/output_video.avi (MJPG/AVI for QuickTime)

import os, csv, re, cv2
from datetime import datetime
from detections import CarDetection, LicencePlateDetection

# ===================== USER SETTINGS =====================
USE_CAMERA = False                          # True: webcam(0), False: use INPUT_VIDEO_PATH
INPUT_VIDEO_PATH = "input_videos/Trim20.mp4"  # used when USE_CAMERA=False

DIRECTION_MODE = "first"   # "gate" or "first"
FIRST_SIGHT_DIRECTION = "in"  # used when DIRECTION_MODE="first" ("in" or "out")

GATE_Y_RATIO = 0.55        # gate position (0 top .. 1 bottom), used in "gate" mode
ENTRY_IS_UP   = True       # if True: up crossing = "in", else down crossing = "in"

ID_BY = "bbox"             # "bbox" (robust with noisy OCR) or "text"
START_ID_AT = 100
# ==========================================================

# ---------- Simple ID bank ----------
class IDBank(object):
    def __init__(self, start_at):
        self._map = {}
        self._next = int(start_at)
    def get(self, key):
        if key not in self._map:
            self._map[key] = self._next
            self._next += 1
        return self._map[key]

# ---------- JP plate parser (best effort) ----------
PLATE_RE = re.compile(u"^\\s*([^\\s\\d]+)\\s+[0-9]{3,4}\\s*([ぁ-ゖァ-ヿ一-龯A-Za-z])", re.UNICODE)
def parse_city_kana(text):
    if not text: return None, None
    m = PLATE_RE.search(text)
    return (m.group(1), m.group(2)) if m else (None, None)

# ---------- CSV helpers ----------
def open_daily_csv():
    date_str = datetime.now().strftime("%Y%m%d")
    os.makedirs("logs", exist_ok=True)
    path = os.path.join("logs", "parking_log_{0}.csv".format(date_str))
    new = not os.path.exists(path)
    f = open(path, "a", newline="", encoding="utf-8")
    w = csv.writer(f)
    if new:
        w.writerow(["timestamp","object_id","vehicle_type","direction","city","kana"])
    return f, w, path

# ---------- geometry ----------
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

# ---------- stable keys ----------
def key_bbox(det):
    cx, cy = center_from_det(det)
    if cx is None: return None
    # quantize in 20px buckets so small jitter maps to same key
    return "bb-{0}-{1}".format(int(cx//20), int(cy//20))

def key_text(text):
    if not text: return None
    # normalize obvious noise like leading asterisks/spaces
    t = re.sub(r"[^\wぁ-ゖァ-ヿ一-龯\-]+", "", text)
    return t if len(t) >= 3 else None

def make_key(text, det):
    if ID_BY == "text":
        return key_text(text) or key_bbox(det)
    else:  # default robust path
        return key_bbox(det) or key_text(text)

def main():
    # ---- video source ----
    cap = cv2.VideoCapture(0 if USE_CAMERA else INPUT_VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera/video source")

    # ---- detectors ----
    car_det = CarDetection(model_path="yolo11n.pt")
    lp_det  = LicencePlateDetection(model_path="models/best.pt")

    # ---- output video ----
    os.makedirs("output_videos", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter("output_videos/output_video.avi", fourcc, fps, (w, h))

    # ---- CSV ----
    csv_file, csv_writer, csv_path = open_daily_csv()
    print("[INFO] Logging to:", csv_path)

    # ---- tracking states ----
    idbank = IDBank(START_ID_AT)
    last_y = {}        # object_id -> last center y
    seen_once = set()  # for "first" mode
    gate_y = int(h * GATE_Y_RATIO)

    try:
        while True:
            ok, frame = cap.read()
            if not ok: break

            # run detectors
            car_dets = car_det.detect_frames([frame], read_from_stub=False)[0]
            all_lp_dets, all_lp_texts = lp_det.detect_frames([frame])
            lp_dets, lp_texts = all_lp_dets[0], all_lp_texts[0]

            # draw
            frame = car_det.draw_bboxes([frame], [car_dets])[0]
            frame = lp_det.draw_bboxes([frame], [lp_dets], [lp_texts])[0]

            # gate line (always drawn so you can see it)
            cv2.line(frame, (0, gate_y), (w, gate_y), (0,255,255), 2)
            cv2.putText(frame, "GATE y={} ({})".format(
                gate_y, "up=in" if ENTRY_IS_UP else "down=in"
            ), (10, max(25, gate_y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2, cv2.LINE_AA)

            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            for det, text in zip(lp_dets, lp_texts):
                key = make_key(text, det)
                if key is None: 
                    continue
                oid = idbank.get(key)
                cx, cy = center_from_det(det)
                if cy is None: 
                    continue

                # ---------- FIRST-SIGHT logging ----------
                if DIRECTION_MODE.lower() == "first":
                    if oid not in seen_once:
                        seen_once.add(oid)
                        city, kana = parse_city_kana(text or "")
                        csv_writer.writerow([ts, oid, "car", FIRST_SIGHT_DIRECTION, city or "", kana or ""])
                        print("[FIRST] {} id={} dir={} text='{}'".format(ts, oid, FIRST_SIGHT_DIRECTION, text))
                    # still update last_y for potential later switch/testing
                    last_y[oid] = cy
                    continue

                # ---------- GATE-CROSS logging ----------
                prev = last_y.get(oid)
                last_y[oid] = cy
                if prev is None:
                    continue
                crossed = (prev - gate_y) * (cy - gate_y) < 0
                if not crossed:
                    continue
                moving_up = cy < prev
                direction = ("in" if moving_up else "out") if ENTRY_IS_UP else ("in" if not moving_up else "out")
                city, kana = parse_city_kana(text or "")
                csv_writer.writerow([ts, oid, "car", direction, city or "", kana or ""])
                print("[EVENT] {} id={} dir={} text='{}'".format(ts, oid, direction, text))

            # show & save
            cv2.imshow("Detection", frame)
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
