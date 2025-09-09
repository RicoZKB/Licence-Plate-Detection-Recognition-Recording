# -*- coding: utf-8 -*-
# Live car + license-plate detection with CSV logging on gate-crossing (Lite/Fast)
# - Stays close to the original (simple & responsive)
# - Adds: camera/video switch + small duplicate guard for gate events
# - Creates logs/parking_log_YYYYMMDD.csv and output_videos/output_video.avi

import cv2
import os
import csv
import re
from datetime import datetime
from detections import CarDetection, LicencePlateDetection

# ========================= User-tunable settings =========================
USE_CAMERA     = False                    # True: USB cam index 0, False: read from file
INPUT_VIDEO_PATH = "input_videos/Trim20.mp4"

GATE_Y_RATIO   = 0.55                   # 0.0 (top) .. 1.0 (bottom). Virtual gate position
ENTRY_IS_UP    = True                   # True: upward crossing = "in". False: downward = "in"
START_ID_AT    = 100                    # first object_id (internal, stable_key->id)
MIN_FRAMES_BETWEEN_EVENTS = 12          # debounce: avoid duplicate logs per same id near gate
# ========================================================================

# ---------- Simple ID manager ----------
class IDBank(object):
    def __init__(self, start_at):
        self._map = {}
        self._next = int(start_at)
    def get_id(self, key):
        if key not in self._map:
            self._map[key] = self._next
            self._next += 1
        return self._map[key]

# ---------- Very light JP plate parser ----------
# Expects like: "神戸 500 わ 12-34" or "大阪 330 ふ 12-34"
PLATE_RE = re.compile(u"^\\s*([^\\s\\d]+)\\s+[0-9]{3,4}\\s*([ぁ-ゖァ-ヿ一-龯A-Za-z])", re.UNICODE)

def parse_city_kana(plate_text):
    if not plate_text:
        return None, None
    m = PLATE_RE.search(plate_text)
    if not m:
        return None, None
    return m.group(1), m.group(2)

# ---------- CSV helpers ----------
def open_daily_csv():
    date_str = datetime.now().strftime("%Y%m%d")
    os.makedirs("logs", exist_ok=True)
    path = os.path.join("logs", "parking_log_{0}.csv".format(date_str))
    new_file = not os.path.exists(path)
    f = open(path, "a", newline="", encoding="utf-8")
    w = csv.writer(f)
    if new_file:
        w.writerow(["timestamp", "object_id", "vehicle_type", "direction", "city", "kana"])
    return f, w, path

# ---------- Geometry helpers ----------
def extract_xyxy(det):
    # Supports [x1,y1,x2,y2,...] or dict with 'bbox' or 'xyxy'
    if isinstance(det, (list, tuple)) and len(det) >= 4:
        return float(det[0]), float(det[1]), float(det[2]), float(det[3])
    if isinstance(det, dict):
        box = det.get("bbox") or det.get("xyxy")
        if box and len(box) >= 4:
            return float(box[0]), float(box[1]), float(box[2]), float(box[3])
    return None

def det_center_y(det):
    xyxy = extract_xyxy(det)
    if xyxy is None:
        return None
    return (xyxy[1] + xyxy[3]) / 2.0

# ---------- Stable key for tracking (robust when OCR text is noisy) ----------
def stable_key(text, det):
    xyxy = extract_xyxy(det)
    if xyxy is None:
        return text if text else None
    # Quantize center so small jitter maps to same bucket
    cx = (xyxy[0] + xyxy[2]) / 2.0
    cy = (xyxy[1] + xyxy[3]) / 2.0
    qx = int(cx // 20)  # bucket size: 20 px
    qy = int(cy // 20)
    # Prefer reasonable text; else, bbox bucket
    if text and len(text) >= 3:
        return text
    return "bb-{0}-{1}".format(qx, qy)

def main():
    # ---- single capture (camera or file) ----
    src = 0 if USE_CAMERA else INPUT_VIDEO_PATH
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError("Could not open video source: {0}".format(str(src)))

    # ---- detectors ----
    car_detector = CarDetection(model_path="yolo11n.pt")
    lp_detector  = LicencePlateDetection(model_path="models/best.pt")

    # ---- output writer (MJPG/AVI) ----
    os.makedirs("output_videos", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    fps    = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1:
        fps = 20.0
    w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    out    = cv2.VideoWriter("output_videos/output_video.avi", fourcc, fps, (w, h))

    # ---- CSV ----
    csv_file, csv_writer, csv_path = open_daily_csv()
    print("[INFO] Logging to: {0}".format(csv_path))

    # ---- tracking state ----
    id_bank = IDBank(START_ID_AT)
    last_y_by_id = {}
    last_event_frame_by_id = {}  # debounce per id

    gate_y = int(h * GATE_Y_RATIO)
    frame_idx = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1

            # Run detections (simple, every frame)
            car_dets = car_detector.detect_frames([frame], read_from_stub=False)[0]
            all_lp_dets, all_lp_texts = lp_detector.detect_frames([frame])
            lp_dets, lp_texts = all_lp_dets[0], all_lp_texts[0]

            # Draw car + plate (simple & direct)
            frame = car_detector.draw_bboxes([frame], [car_dets])[0]
            frame = lp_detector.draw_bboxes([frame], [lp_dets], [lp_texts])[0]

            # Draw virtual gate
            cv2.line(frame, (0, gate_y), (w, gate_y), (0, 255, 255), 2)
            label = "GATE y={0} ({1})".format(gate_y, "up=in" if ENTRY_IS_UP else "down=in")
            cv2.putText(frame, label, (10, max(25, gate_y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

            # Process license-plate detections for crossing events
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            for det, text in zip(lp_dets, lp_texts):
                key = stable_key(text, det)
                if key is None:
                    continue

                oid = id_bank.get_id(key)
                cy  = det_center_y(det)
                if cy is None:
                    continue

                prev_y = last_y_by_id.get(oid)
                last_y_by_id[oid] = cy

                # First sighting for this id — no direction yet
                if prev_y is None:
                    continue

                # Check crossing (sign change around the gate)
                crossed = (prev_y - gate_y) * (cy - gate_y) < 0
                if not crossed:
                    continue

                # Debounce: skip if last event for this id was too recent
                last_evt_f = last_event_frame_by_id.get(oid, -999999)
                if frame_idx - last_evt_f < MIN_FRAMES_BETWEEN_EVENTS:
                    continue
                last_event_frame_by_id[oid] = frame_idx

                moving_up = cy < prev_y
                if ENTRY_IS_UP:
                    direction = "in" if moving_up else "out"
                else:
                    direction = "in" if not moving_up else "out"

                city, kana = parse_city_kana(text or "")
                csv_writer.writerow([ts, oid, "car", direction, city or "", kana or ""])
                print("[EVENT] {0} id={1} dir={2} text='{3}'".format(ts, oid, direction, text))

            # Show & save (keep UI responsive)
            cv2.imshow("Live Detection (Lite)", frame)
            out.write(frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
                break

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")

    finally:
        try: cap.release()
        except: pass
        try: out.release()
        except: pass
        cv2.destroyAllWindows()
        csv_file.close()
        print("[INFO] Video saved to output_videos/output_video.avi")
        print("[INFO] CSV saved to {0}".format(csv_path))

if __name__ == "__main__":
    main()
