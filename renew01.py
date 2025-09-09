# -*- coding: utf-8 -*-
# Gate-crossing logger with 10-slot "leasing" (01..10) and plate-based ID recall.
# - Simple, fast loop (like your old code)
# - On IN: assign preferred old slot if free, else lowest free slot
# - On OUT: free slot
# - Logs exactly once per IN/OUT per vehicle
# CSV: timestamp,object_id,vehicle_type,direction,city,kana
# Video: output_videos/output_video.avi (MJPG/AVI)

import cv2
import os
import csv
import re
from datetime import datetime
from detections import CarDetection, LicencePlateDetection

# ========================= User-tunable settings =========================
USE_CAMERA        = True                  # True: USB camera(0). False: read from file below
INPUT_VIDEO_PATH  = "input_videos/Trim20.mp4"

GATE_Y_RATIO      = 0.55                 # 0.0..1.0 virtual gate position
ENTRY_IS_UP       = True                 # True: upward crossing => "in", else downward => "in"
START_ID_AT       = 100                  # internal id seed (stable_key -> internal id)
SLOT_COUNT        = 10                   # total parking places -> slots 01..10
MIN_EVENT_GAP_FRAMES = 12                # debounce (per internal id) to avoid double logs
# ========================================================================

# ---------- Regex for simple JP plate parsing ----------
PLATE_CITY_KANA_RE = re.compile(u"^\\s*([^\\s\\d]+)\\s+[0-9]{3,4}\\s*([ぁ-ゖァ-ヿ一-龯A-Za-z])", re.UNICODE)
PLATE_SUFFIX_RE    = re.compile(r"([0-9]{2,3}-[0-9]{2})")

def parse_city_kana(plate_text):
    if not plate_text:
        return None, None
    m = PLATE_CITY_KANA_RE.search(plate_text)
    if not m:
        return None, None
    return m.group(1), m.group(2)

def parse_suffix(plate_text):
    if not plate_text:
        return None
    m = PLATE_SUFFIX_RE.search(plate_text)
    return m.group(1) if m else None

# ---------- CSV ----------
def open_daily_csv():
    date_str = datetime.now().strftime("%Y%m%d")
    os.makedirs("logs", exist_ok=True)
    path = os.path.join("logs", "parking_log_{0}.csv".format(date_str))
    new_file = not os.path.exists(path)
    f = open(path, "a", newline="", encoding="utf-8")
    w = csv.writer(f)
    if new_file:
        w.writerow(["timestamp","object_id","vehicle_type","direction","city","kana"])
    return f, w, path

# ---------- Geometry ----------
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

# ---------- Simple internal ID manager (for per-frame tracking) ----------
class IDBank(object):
    def __init__(self, start_at):
        self._map = {}
        self._next = int(start_at)
    def get_id(self, key):
        if key not in self._map:
            self._map[key] = self._next
            self._next += 1
        return self._map[key]

# ---------- Stable key for tracking (bbox bucket fallback if OCR is weak) ----------
def stable_key(text, det):
    xyxy = extract_xyxy(det)
    if xyxy is None:
        return text if text else None
    cx = (xyxy[0] + xyxy[2]) / 2.0
    cy = (xyxy[1] + xyxy[3]) / 2.0
    qx = int(cx // 20)  # 20px grid
    qy = int(cy // 20)
    if text and len(text) >= 3:
        return text
    return "bb-{0}-{1}".format(qx, qy)

# ---------- Slot pool 01..N ----------
class SlotPool(object):
    def __init__(self, n):
        self.free = list(range(1, n+1))  # 1..n
        self.used = set()
    def acquire_lowest(self):
        if not self.free:
            return None
        s = self.free.pop(0)
        self.used.add(s)
        return s
    def try_acquire_specific(self, s):
        if s in self.free:
            self.free.remove(s)
            self.used.add(s)
            return s
        return None
    def release(self, s):
        if s in self.used:
            self.used.remove(s)
            self.free.append(s)
            self.free.sort()

# ---------- Preferred slot memory (plate -> preferred slot) ----------
def make_plate_key(text):
    # Use number suffix if possible (e.g., "70-50"); optionally prepend city for stability.
    if not text:
        return None
    suf = parse_suffix(text)
    if not suf:
        return None
    city, _kana = parse_city_kana(text)
    # If city not recognized, we still use suffix alone.
    return "{0}|{1}".format(city if city else "?", suf)

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

    # ---- tracking + leasing state ----
    id_bank = IDBank(START_ID_AT)
    last_y_by_id = {}
    last_event_frame_by_id = {}  # debounce for IN/OUT events per internal-id

    slot_pool = SlotPool(SLOT_COUNT)
    plate_pref_slot = {}         # plate_key -> preferred slot (remember across visits)
    active_plate_to_slot = {}    # plate_key -> slot currently occupied
    active_id_to_plate = {}      # internal-id -> plate_key for current session

    gate_y = int(h * GATE_Y_RATIO)
    frame_idx = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1

            # Run detections (simple each frame)
            car_dets = car_detector.detect_frames([frame], read_from_stub=False)[0]
            all_lp_dets, all_lp_texts = lp_detector.detect_frames([frame])
            lp_dets, lp_texts = all_lp_dets[0], all_lp_texts[0]

            # Draw car + plate (fast path)
            frame = car_detector.draw_bboxes([frame], [car_dets])[0]
            frame = lp_detector.draw_bboxes([frame], [lp_dets], [lp_texts])[0]

            # Draw virtual gate
            cv2.line(frame, (0, gate_y), (w, gate_y), (0, 255, 255), 2)
            label = "GATE y={0} ({1})".format(gate_y, "up=in" if ENTRY_IS_UP else "down=in")
            cv2.putText(frame, label, (10, max(25, gate_y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2, cv2.LINE_AA)

            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # ---- process plates for crossing events ----
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

                if prev_y is None:
                    continue

                crossed = (prev_y - gate_y) * (cy - gate_y) < 0
                if not crossed:
                    continue

                # Debounce this internal id
                last_evt_f = last_event_frame_by_id.get(oid, -999999)
                if frame_idx - last_evt_f < MIN_EVENT_GAP_FRAMES:
                    continue
                last_event_frame_by_id[oid] = frame_idx

                moving_up = cy < prev_y
                if ENTRY_IS_UP:
                    direction = "in" if moving_up else "out"
                else:
                    direction = "in" if not moving_up else "out"

                # Normalize plate for leasing logic
                plate_key = make_plate_key(text or "")

                if direction == "in":
                    # If plate already active, ignore duplicate "in"
                    if plate_key and plate_key in active_plate_to_slot:
                        # Already parked; do not double-log
                        continue

                    # Choose slot
                    chosen_slot = None
                    if plate_key and plate_key in plate_pref_slot:
                        # Try to re-acquire preferred slot
                        prefer = plate_pref_slot[plate_key]
                        got = slot_pool.try_acquire_specific(prefer)
                        if got is not None:
                            chosen_slot = got

                    if chosen_slot is None:
                        # Acquire lowest free
                        chosen_slot = slot_pool.acquire_lowest()

                    if chosen_slot is None:
                        # Parking full: skip logging but keep running
                        print("[WARN] Parking full: cannot assign slot for plate '{0}'".format(text))
                        continue

                    # Mark active
                    if plate_key:
                        active_plate_to_slot[plate_key] = chosen_slot
                        plate_pref_slot[plate_key] = chosen_slot  # remember preference
                        active_id_to_plate[oid] = plate_key
                    else:
                        # No reliable plate key: remember by internal id only
                        active_id_to_plate[oid] = None  # we can still log, but recall won't work

                    # Build object_id "SS(suffix)" if suffix exists, else "SS"
                    suffix = parse_suffix(text or "")
                    object_id = "{0:02d}({1})".format(chosen_slot, suffix) if suffix else "{0:02d}".format(chosen_slot)

                    city, kana = parse_city_kana(text or "")
                    csv_writer.writerow([ts, object_id, "car", "in", city or "", kana or ""])
                    print("[IN ] {0} object_id={1} plate='{2}'".format(ts, object_id, text))

                else:  # direction == "out"
                    # Find which slot to free:
                    # Prefer matching by plate_key; if missing, fall back to internal id mapping.
                    slot_to_free = None
                    pk = plate_key
                    if pk and pk in active_plate_to_slot:
                        slot_to_free = active_plate_to_slot.get(pk)
                        # clear active mapping
                        try: del active_plate_to_slot[pk]
                        except: pass
                    else:
                        # maybe we saw plate unclearly; try internal id to plate_key
                        pk2 = active_id_to_plate.get(oid)
                        if pk2 and pk2 in active_plate_to_slot:
                            slot_to_free = active_plate_to_slot.get(pk2)
                            try: del active_plate_to_slot[pk2]
                            except: pass

                    if slot_to_free is None:
                        # We didn't have this car recorded as IN (e.g., missed earlier), skip OUT log
                        continue

                    suffix = parse_suffix(text or "")
                    object_id = "{0:02d}({1})".format(slot_to_free, suffix) if suffix else "{0:02d}".format(slot_to_free)
                    city, kana = parse_city_kana(text or "")
                    csv_writer.writerow([ts, object_id, "car", "out", city or "", kana or ""])
                    print("[OUT] {0} object_id={1} plate='{2}'".format(ts, object_id, text))

                    # Free the slot
                    slot_pool.release(slot_to_free)

                    # Do not delete plate_pref_slot: keep preference for future re-entry

            # ---- show & save ----
            cv2.imshow("Live Parking (10-slot leasing)", frame)
            out.write(frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
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
