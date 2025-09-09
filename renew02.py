# -*- coding: utf-8 -*-
# Gate-crossing logger with 10-slot leasing (01..10) + plate-based ID recall.
# Simple, fast loop (one capture). Logs once on IN and once on OUT.
#
# CSV: timestamp,object_id,vehicle_type,direction,city,kana
# object_id: "SS" or "SS(12-34)" where SS is slot 01..10 and "(12-34)" is plate suffix if OCR caught it.
# Video: output_videos/output_video.avi (MJPG/AVI)
#
# Controls at runtime:
#  - ESC: quit
#  - 'd': toggle ENTRY_IS_UP (flip which direction counts as "in")
#  - 'j'/'k': move gate line down/up by 10px to line it up with crossings
#
# Requires your CarDetection and LicencePlateDetection classes:
#   from detections import CarDetection, LicencePlateDetection

import cv2
import os
import csv
import re
from datetime import datetime
from detections import CarDetection, LicencePlateDetection

# ========================= User-tunable settings =========================
USE_CAMERA          = True                   # True: camera index 0; False: read from video file
INPUT_VIDEO_PATH    = "input_videos/Trim20.mp4"

GATE_Y_RATIO        = 0.55                   # 0.0 (top) .. 1.0 (bottom)
ENTRY_IS_UP         = True                   # True: up crossing = "in"; False: down crossing = "in"

SLOT_COUNT          = 10                     # total parking slots: 01..SLOT_COUNT
START_ID_AT         = 100                    # seed for internal ID mapping (stable_key -> int)

MIN_EVENT_GAP_FRAMES = 12                    # debounce per-internal-id to avoid double logs near gate

# Debug UI
DEBUG_DRAW          = True                   # draw plate center + "above/below"
DEBUG_PRINT_NEAR    = True                   # console print when plate center is near gate
NEAR_PIXELS         = 20                     # "near gate" band (pixels)
# ========================================================================

# ---------- JP plate parsing ----------
PLATE_CITY_KANA_RE = re.compile(u"^\\s*([^\\s\\d]+)\\s+[0-9]{3,4}\\s*([ぁ-ゖァ-ヿ一-龯A-Za-z])", re.UNICODE)
PLATE_SUFFIX_RE    = re.compile(r"([0-9]{2,3}-[0-9]{2})")  # e.g., 12-34, 123-45

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

def make_plate_key(plate_text):
    # Use number suffix + optional city to recall same slot across visits.
    if not plate_text:
        return None
    suf = parse_suffix(plate_text)
    if not suf:
        return None
    city, _kana = parse_city_kana(plate_text)
    return "{0}|{1}".format(city if city else "?", suf)

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

# ---------- Internal ID manager ----------
class IDBank(object):
    def __init__(self, start_at):
        self._map = {}
        self._next = int(start_at)
    def get_id(self, key):
        if key not in self._map:
            self._map[key] = self._next
            self._next += 1
        return self._map[key]

# ---------- Stable key (OCR-robust) ----------
def stable_key(text, det):
    # Always include bbox-quantized bucket so OCR noise won't break tracking
    xyxy = extract_xyxy(det)
    if xyxy is None:
        return text if text else None
    cx = (xyxy[0] + xyxy[2]) / 2.0
    cy = (xyxy[1] + xyxy[3]) / 2.0
    qx = int(cx // 20)  # 20px buckets
    qy = int(cy // 20)
    if text and len(text) >= 3:
        return "bb-{0}-{1}|{2}".format(qx, qy, text)
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

def main():
    # ---- capture ----
    src = 0 if USE_CAMERA else INPUT_VIDEO_PATH
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError("Could not open video source: {0}".format(str(src)))

    # ---- detectors ----
    car_detector = CarDetection(model_path="yolo11n.pt")
    lp_detector  = LicencePlateDetection(model_path="models/best.pt")

    # ---- output writer ----
    os.makedirs("output_videos", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1:
        fps = 20.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    out = cv2.VideoWriter("output_videos/output_video.avi", fourcc, fps, (w, h))

    # ---- CSV ----
    csv_file, csv_writer, csv_path = open_daily_csv()
    print("[INFO] Logging to: {0}".format(csv_path))

    # ---- state ----
    id_bank = IDBank(START_ID_AT)
    last_y_by_id = {}
    last_event_frame_by_id = {}

    slot_pool = SlotPool(SLOT_COUNT)
    plate_pref_slot = {}         # plate_key -> preferred slot (remember across visits)
    active_plate_to_slot = {}    # plate_key -> slot currently occupied
    active_id_to_plate = {}      # internal-id -> plate_key for current session

    gate_y = int(h * GATE_Y_RATIO)
    frame_idx = 0

    # IMPORTANT: copy global to local so we can toggle without UnboundLocalError
    entry_is_up = ENTRY_IS_UP

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1

            # ---- detections ----
            car_dets = car_detector.detect_frames([frame], read_from_stub=False)[0]
            all_lp_dets, all_lp_texts = lp_detector.detect_frames([frame])
            lp_dets, lp_texts = all_lp_dets[0], all_lp_texts[0]

            # ---- draw overlays ----
            frame = car_detector.draw_bboxes([frame], [car_dets])[0]
            frame = lp_detector.draw_bboxes([frame], [lp_dets], [lp_texts])[0]

            # gate line + label uses the LOCAL entry_is_up
            cv2.line(frame, (0, gate_y), (w, gate_y), (0, 255, 255), 2)
            label = "GATE y={0} ({1})".format(gate_y, "up=in" if entry_is_up else "down=in")
            cv2.putText(frame, label, (10, max(25, gate_y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2, cv2.LINE_AA)

            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # ---- process plate detections for gate crossing ----
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

                # Debug visuals near gate
                if DEBUG_DRAW:
                    xyxy = extract_xyxy(det)
                    if xyxy:
                        cx = int((xyxy[0] + xyxy[2]) / 2.0)
                        cv2.circle(frame, (cx, int(cy)), 3, (0, 0, 255), -1)
                        side = "above" if cy < gate_y else "below"
                        cv2.putText(frame, side, (cx + 6, int(cy)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
                if DEBUG_PRINT_NEAR and abs(cy - gate_y) <= NEAR_PIXELS:
                    print("[NEAR] cy={:.1f} gate_y={} key={} text='{}'".format(
                        cy, gate_y, key, text))

                # First sighting: no direction yet
                if prev_y is None:
                    continue

                # Crossing test
                crossed = (prev_y - gate_y) * (cy - gate_y) < 0
                if not crossed:
                    continue

                # Debounce per internal id
                last_evt_f = last_event_frame_by_id.get(oid, -999999)
                if frame_idx - last_evt_f < MIN_EVENT_GAP_FRAMES:
                    continue
                last_event_frame_by_id[oid] = frame_idx

                moving_up = cy < prev_y
                direction = "in" if (moving_up == entry_is_up) else "out"

                # Plate key for leasing/recall
                plate_key = make_plate_key(text or "")

                if direction == "in":
                    # Prevent double-IN for same active car
                    if plate_key and plate_key in active_plate_to_slot:
                        continue

                    # Choose a slot
                    chosen_slot = None
                    if plate_key and plate_key in plate_pref_slot:
                        prefer = plate_pref_slot[plate_key]
                        got = slot_pool.try_acquire_specific(prefer)
                        if got is not None:
                            chosen_slot = got
                    if chosen_slot is None:
                        chosen_slot = slot_pool.acquire_lowest()

                    if chosen_slot is None:
                        print("[WARN] Parking full: cannot assign slot for plate '{}'".format(text))
                        continue

                    # Mark active + remember preference
                    if plate_key:
                        active_plate_to_slot[plate_key] = chosen_slot
                        plate_pref_slot[plate_key] = chosen_slot
                        active_id_to_plate[oid] = plate_key
                    else:
                        active_id_to_plate[oid] = None

                    # object_id
                    suffix = parse_suffix(text or "")
                    object_id = "{0:02d}({1})".format(chosen_slot, suffix) if suffix else "{0:02d}".format(chosen_slot)
                    city, kana = parse_city_kana(text or "")
                    csv_writer.writerow([ts, object_id, "car", "in", city or "", kana or ""])
                    print("[IN ] {0} object_id={1} plate='{2}'".format(ts, object_id, text))

                else:  # OUT
                    # Find slot to free
                    slot_to_free = None
                    if plate_key and plate_key in active_plate_to_slot:
                        slot_to_free = active_plate_to_slot.get(plate_key)
                        try: del active_plate_to_slot[plate_key]
                        except: pass
                    else:
                        pk2 = active_id_to_plate.get(oid)
                        if pk2 and pk2 in active_plate_to_slot:
                            slot_to_free = active_plate_to_slot.get(pk2)
                            try: del active_plate_to_slot[pk2]
                            except: pass

                    if slot_to_free is None:
                        # we never logged an IN for this one (missed earlier), skip OUT
                        continue

                    suffix = parse_suffix(text or "")
                    object_id = "{0:02d}({1})".format(slot_to_free, suffix) if suffix else "{0:02d}".format(slot_to_free)
                    city, kana = parse_city_kana(text or "")
                    csv_writer.writerow([ts, object_id, "car", "out", city or "", kana or ""])
                    print("[OUT] {0} object_id={1} plate='{2}'".format(ts, object_id, text))

                    slot_pool.release(slot_to_free)
                    # keep plate_pref_slot so it gets same slot when returning (if free)

            # ---- show & save ----
            cv2.imshow("Live Parking (10-slot leasing)", frame)
            out.write(frame)

            # Keys
            k = cv2.waitKey(1) & 0xFF
            if k == 27:  # ESC
                break
            elif k == ord('d'):
                entry_is_up = not entry_is_up
                print("[TOGGLE] ENTRY_IS_UP (local) -> {}".format(entry_is_up))
            elif k == ord('j'):  # move gate down
                gate_y = min(gate_y + 10, h - 1)
                print("[GATE] gate_y -> {}".format(gate_y))
            elif k == ord('k'):  # move gate up
                gate_y = max(gate_y - 10, 0)
                print("[GATE] gate_y -> {}".format(gate_y))

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
