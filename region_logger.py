# -*- coding: utf-8 -*-
# Region-based logger with 10-slot leasing (01..10) + plate-based slot recall.
# Logs exactly once on region ENTER and once on region EXIT.
#
# CSV: timestamp,object_id,vehicle_type,direction,city,kana
# object_id: "SS" or "SS(12-34)" where SS is slot 01..10 and "(12-34)" is plate suffix if OCR caught it.
# Video: output_videos/output_video.avi (MJPG/AVI)
#
# Controls at runtime:
#   ESC : quit
#   wasd : move region (W=up, A=left, S=down, D=right) by 10 px
#   [ / ]: resize region (decrease/increase) by 10 px per side
#   r    : reset region to default
#
# Requires:
#   from detections import CarDetection, LicencePlateDetection
#   Your models: yolo11n.pt (cars) and models/best.pt (plates+OCR)

import os
import cv2
import csv
import re
from datetime import datetime
from detections import CarDetection, LicencePlateDetection

# ========================= User settings =========================
USE_CAMERA            = False                     # True: webcam(0); False: read from INPUT_VIDEO_PATH
INPUT_VIDEO_PATH      = "input_videos/Trim03.mp4"

SLOT_COUNT            = 10                       # 01..SLOT_COUNT
START_ID_AT           = 100                      # internal ID seed

# Region (as ratios of frame size). You can adjust live with keys.
REGION_XYWH_RATIO     = (0.25, 0.45, 0.50, 0.25) # x,y,w,h as fractions of (W,H)

ENTER_STABLE_FRAMES   = 3    # need this many consecutive frames inside before logging "in"
EXIT_STABLE_FRAMES    = 6    # need this many consecutive frames outside before logging "out"
MIN_EVENT_GAP_FRAMES  = 12   # debounce per internal-id

DRAW_REGION           = True
# =================================================================

# ---------- JP plate parsing ----------
CITY_KANA_RE = re.compile(u"^\\s*([^\\s\\d]+)\\s+[0-9]{3,4}\\s*([ぁ-ゖァ-ヿ一-龯A-Za-z])", re.UNICODE)
PLATE_SUFFIX_RE = re.compile(r"([0-9]{2,3}-[0-9]{2})")

def parse_city_kana(text):
    if not text:
        return None, None
    m = CITY_KANA_RE.search(text)
    return (m.group(1), m.group(2)) if m else (None, None)

def plate_suffix(text):
    if not text:
        return None
    m = PLATE_SUFFIX_RE.search(text)
    return m.group(1) if m else None

def plate_key(text):
    # use suffix + optional city as the stable recall key
    suf = plate_suffix(text or "")
    if not suf:
        return None
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

# ---------- Internal ID bank ----------
class IDBank:
    def __init__(self, start_at=100):
        self._map = {}
        self._next = int(start_at)
    def get(self, key):
        if key not in self._map:
            self._map[key] = self._next
            self._next += 1
        return self._map[key]

# ---------- Stable key (OCR-tolerant) ----------
def stable_key(text, det):
    # bbox-bucket + (optional) text keeps identities stable without heavy tracking
    xy = xyxy_from_det(det)
    if xy is None:
        return text if text else None
    x1,y1,x2,y2 = xy
    qx = int(((x1+x2)/2.0) // 20)
    qy = int(((y1+y2)/2.0) // 20)
    t  = text if (text and len(text)>=3) else ""
    return f"bb-{qx}-{qy}|{t}"

# ---------- Slot pool ----------
class SlotPool:
    def __init__(self, n):
        self.free = list(range(1, n+1))  # 1..n
        self.used = set()
    def acquire_lowest(self):
        if not self.free: return None
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

# ---------- Helpers ----------
def draw_region(frame, rx, ry, rw, rh):
    cv2.rectangle(frame, (rx, ry), (rx+rw, ry+rh), (0,255,255), 2)
    cv2.putText(frame, "REGION (enter/exit here)", (rx+5, max(15, ry-6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2, cv2.LINE_AA)

def inside_region(cx, cy, rx, ry, rw, rh):
    return (cx is not None) and (cy is not None) and (rx <= cx <= rx+rw) and (ry <= cy <= ry+rh)

def clamp_region(rx, ry, rw, rh, W, H):
    rx = max(0, min(rx, W-1))
    ry = max(0, min(ry, H-1))
    rw = max(10, min(rw, W-rx))
    rh = max(10, min(rh, H-ry))
    return rx, ry, rw, rh

def main():
    # ---- video source ----
    src = 0 if USE_CAMERA else INPUT_VIDEO_PATH
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {src}")

    # ---- detectors ----
    car_det = CarDetection(model_path="yolo11n.pt")
    lp_det  = LicencePlateDetection(model_path="models/best.pt")

    # ---- writer ----
    os.makedirs("output_videos", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    out = cv2.VideoWriter("output_videos/output_video.avi", fourcc, fps, (W, H))

    # ---- CSV ----
    csv_file, csv_writer, csv_path = open_daily_csv()
    print("[INFO] Logging to:", csv_path)

    # ---- region (pixels) from ratios ----
    rx = int(REGION_XYWH_RATIO[0] * W)
    ry = int(REGION_XYWH_RATIO[1] * H)
    rw = int(REGION_XYWH_RATIO[2] * W)
    rh = int(REGION_XYWH_RATIO[3] * H)
    rx, ry, rw, rh = clamp_region(rx, ry, rw, rh, W, H)

    # ---- state ----
    idbank = IDBank(START_ID_AT)
    frame_idx = 0

    # per-internal-id
    prev_inside = {}           # oid -> bool
    inside_count = {}          # oid -> frames consecutively inside
    outside_count = {}         # oid -> frames consecutively outside
    last_event_f = {}          # oid -> frame index of last event

    slot_pool = SlotPool(SLOT_COUNT)
    plate_pref_slot = {}       # plate_key -> preferred slot (sticky across visits)
    active_plate_to_slot = {}  # plate_key -> slot currently in use
    active_id_to_plate = {}    # oid -> plate_key used at entry

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1

            # detect
            car_dets = car_det.detect_frames([frame], read_from_stub=False)[0]
            all_lp_dets, all_lp_texts = lp_det.detect_frames([frame])
            lp_dets, lp_texts = all_lp_dets[0], all_lp_texts[0]

            # draw detections
            frame = car_det.draw_bboxes([frame], [car_dets])[0]
            frame = lp_det.draw_bboxes([frame], [lp_dets], [lp_texts])[0]

            # draw region
            if DRAW_REGION:
                draw_region(frame, rx, ry, rw, rh)

            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # process plates
            seen_oids = set()
            for det, text in zip(lp_dets, lp_texts):
                key = stable_key(text, det)
                if key is None:
                    continue
                oid = idbank.get(key)
                seen_oids.add(oid)

                cx, cy = center_from_det(det)
                is_in = inside_region(cx, cy, rx, ry, rw, rh)

                # init counters
                if oid not in prev_inside:
                    prev_inside[oid] = is_in
                    inside_count[oid] = 1 if is_in else 0
                    outside_count[oid] = 0 if is_in else 1
                    continue

                # update streaks
                if is_in:
                    inside_count[oid] = inside_count.get(oid,0) + 1
                    outside_count[oid] = 0
                else:
                    outside_count[oid] = outside_count.get(oid,0) + 1
                    inside_count[oid] = 0

                # debounce
                if frame_idx - last_event_f.get(oid, -10**9) < MIN_EVENT_GAP_FRAMES:
                    prev_inside[oid] = is_in
                    continue

                pk = plate_key(text or "")
                suffix = plate_suffix(text or "")
                city, kana = parse_city_kana(text or "")

                # ENTER event (outside -> inside and stable)
                if (not prev_inside[oid]) and is_in and inside_count[oid] >= ENTER_STABLE_FRAMES:
                    chosen_slot = None
                    if pk and pk in plate_pref_slot:
                        got = slot_pool.try_acquire_specific(plate_pref_slot[pk])
                        if got is not None:
                            chosen_slot = got
                    if chosen_slot is None:
                        chosen_slot = slot_pool.acquire_lowest()
                    if chosen_slot is None:
                        print("[WARN] Parking full, dropping ENTER")
                    else:
                        if pk:
                            active_plate_to_slot[pk] = chosen_slot
                            plate_pref_slot[pk] = chosen_slot
                            active_id_to_plate[oid] = pk
                        else:
                            active_id_to_plate[oid] = None
                        object_id = f"{chosen_slot:02d}({suffix})" if suffix else f"{chosen_slot:02d}"
                        csv_writer.writerow([ts, object_id, "car", "in", city or "", kana or ""])
                        last_event_f[oid] = frame_idx
                        print(f"[IN ] {ts} object_id={object_id} plate='{text}'")

                # EXIT event (inside -> outside and stable)
                if prev_inside[oid] and (not is_in) and outside_count[oid] >= EXIT_STABLE_FRAMES:
                    slot_to_free = None
                    if pk and pk in active_plate_to_slot:
                        slot_to_free = active_plate_to_slot.pop(pk, None)
                    else:
                        pk2 = active_id_to_plate.get(oid)
                        if pk2 and pk2 in active_plate_to_slot:
                            slot_to_free = active_plate_to_slot.pop(pk2, None)

                    if slot_to_free is not None:
                        object_id = f"{slot_to_free:02d}({suffix})" if suffix else f"{slot_to_free:02d}"
                        csv_writer.writerow([ts, object_id, "car", "out", city or "", kana or ""])
                        last_event_f[oid] = frame_idx
                        slot_pool.release(slot_to_free)
                        print(f"[OUT] {ts} object_id={object_id} plate='{text}'")

                prev_inside[oid] = is_in

            # show & save
            cv2.imshow("Parking (region enter/exit)", frame)
            out.write(frame)

            # key controls for region
            k = cv2.waitKey(1) & 0xFF
            if k == 27:  # ESC
                break
            step = 10
            if k == ord('w'): ry = max(0, ry - step)
            elif k == ord('s'): ry = min(H-1, ry + step)
            elif k == ord('a'): rx = max(0, rx - step)
            elif k == ord('d'): rx = min(W-1, rx + step)
            elif k == ord('['):  # shrink
                rx += step; ry += step; rw -= 2*step; rh -= 2*step
            elif k == ord(']'):  # enlarge
                rx -= step; ry -= step; rw += 2*step; rh += 2*step
            elif k == ord('r'):  # reset
                rx = int(REGION_XYWH_RATIO[0] * W)
                ry = int(REGION_XYWH_RATIO[1] * H)
                rw = int(REGION_XYWH_RATIO[2] * W)
                rh = int(REGION_XYWH_RATIO[3] * H)
            rx, ry, rw, rh = clamp_region(rx, ry, rw, rh, W, H)

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
        print("[INFO] CSV saved to", csv_path)

if __name__ == "__main__":
    main()
