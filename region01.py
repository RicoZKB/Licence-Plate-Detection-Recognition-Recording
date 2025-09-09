# -*- coding: utf-8 -*-
# Region-based plate logger with slot leasing + immediate CSV flush.
# Logs once on ENTER (leases slot 01..10, or recalls previous slot for that plate)
# and once on EXIT (frees slot). Lightweight & real-time focused.

import os, cv2, csv, re
from datetime import datetime
from detections import CarDetection, LicencePlateDetection

# ===================== User settings =====================
USE_CAMERA         = False
INPUT_VIDEO_PATH   = "input_videos/Trim03.mp4"

SLOT_COUNT         = 10
START_ID_AT        = 100

# Region as ratios of frame (x, y, w, h). Move with WASD, resize with [ ]
REGION_XYWH_RATIO  = (0.20, 0.55, 0.60, 0.30)

ENTER_STABLE_FRAMES = 2     # fewer frames = easier to log (faster)
EXIT_STABLE_FRAMES  = 4
MIN_EVENT_GAP_FRAMES= 10

# Performance toggles
USE_CAR_DETECTOR    = False   # True = draw cars (more GPU/CPU)
WRITE_VIDEO         = False   # True = save AVI (slower)
DRAW_BBOXES         = True    # False = skip drawing overlays (faster)
SHOW_FPS            = True

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
    cv2.putText(frame, "ROI: move=WASD resize=[ ]", (rx+6, max(18, ry-6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255), 2, cv2.LINE_AA)

def main():
    # Source
    src = 0 if USE_CAMERA else INPUT_VIDEO_PATH
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {src}")

    # Keep frame small for speed (comment out if you already get 640x384/480)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 384)

    # Detectors (car optional)
    car_det = CarDetection(model_path="yolo11n.pt") if USE_CAR_DETECTOR else None
    lp_det  = LicencePlateDetection(model_path="models/best.pt")

    # Canvas/video
    os.makedirs("output_videos", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 384)
    out = cv2.VideoWriter("output_videos/output_video.avi", fourcc, fps, (W, H)) if WRITE_VIDEO else None

    # CSV
    csv_file, csv_writer, csv_path = open_daily_csv()
    print("[INFO] Logging to:", csv_path)

    # Region pixels
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
    import time
    t_last = time.time(); f_count = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok: break
            frame_idx += 1; f_count += 1

            # Plate detect (fast path)
            all_lp_dets, all_lp_texts = lp_det.detect_frames([frame])
            lp_dets, lp_texts = all_lp_dets[0], all_lp_texts[0]

            # Optional car boxes
            if USE_CAR_DETECTOR:
                car_dets = car_det.detect_frames([frame], read_from_stub=False)[0]
            else:
                car_dets = []

            if DRAW_BBOXES:
                if USE_CAR_DETECTOR:
                    frame = car_det.draw_bboxes([frame], [car_dets])[0]
                frame = lp_det.draw_bboxes([frame], [lp_dets], [lp_texts])[0]
                draw_region(frame, rx, ry, rw, rh)

            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Process plates for region events
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

                pk = plate_key(text or "")
                suf = plate_suffix(text or "")
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
                        csv_file.flush()  # <-- immediate write!
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
                        csv_file.flush()  # <-- immediate write!
                        last_event_f[oid] = frame_idx
                        slot_pool.release(slot_to_free)
                        if DRAW_BBOXES:
                            cv2.putText(frame, f"OUT {object_id}", (int(cx), int(cy)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2, cv2.LINE_AA)

                prev_inside[oid] = is_in

            # Show FPS
            if SHOW_FPS:
                now = time.time()
                if now - t_last >= 0.5:
                    fps_est = f_count / (now - t_last)
                    t_last = now; f_count = 0
                try:
                    cv2.putText(frame, f"FPS ~ {fps_est:.1f}", (10, 24),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
                except:
                    pass

            # Display / write
            cv2.imshow("Parking (region enter/exit)", frame)
            if out is not None: out.write(frame)

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
