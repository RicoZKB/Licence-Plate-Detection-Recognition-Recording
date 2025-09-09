# -*- coding: utf-8 -*-
# region_midshot_roi.py — One-shot at ROI middle, ROI-only scanning, outside throttled.

import os, cv2, csv, re, time
from datetime import datetime
from detections import CarDetection, LicencePlateDetection

# ===================== User settings =====================
USE_CAMERA           = False
INPUT_VIDEO_PATH     = "input_videos/Trim03.mp4"

SLOT_COUNT           = 10
START_ID_AT          = 100

# ROI (x, y, w, h) in frame ratios
REGION_XYWH_RATIO    = (0.32, 0.60, 0.36, 0.20)
LOCK_ROI             = True          # block WASD / [ ] if True
TARGET_WIDTH         = 640

# “Middle” capture band inside ROI (0..1 of ROI size)
MID_BAND_RATIO_X     = 0.30
MID_BAND_RATIO_Y     = 0.30

# Scan throttling
SCAN_OUTSIDE_EVERY_N = 5             # when nothing inside ROI, scan every N frames
INSIDE_SCAN_EVERY_N  = 1             # while waiting to hit center (usually 1)

# OCR acceptance
PLATE_SUFFIX_RE      = re.compile(r"([0-9]{2,3}-[0-9]{2})")
MIN_SHARPNESS        = 40.0

# Events
ENTER_STABLE_FRAMES  = 1
EXIT_STABLE_FRAMES   = 4
MIN_EVENT_GAP_FRAMES = 10

# UI / perf
USE_CAR_DETECTOR     = False
WRITE_VIDEO          = False
DRAW_BBOXES          = True
SHOW_FPS             = True
BOX_THICK            = 1
ROI_GUARD_RATIO      = 0.05          # add 5% margin around ROI for the crop
# =========================================================

cv2.setUseOptimized(True)
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")
os.environ.setdefault("FLAGS_minloglevel", "2")

CITY_KANA_RE = re.compile(u"^\\s*([^\\s\\d]+)\\s+[0-9]{3,4}\\s*([ぁ-ゖァ-ヿ一-龯A-Za-z])", re.UNICODE)
def parse_city_kana(t):
    if not t: return None, None
    m = CITY_KANA_RE.search(t);  return (m.group(1), m.group(2)) if m else (None, None)
def plate_suffix(t):
    if not t: return None
    m = PLATE_SUFFIX_RE.search(t);  return m.group(1) if m else None
def plate_key(t):
    suf = plate_suffix(t or "");  city, _ = parse_city_kana(t or "")
    if not suf: return None
    return f"{city if city else '?'}|{suf}"

def open_daily_csv():
    date_str = datetime.now().strftime("%Y%m%d")
    os.makedirs("logs", exist_ok=True)
    path = os.path.join("logs", f"parking_log_{date_str}.csv")
    new = not os.path.exists(path)
    f = open(path, "a", newline="", encoding="utf-8", buffering=1)
    w = csv.writer(f)
    if new:
        w.writerow(["timestamp","object_id","vehicle_type","direction","city","kana"])
        f.flush()
    return f, w, path

def write_row_flush(w, f, row):
    w.writerow(row); f.flush()
    try: os.fsync(f.fileno())
    except Exception: pass

# ---------- geom ----------
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
def variance_of_laplacian(img):
    if img is None or img.size == 0: return 0.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape)==3 else img
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())
def inside_region(cx, cy, rx, ry, rw, rh):
    return (cx is not None) and (cy is not None) and (rx <= cx <= rx+rw) and (ry <= cy <= ry+rh)
def clamp_region(rx, ry, rw, rh, W, H):
    rx = max(0, min(rx, W-10)); ry = max(0, min(ry, H-10))
    rw = max(20, min(rw, W-rx)); rh = max(20, min(rh, H-ry))
    return rx, ry, rw, rh
def draw_region(frame, rx, ry, rw, rh):
    cv2.rectangle(frame, (rx, ry), (rx+rw, ry+rh), (0,255,255), BOX_THICK)
    msg = "ROI (locked)" if LOCK_ROI else "ROI: WASD move, [ ] resize, r reset"
    cv2.putText(frame, msg, (rx+6, max(18, ry-6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255), 2, cv2.LINE_AA)
def draw_midband(frame, rx, ry, rw, rh):
    cx = rx + rw//2; cy = ry + rh//2
    bw = int(rw * MID_BAND_RATIO_X / 2.0)
    bh = int(rh * MID_BAND_RATIO_Y / 2.0)
    cv2.rectangle(frame, (cx-bw, cy-bh), (cx+bw, cy+bh), (0,255,0), BOX_THICK)

# ---------- id/slots ----------
class IDBank:
    def __init__(self, start_at=100):
        self._map = {}
        self._next = int(start_at)

    def get(self, key):
        if key not in self._map:
            self._map[key]=self._next; self._next+=1
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

# ---------- main ----------
def main():
    src = 0 if USE_CAMERA else INPUT_VIDEO_PATH
    cap = cv2.VideoCapture(src)
    if not cap.isOpened(): raise RuntimeError(f"Could not open video source: {src}")
    try: cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)
    except: pass

    car_det = CarDetection(model_path="yolo11n.pt") if USE_CAR_DETECTOR else None
    lp_det  = LicencePlateDetection(model_path="models/best.pt")

    os.makedirs("output_videos", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or TARGET_WIDTH)
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or int(TARGET_WIDTH*3/5))
    out = cv2.VideoWriter("output_videos/output_video.avi", fourcc, fps, (W, H)) if WRITE_VIDEO else None

    csv_file, csv_writer, csv_path = open_daily_csv()
    print("[INFO] Logging to:", csv_path)

    rx = int(REGION_XYWH_RATIO[0]*W); ry = int(REGION_XYWH_RATIO[1]*H)
    rw = int(REGION_XYWH_RATIO[2]*W); rh = int(REGION_XYWH_RATIO[3]*H)
    rx, ry, rw, rh = clamp_region(rx, ry, rw, rh, W, H)

    idbank = IDBank(START_ID_AT)
    prev_inside = {}
    inside_count, outside_count, last_event_f = {}, {}, {}

    slot_pool = SlotPool(SLOT_COUNT)
    plate_pref_slot, active_plate_to_slot, active_id_to_plate = {}, {}, {}

    waiting_mid = set()   # entered, waiting for mid
    captured_in = set()   # already logged IN

    frame_idx = 0
    t0 = time.time(); fcount = 0; fps_est = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok: break
            frame_idx += 1; fcount += 1

            # ---------- ROI-only crop (with small guard margin) ----------
            gx = int(rx - rw * ROI_GUARD_RATIO); gy = int(ry - rh * ROI_GUARD_RATIO)
            gw = int(rw * (1.0 + 2*ROI_GUARD_RATIO)); gh = int(rh * (1.0 + 2*ROI_GUARD_RATIO))
            gx = max(0, gx); gy = max(0, gy)
            gw = min(gw, W-gx); gh = min(gh, H-gy)
            roi = frame[gy:gy+gh, gx:gx+gw]

            # Decide if we scan this frame (throttle when nothing inside)
            any_inside = any(prev_inside.get(oid, False) for oid in prev_inside.keys())
            every_n = INSIDE_SCAN_EVERY_N if any_inside else SCAN_OUTSIDE_EVERY_N
            do_scan = (frame_idx % max(1, every_n) == 0)

            lp_dets, lp_texts = [], []
            if do_scan:
                all_lp_dets, all_lp_texts = lp_det.detect_frames([roi])
                # remap dets back to full-frame coords
                rdets = []
                for d in all_lp_dets[0]:
                    x1,y1,x2,y2 = map(int, xyxy_from_det(d))
                    rdets.append([x1+gx, y1+gy, x2+gx, y2+gy])
                lp_dets, lp_texts = rdets, all_lp_texts[0]

            # overlays
            if DRAW_BBOXES:
                cv2.rectangle(frame, (rx, ry), (rx+rw, ry+rh), (0,255,255), BOX_THICK)
                cx = rx + rw//2; cy = ry + rh//2
                bw = int(rw * MID_BAND_RATIO_X / 2.0); bh = int(rh * MID_BAND_RATIO_Y / 2.0)
                cv2.rectangle(frame, (cx-bw, cy-bh), (cx+bw, cy+bh), (0,255,0), BOX_THICK)
                if do_scan and lp_dets:
                    for d in lp_dets:
                        x1,y1,x2,y2 = map(int, d)
                        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 1)

            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            if do_scan:
                for det, text in zip(lp_dets, lp_texts):
                    key = stable_key(text, det)
                    if key is None: continue
                    oid = idbank.get(key)

                    cxp, cyp = center_from_det(det)
                    is_in = inside_region(cxp, cyp, rx, ry, rw, rh)

                    if oid not in prev_inside:
                        prev_inside[oid] = is_in
                        inside_count[oid]  = 1 if is_in else 0
                        outside_count[oid] = 0 if is_in else 1
                        if is_in and oid not in captured_in: waiting_mid.add(oid)
                        continue

                    if is_in:
                        inside_count[oid]  = inside_count.get(oid,0) + 1
                        outside_count[oid] = 0
                    else:
                        outside_count[oid] = outside_count.get(oid,0) + 1
                        inside_count[oid]  = 0

                    # ENTER → wait for mid
                    if (not prev_inside[oid]) and is_in and inside_count[oid] >= ENTER_STABLE_FRAMES:
                        if oid not in captured_in: waiting_mid.add(oid)

                    # middle band check
                    if is_in and (oid in waiting_mid) and (oid not in captured_in):
                        mid_cx = rx + rw/2.0; mid_cy = ry + rh/2.0
                        bx = rw * MID_BAND_RATIO_X / 2.0
                        by = rh * MID_BAND_RATIO_Y / 2.0
                        if abs(cxp - mid_cx) <= bx and abs(cyp - mid_cy) <= by:
                            x1,y1,x2,y2 = map(int, xyxy_from_det(det))
                            plate_img = frame[max(0,y1):min(H,y2), max(0,x1):min(W,x2)]
                            sharp = variance_of_laplacian(plate_img)
                            suf = plate_suffix(text or "")
                            if suf and sharp >= MIN_SHARPNESS:
                                pk = plate_key(text or "")
                                city, kana = parse_city_kana(text or "")
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
                                    object_id = f"{chosen:02d}({suf})"
                                    write_row_flush(csv_writer, csv_file,
                                                    [ts, object_id, "car", "in", city or "", kana or ""])
                                    last_event_f[oid] = frame_idx
                                    captured_in.add(oid); waiting_mid.discard(oid)
                                    if DRAW_BBOXES:
                                        cv2.putText(frame, f"IN {object_id}", (int(cxp), int(cyp)),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv2.LINE_AA)

                    # EXIT
                    if prev_inside[oid] and (not is_in) and outside_count[oid] >= EXIT_STABLE_FRAMES:
                        if frame_idx - last_event_f.get(oid, -10**9) >= MIN_EVENT_GAP_FRAMES:
                            pk_now = active_id_to_plate.get(oid)
                            slot_to_free = None
                            if pk_now and pk_now in active_plate_to_slot:
                                slot_to_free = active_plate_to_slot.pop(pk_now, None)
                            if slot_to_free is not None:
                                suf_out = plate_suffix(text or "")
                                city, kana = parse_city_kana(text or "")
                                object_id = f"{slot_to_free:02d}({suf_out})" if suf_out else f"{slot_to_free:02d}"
                                write_row_flush(csv_writer, csv_file,
                                                [ts, object_id, "car", "out", city or "", kana or ""])
                                slot_pool.release(slot_to_free)
                                last_event_f[oid] = frame_idx
                                if DRAW_BBOXES:
                                    cv2.putText(frame, f"OUT {object_id}", (int(cxp), int(cyp)),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv2.LINE_AA)
                        waiting_mid.discard(oid); captured_in.discard(oid)

                    prev_inside[oid] = is_in

            # FPS
            if SHOW_FPS:
                now = time.time()
                if now - t0 >= 0.5:
                    fps_est = fcount / (now - t0); t0 = now; fcount = 0
                cv2.putText(frame, f"FPS ~ {fps_est:.1f}  "
                                   f"{'(outside x%df)'%SCAN_OUTSIDE_EVERY_N if not any_inside else ''}",
                            (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

            cv2.imshow("Parking (mid-shot ROI only)", frame)
            if WRITE_VIDEO and out is not None: out.write(frame)

            k = cv2.waitKey(1) & 0xFF
            if k == 27: break
            if not LOCK_ROI:
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
