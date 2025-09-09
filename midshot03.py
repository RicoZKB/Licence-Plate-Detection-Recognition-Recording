# -*- coding: utf-8 -*-
# region07.py — ROI-cropped, one-shot plate capture + idle motion gate
# - Absolutely no detector/OCR when ROI is idle (cheap motion gate only)
# - Runs detector/OCR only INSIDE the yellow ROI (cropped inference = faster)
# - Logs once on ENTER as soon as a "good" frame appears (sharp + valid suffix)
# - Skips further OCR for that object until EXIT (lightweight while inside)
# - Immediate CSV flush on every write

import os, cv2, csv, re, time
from datetime import datetime
from detections import CarDetection, LicencePlateDetection

# ===================== User settings =====================
USE_CAMERA               = False
INPUT_VIDEO_PATH         = "input_videos/Trim03.mp4"

SLOT_COUNT               = 10
START_ID_AT              = 100

# Region as ratios of frame (x, y, w, h)
REGION_XYWH_RATIO        = (0.32, 0.60, 0.36, 0.20)

# Lock ROI from code (disable WASD/[ ])
LOCK_ROI                 = True

# Stability & gaps
ENTER_STABLE_FRAMES      = 1       # lock early
EXIT_STABLE_FRAMES       = 4
MIN_EVENT_GAP_FRAMES     = 10

# One-shot window (fallback if sharpness never crosses threshold)
CAPTURE_WINDOW_FRAMES    = 6

# While car remains inside, only process heavy every N frames
INSIDE_PROCESS_EVERY_N   = 8

# One-shot accept rules
PLATE_SUFFIX_RE          = re.compile(r"([0-9]{2,3}-[0-9]{2})")
MIN_SHARPNESS_LOCK       = 60.0     # Laplacian variance to accept immediately

# Performance toggles
USE_CAR_DETECTOR         = False    # plates only
WRITE_VIDEO              = False
DRAW_BBOXES              = True
SHOW_FPS                 = True
TARGET_WIDTH             = 640
BOX_THICK                = 1

# ---- Idle motion gate (skip all NN work when ROI is still) ----
IDLE_MOTION_GATE         = True     # True = use cheap motion detector when ROI empty
MOTION_DOWNSCALE         = 0.25     # resize ROI before motion diff (0.25 = quarter-size)
MOTION_BLUR_KSIZE        = 5        # Gaussian blur kernel (noise suppression)
MOTION_THRESH            = 6.0      # mean absdiff threshold to "wake" detector
IDLE_MIN_GAP_FRAMES      = 8        # after a negative check, wait this many frames
# =========================================================

# Reduce OpenCV overhead a bit
cv2.setUseOptimized(True)
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")
os.environ.setdefault("FLAGS_minloglevel", "2")  # quiet some runtimes

CITY_KANA_RE = re.compile(u"^\\s*([^\\s\\d]+)\\s+[0-9]{3,4}\\s*([ぁ-ゖァ-ヿ一-龯A-Za-z])", re.UNICODE)

def parse_city_kana(text):
    if not text: return None, None
    m = CITY_KANA_RE.search(text)
    return (m.group(1), m.group(2)) if m else (None, None)

def plate_suffix(text):
    if not text: return None
    m = PLATE_SUFFIX_RE.search(text)
    return m.group(1) if m else None

def plate_key(text):
    suf = plate_suffix(text or "");  city, _ = parse_city_kana(text or "")
    if not suf: return None
    return f"{city if city else '?'}|{suf}"

def open_daily_csv():
    date_str = datetime.now().strftime("%Y%m%d")
    os.makedirs("logs", exist_ok=True)
    path = os.path.join("logs", f"parking_log_{date_str}.csv")
    new = not os.path.exists(path)
    f = open(path, "a", newline="", encoding="utf-8", buffering=1)  # line-buffered
    w = csv.writer(f)
    if new:
        w.writerow(["timestamp","object_id","vehicle_type","direction","city","kana"])
        f.flush()
    return f, w, path

def write_row_flush(w, f, row):
    w.writerow(row); f.flush()
    try: os.fsync(f.fileno())
    except Exception: pass

# ---------- geometry helpers ----------
def xyxy_from_det(det):
    if isinstance(det, (list, tuple)) and len(det) >= 4:
        return float(det[0]), float(det[1]), float(det[2]), float(det[3])
    if isinstance(det, dict):
        box = det.get("bbox") or det.get("xyxy")
        if box and len(box) >= 4:
            return float(box[0]), float(box[1]), float(box[2]), float(box[3])
    return None

def offset_det(det, ox, oy):
    """Shift a detection back to full-frame coords after ROI crop."""
    if isinstance(det, (list, tuple)) and len(det) >= 4:
        x1,y1,x2,y2 = det[:4]
        return [x1+ox, y1+oy, x2+ox, y2+oy]
    if isinstance(det, dict):
        box = det.get("bbox") or det.get("xyxy")
        if box and len(box) >= 4:
            box = [box[0]+ox, box[1]+oy, box[2]+ox, box[3]+oy]
            if "bbox" in det: det["bbox"] = box
            else: det["xyxy"] = box
            return det
    return det

def center_from_det(det):
    xy = xyxy_from_det(det)
    if not xy: return None, None
    x1,y1,x2,y2 = xy
    return (x1+x2)/2.0, (y1+y2)/2.0

def bbox_area(det):
    xy = xyxy_from_det(det)
    if not xy: return 0.0
    x1,y1,x2,y2 = xy
    return max(0, x2-x1) * max(0, y2-y1)

def variance_of_laplacian(img):
    if img is None or img.size == 0: return 0.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape)==3 else img
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def crop_frame(frame, rx, ry, rw, rh):
    return frame[ry:ry+rh, rx:rx+rw]

def motion_metric(prev_small_gray, cur_small_gray):
    diff = cv2.absdiff(prev_small_gray, cur_small_gray)
    return float(diff.mean())

# ---------- ID / slots ----------
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

# ---------- region utils ----------
def inside_region(cx, cy, rx, ry, rw, rh):
    return (cx is not None) and (cy is not None) and (rx <= cx <= rx+rw) and (ry <= cy <= ry+rh)

def clamp_region(rx, ry, rw, rh, W, H):
    rx = max(0, min(rx, W-10)); ry = max(0, min(ry, H-10))
    rw = max(20, min(rw, W-rx)); rh = max(20, min(rh, H-ry))
    return rx, ry, rw, rh

def draw_region(frame, rx, ry, rw, rh):
    cv2.rectangle(frame, (rx, ry), (rx+rw, ry+rh), (0,255,255), BOX_THICK)
    cv2.putText(frame, "ROI (locked)" if LOCK_ROI else "ROI: WASD move, [ ] resize, r reset",
                (rx+6, max(18, ry-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255), 2, cv2.LINE_AA)

def draw_lp_boxes(frame, dets, color=(0,255,0)):
    for d in dets:
        xy = xyxy_from_det(d)
        if not xy: continue
        x1,y1,x2,y2 = map(int, xy)
        cv2.rectangle(frame, (x1,y1), (x2,y2), color, BOX_THICK)

# ---------- main ----------
def main():
    src = 0 if USE_CAMERA else INPUT_VIDEO_PATH
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {src}")

    if USE_CAMERA:
        try: cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # keep camera frames fresh
        except: pass

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

    # ROI
    rx = int(REGION_XYWH_RATIO[0]*W); ry = int(REGION_XYWH_RATIO[1]*H)
    rw = int(REGION_XYWH_RATIO[2]*W); rh = int(REGION_XYWH_RATIO[3]*H)
    rx, ry, rw, rh = clamp_region(rx, ry, rw, rh, W, H)

    idbank = IDBank(START_ID_AT)
    prev_inside = {}
    inside_count, outside_count, last_event_f = {}, {}, {}

    slot_pool = SlotPool(SLOT_COUNT)
    plate_pref_slot, active_plate_to_slot, active_id_to_plate = {}, {}, {}

    # one-shot bookkeeping
    capture_open = {}    # oid -> frames left (0 close, -1 locked)
    best_score  = {}     # oid -> area*sharpness (fallback ranking)
    best_text   = {}     # oid -> best OCR so far
    best_det    = {}     # oid -> det for best
    captured_in = set()

    # Idle motion gate state
    last_roi_small_gray = None
    idle_cooldown = 0

    frame_idx = 0
    t0 = time.time(); fcount = 0; fps_est = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok: break
            frame_idx += 1; fcount += 1

            # --- ROI crop for inference only ---
            rx, ry, rw, rh = clamp_region(rx, ry, rw, rh, W, H)
            roi = crop_frame(frame, rx, ry, rw, rh)

            # Is anyone currently inside?
            someone_inside = any(prev_inside.get(oid, False) for oid in prev_inside)

            # Decide if heavy work should run this frame
            do_heavy_this_frame = False

            if someone_inside:
                have_locked_inside = any((oid in captured_in) and prev_inside.get(oid, False)
                                         for oid in prev_inside)
                # follow inside throttling
                do_heavy_this_frame = (not have_locked_inside) or (frame_idx % INSIDE_PROCESS_EVERY_N == 0)
            else:
                if IDLE_MOTION_GATE:
                    # tiny motion gate
                    small = cv2.resize(roi, (0,0), fx=MOTION_DOWNSCALE, fy=MOTION_DOWNSCALE,
                                       interpolation=cv2.INTER_AREA)
                    small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                    if MOTION_BLUR_KSIZE > 1:
                        small = cv2.GaussianBlur(small, (MOTION_BLUR_KSIZE, MOTION_BLUR_KSIZE), 0)

                    if idle_cooldown > 0:
                        idle_cooldown -= 1
                        do_heavy_this_frame = False
                    else:
                        if last_roi_small_gray is None:
                            do_heavy_this_frame = True  # warmup sniff
                        else:
                            m = motion_metric(last_roi_small_gray, small)
                            do_heavy_this_frame = (m >= MOTION_THRESH)
                            if not do_heavy_this_frame:
                                idle_cooldown = IDLE_MIN_GAP_FRAMES
                    last_roi_small_gray = small
                else:
                    do_heavy_this_frame = False  # truly nothing when empty

            lp_dets_full, lp_texts = [], []

            if do_heavy_this_frame:
                # detect only on ROI, then shift boxes back to full-frame coords
                all_lp_dets, all_lp_texts = lp_det.detect_frames([roi])
                roi_dets = all_lp_dets[0]; lp_texts = all_lp_texts[0]

                lp_dets_full = [offset_det(d, rx, ry) for d in roi_dets]

            # Optional car boxes (also cropped for speed)
            car_dets_full = []
            if USE_CAR_DETECTOR and do_heavy_this_frame:
                car_dets = car_det.detect_frames([roi], read_from_stub=False)[0]
                car_dets_full = [offset_det(d, rx, ry) for d in car_dets]

            # Draw overlays
            if DRAW_BBOXES and do_heavy_this_frame:
                draw_lp_boxes(frame, lp_dets_full, color=(0,255,0))
                if USE_CAR_DETECTOR:
                    draw_lp_boxes(frame, car_dets_full, color=(255,200,0))
            if DRAW_BBOXES:
                draw_region(frame, rx, ry, rw, rh)

            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Process detections (only when we actually ran them)
            if do_heavy_this_frame:
                for det, text in zip(lp_dets_full, lp_texts):
                    key = stable_key(text, det)
                    if key is None: continue
                    oid = idbank.get(key)

                    cx, cy = center_from_det(det)
                    is_in = inside_region(cx, cy, rx, ry, rw, rh)

                    # init
                    if oid not in prev_inside:
                        prev_inside[oid] = is_in
                        inside_count[oid]  = 1 if is_in else 0
                        outside_count[oid] = 0 if is_in else 1
                        if is_in:
                            capture_open[oid] = CAPTURE_WINDOW_FRAMES
                            best_score[oid] = 0.0; best_text[oid] = ""; best_det[oid] = det
                        continue

                    # streaks
                    if is_in:
                        inside_count[oid]  = inside_count.get(oid,0) + 1
                        outside_count[oid] = 0
                    else:
                        outside_count[oid] = outside_count.get(oid,0) + 1
                        inside_count[oid]  = 0

                    # skip anything already locked (-1)
                    if capture_open.get(oid, 0) == -1:
                        prev_inside[oid] = is_in
                        continue

                    # one-shot capture maintenance
                    if is_in and (oid not in captured_in):
                        # rank by area*sharpness as fallback
                        if capture_open.get(oid, 0) > 0:
                            x1,y1,x2,y2 = map(int, xyxy_from_det(det))
                            y1c, y2c = max(0,y1), min(H,y2)
                            x1c, x2c = max(0,x1), min(W,x2)
                            plate_img = frame[y1c:y2c, x1c:x2c]
                            sharp = variance_of_laplacian(plate_img)
                            score = bbox_area(det) * max(1.0, sharp)

                            if score > best_score.get(oid, 0.0) and plate_suffix(text or ""):
                                best_score[oid] = score
                                best_text[oid]  = text
                                best_det[oid]   = det

                            if plate_suffix(text or "") and sharp >= MIN_SHARPNESS_LOCK:
                                capture_open[oid] = 0
                            else:
                                capture_open[oid] -= 1

                        # log IN once window closes
                        if capture_open.get(oid, 0) == 0 and inside_count[oid] >= ENTER_STABLE_FRAMES:
                            chosen_text = best_text.get(oid,"") or text or ""
                            pk = plate_key(chosen_text)
                            suf = plate_suffix(chosen_text)
                            city, kana = parse_city_kana(chosen_text)

                            chosen_slot = None
                            if pk and pk in plate_pref_slot:
                                got = slot_pool.try_acquire_specific(plate_pref_slot[pk])
                                if got is not None: chosen_slot = got
                            if chosen_slot is None:
                                chosen_slot = slot_pool.acquire_lowest()

                            if chosen_slot is not None:
                                if pk:
                                    active_plate_to_slot[pk] = chosen_slot
                                    plate_pref_slot[pk] = chosen_slot
                                    active_id_to_plate[oid] = pk
                                else:
                                    active_id_to_plate[oid] = None

                                object_id = f"{chosen_slot:02d}({suf})" if suf else f"{chosen_slot:02d}"
                                write_row_flush(csv_writer, csv_file, [ts, object_id, "car", "in", city or "", kana or ""])
                                captured_in.add(oid)
                                last_event_f[oid] = frame_idx
                                capture_open[oid] = -1  # LOCKED
                                if DRAW_BBOXES:
                                    cv2.putText(frame, f"IN {object_id}", (int(cx), int(cy)),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv2.LINE_AA)

                    # EXIT handling
                    if prev_inside[oid] and (not is_in) and outside_count[oid] >= EXIT_STABLE_FRAMES:
                        if frame_idx - last_event_f.get(oid, -10**9) >= MIN_EVENT_GAP_FRAMES:
                            pk_now = active_id_to_plate.get(oid)
                            slot_to_free = None
                            if pk_now and pk_now in active_plate_to_slot:
                                slot_to_free = active_plate_to_slot.pop(pk_now, None)

                            if slot_to_free is not None:
                                chosen_text = best_text.get(oid,"") or text or ""
                                suf_out = plate_suffix(chosen_text)
                                city, kana = parse_city_kana(chosen_text)
                                object_id = f"{slot_to_free:02d}({suf_out})" if suf_out else f"{slot_to_free:02d}"
                                write_row_flush(csv_writer, csv_file, [ts, object_id, "car", "out", city or "", kana or ""])
                                slot_pool.release(slot_to_free)
                                last_event_f[oid] = frame_idx
                                if DRAW_BBOXES:
                                    cv2.putText(frame, f"OUT {object_id}", (int(cx), int(cy)),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv2.LINE_AA)

                            # clear state so future re-entry will capture again
                            for dct in (capture_open, best_score, best_text, best_det):
                                dct.pop(oid, None)
                            if oid in captured_in: captured_in.remove(oid)

                    prev_inside[oid] = is_in

            # FPS overlay
            if SHOW_FPS:
                now = time.time()
                if now - t0 >= 0.5:
                    fps_est = fcount / (now - t0); t0 = now; fcount = 0
                tag = "" if do_heavy_this_frame else "  (idle)"
                cv2.putText(frame, f"FPS ~ {fps_est:.1f}{tag}", (10, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

            # Display / write
            cv2.imshow("Parking (region07 ROI one-shot + motion gate)", frame)
            if out is not None and WRITE_VIDEO: out.write(frame)

            # optional ROI keyboard control if not locked
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
