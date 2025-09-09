# -*- coding: utf-8 -*-
# region07.py — ROI-cropped, one-shot plate capture + motion-trigger + resizable ROI
# - Heavy work only when ROI shows motion (no object = no scan)
# - Detect/OCR only INSIDE the yellow ROI (cropped inference = faster)
# - One-shot capture when a good frame appears near mid-band
# - After logging "IN", skip/throttle OCR until EXIT
# - Immediate CSV flush

import os, cv2, csv, re, time
from datetime import datetime
from detections import LicencePlateDetection  # (only plates)

# ===================== User settings =====================
USE_CAMERA                = False
INPUT_VIDEO_PATH          = "input_videos/Trim03.mp4"

# ROI as ratios of frame (x, y, w, h)
REGION_XYWH_RATIO         = (0.32, 0.60, 0.36, 0.20)

# Lock ROI from code (set False to allow mouse resize/move)
LOCK_ROI                  = False

# One-shot acceptance
PLATE_SUFFIX_RE           = re.compile(r"([0-9]{2,3}-[0-9]{2})")
MIN_SHARPNESS_LOCK        = 60.0      # Laplacian variance to accept immediately
CAPTURE_WINDOW_FRAMES     = 6         # fallback window to pick best if never sharp
MID_BAND_RATIO_Y          = 0.50      # "middle" band (0..1) within ROI

# While inside (after we've logged), do heavy work every N frames only
INSIDE_PROCESS_EVERY_N    = 8

# Exit/enter stability
ENTER_STABLE_FRAMES       = 1
EXIT_STABLE_FRAMES        = 4
MIN_EVENT_GAP_FRAMES      = 10

# Motion trigger (to avoid scanning when nothing is happening)
MOTION_DIFF_THRESHOLD     = 4.0       # mean abs diff (0..255); higher = less sensitive
MOTION_COOLDOWN_FRAMES    = 2         # run heavy for a few frames after motion seen

# I/O & draw
WRITE_VIDEO               = False
DRAW_BBOXES               = True
SHOW_FPS                  = True
TARGET_WIDTH              = 640
BOX_THICK                 = 1
# =========================================================

# Reduce OpenCV noise a bit
cv2.setUseOptimized(True)
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")
os.environ.setdefault("FLAGS_minloglevel", "2")


# -------------- text helpers --------------
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

# -------------- CSV --------------
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

# -------------- geometry --------------
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

def bbox_area(det):
    xy = xyxy_from_det(det)
    if not xy: return 0.0
    x1,y1,x2,y2 = xy
    return max(0, x2-x1) * max(0, y2-y1)

def clamp_region(rx, ry, rw, rh, W, H):
    rx = max(0, min(rx, W-10)); ry = max(0, min(ry, H-10))
    rw = max(20, min(rw, W-rx)); rh = max(20, min(rh, H-ry))
    return rx, ry, rw, rh

def crop_frame(frame, rx, ry, rw, rh):
    return frame[ry:ry+rh, rx:rx+rw]

def variance_of_laplacian(img):
    if img is None or img.size == 0: return 0.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape)==3 else img
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

# -------------- drawing --------------
def draw_region(frame, rx, ry, rw, rh):
    cv2.rectangle(frame, (rx, ry), (rx+rw, ry+rh), (0,255,255), BOX_THICK)
    label = "ROI (locked)" if LOCK_ROI else "ROI: drag to move • drag edges to resize"
    cv2.putText(frame, label, (rx+6, max(18, ry-6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255), 2, cv2.LINE_AA)

def draw_lp_boxes(frame, dets, color=(0,255,0)):
    for d in dets:
        xy = xyxy_from_det(d)
        if not xy: continue
        x1,y1,x2,y2 = map(int, xy)
        cv2.rectangle(frame, (x1,y1), (x2,y2), color, BOX_THICK)

# -------------- main --------------
def main():
    src = 0 if USE_CAMERA else INPUT_VIDEO_PATH
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {src}")

    if USE_CAMERA:
        try: cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except: pass

    try: cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)
    except: pass

    lp_det = LicencePlateDetection(model_path="models/best.pt")

    os.makedirs("output_videos", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or TARGET_WIDTH)
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or int(TARGET_WIDTH*3/5))
    out = cv2.VideoWriter("output_videos/output_video.avi", fourcc, fps, (W, H)) if WRITE_VIDEO else None

    csv_file, csv_writer, csv_path = open_daily_csv()
    print("[INFO] Logging to:", csv_path)

    # ROI init
    rx = int(REGION_XYWH_RATIO[0]*W); ry = int(REGION_XYWH_RATIO[1]*H)
    rw = int(REGION_XYWH_RATIO[2]*W); rh = int(REGION_XYWH_RATIO[3]*H)
    rx, ry, rw, rh = clamp_region(rx, ry, rw, rh, W, H)

    # One-shot bookkeeping per logical object (simplified to ROI session)
    capture_open = 0          # countdown frames to try capture (0=closed, -1=locked)
    best_score  = 0.0
    best_text   = ""
    best_det    = None
    captured_in = False
    last_event_f = -10**9

    # Motion trigger state (to avoid heavy work when nothing happening)
    prev_roi_gray = None
    motion_boost = 0  # allow heavy scans for a few frames after motion

    # FPS
    frame_idx = 0
    t0 = time.time(); fcount = 0; fps_est = 0.0

    # ---- ROI: mouse-resizable (state enclosed in main) -----------------------
    win = "Parking (region07 resizable)"
    cv2.namedWindow(win)

    _drag_mode   = None        # one of: move,l,r,t,b,tl,tr,bl,br
    _drag_anchor = (0, 0)      # last mouse position (ax, ay)
    EDGE = 12                  # px hit margin for edges/corners

    def _hit_test(rx, ry, rw, rh, x, y):
        inside = (rx <= x <= rx+rw) and (ry <= y <= ry+rh)
        if not inside: return None
        left   = abs(x - rx)      <= EDGE
        right  = abs(x - (rx+rw)) <= EDGE
        top    = abs(y - ry)      <= EDGE
        bottom = abs(y - (ry+rh)) <= EDGE
        if left and top: return "tl"
        if right and top: return "tr"
        if left and bottom: return "bl"
        if right and bottom: return "br"
        if left: return "l"
        if right: return "r"
        if top: return "t"
        if bottom: return "b"
        return "move"

    def _apply_drag(mode, rx, ry, rw, rh, x, y, ax, ay, W, H):
        dx, dy = int(x - ax), int(y - ay)
        x2, y2, w2, h2 = rx, ry, rw, rh
        if mode == "move": x2 += dx; y2 += dy
        elif mode == "l":  x2 += dx; w2 -= dx
        elif mode == "r":  w2 += dx
        elif mode == "t":  y2 += dy; h2 -= dy
        elif mode == "b":  h2 += dy
        elif mode == "tl": x2 += dx; w2 -= dx; y2 += dy; h2 -= dy
        elif mode == "tr": w2 += dx; y2 += dy; h2 -= dy
        elif mode == "bl": x2 += dx; w2 -= dx; h2 += dy
        elif mode == "br": w2 += dx; h2 += dy
        return clamp_region(x2, y2, w2, h2, W, H)

    def on_mouse(event, x, y, flags, userdata):
        nonlocal rx, ry, rw, rh, _drag_mode, _drag_anchor
        if LOCK_ROI:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            mode = _hit_test(rx, ry, rw, rh, x, y)
            if mode:
                _drag_mode = mode
                _drag_anchor = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and _drag_mode:
            ax, ay = _drag_anchor
            rx, ry, rw, rh = _apply_drag(_drag_mode, rx, ry, rw, rh, x, y, ax, ay, W, H)
            _drag_anchor = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            _drag_mode = None

    cv2.setMouseCallback(win, on_mouse)
    # -------------------------------------------------------------------------

    try:
        while True:
            ok, frame = cap.read()
            if not ok: break
            frame_idx += 1; fcount += 1

            # ROI crop
            rx, ry, rw, rh = clamp_region(rx, ry, rw, rh, W, H)
            roi = crop_frame(frame, rx, ry, rw, rh)

            # --------- Motion trigger (cheap) ----------
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            do_heavy = False
            if prev_roi_gray is None:
                prev_roi_gray = roi_gray.copy()
            else:
                diff = cv2.absdiff(roi_gray, prev_roi_gray)
                mean_diff = float(diff.mean())
                prev_roi_gray = roi_gray.copy()

                if mean_diff >= MOTION_DIFF_THRESHOLD:
                    motion_boost = MOTION_COOLDOWN_FRAMES
                    do_heavy = True
                elif motion_boost > 0:
                    motion_boost -= 1
                    do_heavy = True
                else:
                    do_heavy = False

            # While a plate is already captured and inside, throttle further work
            if captured_in and (frame_idx % INSIDE_PROCESS_EVERY_N != 0):
                do_heavy = False

            lp_dets_full, lp_texts = [], []
            if do_heavy:
                # detect only on ROI; shift boxes back to full-frame
                all_lp_dets, all_lp_texts = lp_det.detect_frames([roi])
                roi_dets = all_lp_dets[0]; lp_texts = all_lp_texts[0]
                # offset back
                for d in roi_dets:
                    if isinstance(d, dict) and "bbox" in d:
                        x1,y1,x2,y2 = d["bbox"]
                        d["bbox"] = [x1+rx, y1+ry, x2+rx, y2+ry]
                    elif isinstance(d, (list, tuple)) and len(d)>=4:
                        x1,y1,x2,y2 = d[:4]
                        lp_dets_full.append([x1+rx, y1+ry, x2+rx, y2+ry])
                        continue
                    lp_dets_full.append(d)

            # Draw overlays
            if DRAW_BBOXES and do_heavy:
                draw_lp_boxes(frame, lp_dets_full, color=(0,255,0))
            if DRAW_BBOXES:
                draw_region(frame, rx, ry, rw, rh)
                # small mid band guide
                cy = int(ry + rh*MID_BAND_RATIO_Y)
                cv2.line(frame, (rx, cy), (rx+rw, cy), (0,255,0), 1)

            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # ---------- One-shot plate capture -----------
            if do_heavy:
                for det, text in zip(lp_dets_full, lp_texts):
                    cx, cy = center_from_det(det)
                    in_roi = (cx is not None) and (rx <= cx <= rx+rw and ry <= cy <= ry+rh)

                    # open capture window when we see a plate near the mid band
                    if (not captured_in) and in_roi:
                        band_y = ry + rh*MID_BAND_RATIO_Y
                        if abs(cy - band_y) <= rh*0.20:  # ~middle 40% tall
                            if capture_open == 0:
                                capture_open = CAPTURE_WINDOW_FRAMES
                                best_score = 0.0; best_text = ""; best_det = det

                    # maintain window if open
                    if capture_open > 0 and in_roi:
                        x1,y1,x2,y2 = map(int, xyxy_from_det(det))
                        plate_img = frame[max(0,y1):min(H,y2), max(0,x1):min(W,x2)]
                        sharp = variance_of_laplacian(plate_img)
                        score = bbox_area(det) * max(1.0, sharp)

                        # keep best only if suffix present (e.g., 70-50)
                        if plate_suffix(text or "") and score > best_score:
                            best_score = score
                            best_text  = text
                            best_det   = det

                        # early accept if very sharp
                        if plate_suffix(text or "") and sharp >= MIN_SHARPNESS_LOCK:
                            capture_open = 1  # close next lines immediately

                        capture_open -= 1

                        # window closing -> log IN
                        if capture_open == 0:
                            chosen = best_text or text or ""
                            suf    = plate_suffix(chosen)
                            city, kana = parse_city_kana(chosen)
                            object_id = f"01({suf})" if suf else "01"
                            write_row_flush(csv_writer, csv_file,
                                [ts, object_id, "car", "in", city or "", kana or ""])
                            captured_in = True
                            last_event_f = frame_idx
                            # lock (avoid re-ocr) until exit is detected by motion calm
                            # (we approximate exit by lack of motion for some frames)
                            if DRAW_BBOXES and best_det is not None:
                                bx1,by1,bx2,by2 = map(int, xyxy_from_det(best_det))
                                cv2.putText(frame, f"IN {object_id}",
                                            (bx1, max(20, by1-6)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv2.LINE_AA)

                # crude "exit": if captured and no motion for a while, mark OUT
                # (you can swap to a tracker if you prefer)
                if captured_in and motion_boost == 0 and do_heavy is False:
                    # nothing to do on this frame; wait for the calm streak below
                    pass

            # If we logged IN earlier and the ROI has been calm long enough,
            # consider the car gone -> write OUT once.
            if captured_in:
                # calm if no boost and last heavy a while ago
                calm = (motion_boost == 0)
                if calm and (frame_idx - last_event_f) >= EXIT_STABLE_FRAMES:
                    write_row_flush(csv_writer, csv_file,
                        [ts, "01", "car", "out", "", ""])
                    captured_in = False
                    best_score = 0.0; best_text=""; best_det=None
                    capture_open = 0
                    last_event_f = frame_idx

            # FPS overlay
            if SHOW_FPS:
                now = time.time()
                if now - t0 >= 0.5:
                    fps_est = fcount / (now - t0); t0 = now; fcount = 0
                tag = "" if do_heavy else "  (idle)"
                cv2.putText(frame, f"FPS ~ {fps_est:.1f}{tag}", (10, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

            # Display / write
            cv2.imshow(win, frame)
            if out is not None and WRITE_VIDEO: out.write(frame)

            k = cv2.waitKey(1) & 0xFF
            if k == 27: break   # ESC
            # (No keyboard ROI controls; mouse handles it when LOCK_ROI=False)

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
