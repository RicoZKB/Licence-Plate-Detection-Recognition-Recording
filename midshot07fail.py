# -*- coding: utf-8 -*-
# region08_one_shot.py — ROI-only one-shot LP capture (ENG)
# - Outside ROI: do nothing heavy (pure preview)
# - Inside ROI: cheap motion check; when motion indicates entry, run LP+OCR ONCE
# - Choose the best single frame by sharpness & suffix; then lock until exit
# - ROI is resizable from keyboard; thin overlays for speed
#
# Keys:  ESC quit | WASD move ROI | [ ] resize ROI | r reset ROI

import os, cv2, csv, re, time
from datetime import datetime
from detections import LicencePlateDetection

# ===================== User settings =====================
USE_CAMERA               = False
INPUT_VIDEO_PATH         = "input_videos/Trim03.mp4"

# Region as ratios of frame (x, y, w, h)
REGION_XYWH_RATIO        = (0.32, 0.60, 0.36, 0.20)
LOCK_ROI                 = False     # <— ROI is sizable

# Logging
WRITE_CSV                = True

# One-shot capture rules
PLATE_SUFFIX_RE          = re.compile(r"([0-9]{2,3}-[0-9]{2})")
REQUIRE_CENTERED         = True      # demand roughly centered LP before accept
CENTER_BAND              = 0.35      # fraction of ROI width that counts as "middle"
MIN_SHARPNESS_LOCK       = 60.0      # Laplacian variance threshold for instant accept
CAPTURE_WINDOW_FRAMES    = 8         # fallback window to pick best (area*sharpness)

# Motion / exit logic (cheap)
MOTION_BLUR_SIZE         = 5
MOTION_ABS_THRESH        = 14        # per-pixel threshold for absdiff
MOTION_RATIO_ENTER       = 0.015     # % changed pixels to consider "entered"
MOTION_RATIO_KEEP        = 0.006     # % to keep "still inside"
NO_MOTION_EXIT_FRAMES    = 10        # consecutive low-motion frames => exit

# Performance toggles
DRAW_BBOXES              = True
SHOW_FPS                 = True
TARGET_WIDTH             = 640       # small input for speed
BOX_THICK                = 1         # thin overlays
# =========================================================

# --- tiny OpenCV speed ups ---
cv2.setUseOptimized(True)
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")
os.environ.setdefault("FLAGS_minloglevel", "2")

# ---------- helpers ----------
def open_daily_csv():
    date_str = datetime.now().strftime("%Y%m%d")
    os.makedirs("logs", exist_ok=True)
    path = os.path.join("logs", f"parking_log_{date_str}.csv")
    new = not os.path.exists(path)
    f = open(path, "a", newline="", encoding="utf-8", buffering=1)  # line-buffered
    w = csv.writer(f)
    if new:
        w.writerow(["timestamp","text","city","kana","suffix","sharpness"])
        f.flush()
    return f, w, path

CITY_KANA_RE = re.compile(u"^\\s*([^\\s\\d]+)\\s+[0-9]{3,4}\\s*([ぁ-ゖァ-ヿ一-龯A-Za-z])", re.UNICODE)

def parse_city_kana(text):
    if not text: return None, None
    m = CITY_KANA_RE.search(text)
    return (m.group(1), m.group(2)) if m else (None, None)

def plate_suffix(text):
    if not text: return None
    m = PLATE_SUFFIX_RE.search(text)
    return m.group(1) if m else None

def variance_of_laplacian(img):
    if img is None or img.size == 0: return 0.0
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape)==3 else img
    return float(cv2.Laplacian(g, cv2.CV_64F).var())

def clamp_region(rx, ry, rw, rh, W, H):
    rx = max(0, min(rx, W-10)); ry = max(0, min(ry, H-10))
    rw = max(20, min(rw, W-rx)); rh = max(20, min(rh, H-ry))
    return rx, ry, rw, rh

def draw_region(frame, rx, ry, rw, rh):
    cv2.rectangle(frame, (rx, ry), (rx+rw, ry+rh), (0,255,255), BOX_THICK)
    if not LOCK_ROI:
        cv2.putText(frame, "ROI: WASD move, [ ] resize, r reset",
                    (rx+6, max(18, ry-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1, cv2.LINE_AA)

def draw_lp_boxes(frame, dets, color=(0,255,0)):
    for d in dets:
        if hasattr(d, "bbox"): box = d.bbox
        elif isinstance(d, (list, tuple)) and len(d)>=4: box = d[:4]
        elif isinstance(d, dict) and ("bbox" in d or "xyxy" in d):
            box = (d.get("bbox") or d.get("xyxy"))[:4]
        else:
            box = None
        if not box: continue
        x1,y1,x2,y2 = map(int, box)
        cv2.rectangle(frame, (x1,y1), (x2,y2), color, BOX_THICK)

def roi_crop(frame, rx, ry, rw, rh):
    return frame[ry:ry+rh, rx:rx+rw]

def is_centered(det, rx, ry, rw, rh):
    # det can be [x1,y1,x2,y2] in full-frame coords
    if det is None: return False
    if isinstance(det, dict):
        box = det.get("bbox") or det.get("xyxy")
        if not box: return False
        x1,y1,x2,y2 = map(float, box[:4])
    else:
        x1,y1,x2,y2 = map(float, det[:4])
    cx = (x1 + x2) * 0.5
    # compute center band in full-frame coords
    band_w = rw * CENTER_BAND
    band_x1 = rx + (rw - band_w) * 0.5
    band_x2 = band_x1 + band_w
    return (band_x1 <= cx <= band_x2)

def offset_det(det, ox, oy):
    if isinstance(det, (list, tuple)) and len(det) >= 4:
        x1,y1,x2,y2 = det[:4]
        return [x1+ox, y1+oy, x2+ox, y2+oy]
    if isinstance(det, dict):
        box = det.get("bbox") or det.get("xyxy")
        if box and len(box) >= 4:
            nb = [box[0]+ox, box[1]+oy, box[2]+ox, box[3]+oy]
            if "bbox" in det: det["bbox"] = nb
            else: det["xyxy"] = nb
            return det
    return det

# -------------- main --------------
def main():
    src = 0 if USE_CAMERA else INPUT_VIDEO_PATH
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {src}")

    try: cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)
    except: pass

    lp_det = LicencePlateDetection(model_path="models/best.pt")

    csv_file = None; csv_writer = None; csv_path = None
    if WRITE_CSV:
        csv_file, csv_writer, csv_path = open_daily_csv()
        print("[INFO] Logging to:", csv_path)

    fps_src = cap.get(cv2.CAP_PROP_FPS) or 20.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or TARGET_WIDTH)
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or int(TARGET_WIDTH*3/5))

    # ROI init from ratios
    rx = int(REGION_XYWH_RATIO[0]*W); ry = int(REGION_XYWH_RATIO[1]*H)
    rw = int(REGION_XYWH_RATIO[2]*W); rh = int(REGION_XYWH_RATIO[3]*H)
    rx, ry, rw, rh = clamp_region(rx, ry, rw, rh, W, H)

    # Motion state
    prev_gray = None
    inside = False               # motion says "object present" inside ROI
    still_counter = 0            # low-motion streak = candidate exit
    captured = False             # we already grabbed a plate for this visit
    capture_window = 0
    best_score = 0.0
    best_text  = ""
    best_det   = None
    best_sharp = 0.0

    # FPS overlay
    t0 = time.time(); fcount = 0; fps_est = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            fcount += 1

            rx, ry, rw, rh = clamp_region(rx, ry, rw, rh, W, H)
            roi = roi_crop(frame, rx, ry, rw, rh)

            # --- cheap motion detector INSIDE ROI (no model calls) ---
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (MOTION_BLUR_SIZE, MOTION_BLUR_SIZE), 0)

            motion_ratio = 0.0
            if prev_gray is not None and prev_gray.shape == gray.shape:
                diff = cv2.absdiff(prev_gray, gray)
                _, th = cv2.threshold(diff, MOTION_ABS_THRESH, 255, cv2.THRESH_BINARY)
                changed = int(th.sum() // 255)
                motion_ratio = changed / float(th.size)

            prev_gray = gray

            # enter / keep / exit logic (purely motion-based)
            if not inside:
                if motion_ratio >= MOTION_RATIO_ENTER:
                    inside = True
                    captured = False
                    capture_window = CAPTURE_WINDOW_FRAMES
                    best_score = 0.0; best_text = ""; best_det = None; best_sharp = 0.0
            else:
                # already inside
                if motion_ratio < MOTION_RATIO_KEEP:
                    still_counter += 1
                    if still_counter >= NO_MOTION_EXIT_FRAMES:
                        inside = False
                        still_counter = 0
                        captured = False       # ready for next car
                        capture_window = 0
                        best_score = 0.0; best_text = ""; best_det = None; best_sharp = 0.0
                else:
                    still_counter = 0

            # --- One-shot LP capture ONLY when inside & not yet captured ---
            did_heavy = False
            if inside and not captured:
                # Run detector/OCR just while our small window is open,
                # or immediately accept a sharp + valid frame.
                did_heavy = True
                all_lp_dets, all_lp_texts = lp_det.detect_frames([roi])
                roi_dets = all_lp_dets[0] if all_lp_dets else []
                roi_txts = all_lp_texts[0] if all_lp_texts else []

                # offset boxes back to full-frame
                lp_dets_full = [offset_det(d, rx, ry) for d in roi_dets]

                # Evaluate candidates
                for det, text in zip(lp_dets_full, roi_txts):
                    # crop plate patch from full frame
                    if isinstance(det, dict):
                        box = det.get("bbox") or det.get("xyxy")
                    else:
                        box = det if (isinstance(det, (list, tuple)) and len(det)>=4) else None
                    if not box: continue
                    x1,y1,x2,y2 = map(int, box[:4])
                    x1 = max(0, min(x1, W-1)); x2 = max(0, min(x2, W-1))
                    y1 = max(0, min(y1, H-1)); y2 = max(0, min(y2, H-1))
                    if x2 <= x1 or y2 <= y1: continue

                    patch = frame[y1:y2, x1:x2]
                    sharp = variance_of_laplacian(patch)
                    suf = plate_suffix(text or "")

                    # keep best by (area * sharpness) but only if suffix looks valid
                    area = (x2 - x1) * (y2 - y1)
                    score = area * max(1.0, sharp)

                    # must be within center band (if enabled)
                    centered_ok = (not REQUIRE_CENTERED) or is_centered([x1,y1,x2,y2], rx, ry, rw, rh)

                    if suf and centered_ok and score > best_score:
                        best_score = score
                        best_text  = text
                        best_det   = [x1,y1,x2,y2]
                        best_sharp = sharp

                    # instant lock if very sharp & valid & centered
                    if suf and centered_ok and sharp >= MIN_SHARPNESS_LOCK:
                        capture_window = 1  # force close next block

                # close window and commit once
                capture_window -= 1
                if capture_window <= 0:
                    chosen = best_text
                    if chosen:
                        city, kana = parse_city_kana(chosen)
                        suf = plate_suffix(chosen)
                        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        if WRITE_CSV:
                            csv_writer.writerow([ts, chosen, city or "", kana or "", suf or "", f"{best_sharp:.1f}"])
                            csv_file.flush()
                            try: os.fsync(csv_file.fileno())
                            except Exception: pass

                        # draw a one-time label
                        if DRAW_BBOXES and best_det is not None:
                            x1,y1,x2,y2 = map(int, best_det)
                            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), BOX_THICK)
                            cv2.putText(frame, f"{chosen}", (x1, max(20, y1-6)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)

                    # lock until exit (no more heavy work)
                    captured = True

            # --- overlays (very cheap) ---
            if DRAW_BBOXES:
                draw_region(frame, rx, ry, rw, rh)

            # FPS overlay
            if SHOW_FPS:
                now = time.time()
                if now - t0 >= 0.5:
                    fps_est = fcount / (now - t0)
                    t0 = now; fcount = 0
                tag = "" if not (inside and not captured and did_heavy) else "  (capture)"
                cv2.putText(frame, f"FPS ~ {fps_est:.1f}{tag}", (10, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

            cv2.imshow("Parking (ROI one-shot, LP only)", frame)

            # ROI keyboard control
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
        cv2.destroyAllWindows()
        if WRITE_CSV and csv_file:
            csv_file.close()
            print("[INFO] CSV saved to", csv_path)

if __name__ == "__main__":
    main()
