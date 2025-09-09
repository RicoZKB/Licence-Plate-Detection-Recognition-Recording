# -*- coding: utf-8 -*-
"""
region04.py — Single-thread, no "latest-frame grabber", fast gate logger

- Plain cv2.VideoCapture loop (no threads)
- Drain camera buffer each loop to avoid lag
- Optional frame skipping for file playback
- Slot leasing (01..10), OCR-once-per-entry lock
- Immediate CSV flush+fsync on every write
"""

import os, csv, re, cv2, time
from datetime import datetime
from detections import CarDetection, LicencePlateDetection  # your classes

# ===================== USER SETTINGS =====================
USE_CAMERA           = False
INPUT_VIDEO_PATH     = "input_videos/Trim03.mp4"

SLOT_COUNT           = 10
START_ID_AT          = 100

# Viewer & perf
SHOW_BOXES           = True
WRITE_VIDEO          = False
TARGET_WIDTH         = 640           # resize width (keep aspect)
OCR_MIN_SCORE        = 0.85
OCR_EVERY_N_FRAMES   = 4             # used by our fallback scoring
FILE_SKIP_EVERY_N    = 2             # for file input: process 1, skip (N-1); set 1 to process all
CAM_DRAIN_GRABS      = 3             # for camera: extra cap.grab() calls to drop stale frames

# Gate (horizontal line at a y-ratio of the frame)
GATE_Y_RATIO         = 0.93          # 0.0 top .. 1.0 bottom
ENTRY_IS_UP          = False         # False => downward crossing = "in"
MISS_GRACE_FRAMES    = 12            # prune unseen tracks
# ========================================================

# ---------- CSV ----------
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
        try: os.fsync(f.fileno())
        except Exception: pass
    return f, w, path

def write_csv_row(w, f, row):
    w.writerow(row)
    f.flush()
    try: os.fsync(f.fileno())
    except Exception: pass

# ---------- Simple helpers ----------
CITY_KANA_RE = re.compile(r"^\s*([^\s\d]+)\s+[0-9]{3,4}\s*([ぁ-ゖァ-ヿ一-龯A-Za-z])")
NUM_SUFFIX_RE= re.compile(r"([0-9]{2,3}-[0-9]{2})")
def parse_city_kana(text):
    if not text: return None, None
    m = CITY_KANA_RE.search(text);  return (m.group(1), m.group(2)) if m else (None, None)
def parse_number_suffix(text):
    if not text: return None
    m = NUM_SUFFIX_RE.search(text); return m.group(1) if m else None

def xyxy_from_det(det):
    if isinstance(det, (list, tuple)) and len(det) >= 4:
        return float(det[0]), float(det[1]), float(det[2]), float(det[3])
    if isinstance(det, dict):
        box = det.get("bbox") or det.get("xyxy")
        if box and len(box) >= 4:
            return float(box[0]), float(box[1]), float(box[2]), float(box[3])
    return None

def center_y(det):
    xy = xyxy_from_det(det)
    return None if not xy else (xy[1]+xy[3])/2.0

def key_bbox(det):
    xy = xyxy_from_det(det)
    if not xy: return None
    x1,y1,x2,y2 = xy
    cx = (x1+x2)/2.0; cy = (y1+y2)/2.0
    return f"bb-{int(cx//20)}-{int(cy//20)}"

# ---------- ID/Slot ----------
class IDBank:
    def __init__(self, start_at=100):
        self._map = {}; self._next = int(start_at)
    def get(self, key):
        if key not in self._map:
            self._map[key] = self._next; self._next += 1
        return self._map[key]

class SlotPool:
    def __init__(self, n):
        self.free = list(range(1, n+1)); self.used = set()
    def acquire(self):
        if not self.free: return None
        s = self.free.pop(0); self.used.add(s); return s
    def release(self, s):
        if s in self.used:
            self.used.remove(s); self.free.append(s); self.free.sort()

# ---------- Track ----------
class Track:
    def __init__(self, oid):
        self.oid = oid
        self.slot = None
        self.best_text = ""
        self.text_locked = False
        self.last_seen_f = 0
        self.state = "new"
        self.prev_y = None
        self.num_suffix = ""

    def update_text_once(self, text, score):
        if self.text_locked: return
        if not text or score < OCR_MIN_SCORE: return
        cand_suffix = parse_number_suffix(text)
        if cand_suffix:
            self.best_text = text
            self.num_suffix = cand_suffix
            self.text_locked = True

# ---------- Fallback wrapper (scores or synthesized) ----------
def _detect_lp_with_scores(lp_det, frame, ocr_every, frame_idx):
    """Try detect_frames_with_scores; if absent, use detect_frames and synthesize scores."""
    try:
        return lp_det.detect_frames_with_scores([frame], ocr_every=ocr_every)
    except AttributeError:
        all_lp_dets, all_lp_texts = lp_det.detect_frames([frame])

        def conf_from_text(t):
            if not t: return 0.0
            return 0.95 if NUM_SUFFIX_RE.search(t) else 0.60

        if all_lp_texts and len(all_lp_texts) > 0:
            scores0 = [conf_from_text(t) for t in all_lp_texts[0]]
        else:
            scores0 = []
        return all_lp_dets, all_lp_texts, [scores0]

# ---------- main ----------
def main():
    # open capture
    source = 0 if USE_CAMERA else INPUT_VIDEO_PATH
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera/video source")

    # keep buffer tiny if backend supports it (helps with webcams)
    try: cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception: pass

    # detectors (plate only for speed)
    lp_det  = LicencePlateDetection(model_path="models/best.pt")
    car_det = None  # not used

    # sizes
    w0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    h0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    if w0 != TARGET_WIDTH:
        scale = TARGET_WIDTH / float(w0); out_w = TARGET_WIDTH; out_h = int(h0*scale)
    else:
        out_w, out_h = w0, h0

    # video writer (optional)
    out = None
    if WRITE_VIDEO:
        os.makedirs("output_videos", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
        out = cv2.VideoWriter("output_videos/output_video.avi", fourcc, fps, (out_w, out_h))

    # CSV
    csv_file, csv_writer, csv_path = open_daily_csv()
    print("[INFO] Logging to:", csv_path)

    # state
    idbank = IDBank(START_ID_AT)
    tracks = {}                # oid -> Track
    slot_pool = SlotPool(SLOT_COUNT)
    frame_idx = 0
    file_frame_counter = 0

    try:
        while True:
            # Drain camera buffer a bit (keeps UI realtime). For file input we skip via counter.
            if USE_CAMERA:
                for _ in range(max(0, CAM_DRAIN_GRABS - 1)):
                    cap.grab()  # drop frame quickly

            ok, frame = cap.read()
            if not ok:
                break

            # Resize
            if (frame.shape[1] != out_w) or (frame.shape[0] != out_h):
                frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

            frame_idx += 1

            # For file: skip frames to avoid slow-mo playback
            if not USE_CAMERA and FILE_SKIP_EVERY_N > 1:
                file_frame_counter = (file_frame_counter + 1) % FILE_SKIP_EVERY_N
                if file_frame_counter != 0:
                    # still show something responsive (optional)
                    cv2.imshow("FAST Parking (region04)", frame)
                    if cv2.waitKey(1) & 0xFF in (27, ord('q')): break
                    if WRITE_VIDEO and out is not None: out.write(frame)
                    continue  # skip processing this frame

            H, W = frame.shape[:2]
            gate_y = int(H * GATE_Y_RATIO)

            # plate detection (scores may be synthesized)
            all_lp_dets, all_lp_texts, all_lp_scores = _detect_lp_with_scores(
                lp_det, frame, OCR_EVERY_N_FRAMES, frame_idx
            )
            lp_dets  = all_lp_dets[0] if all_lp_dets else []
            lp_texts = all_lp_texts[0] if all_lp_texts else []
            lp_scores= all_lp_scores[0] if all_lp_scores else []

            # draw gate
            cv2.line(frame, (0, gate_y), (W, gate_y), (0, 255, 255), 2)

            # update tracks
            seen = set()
            for det, text, score in zip(lp_dets, lp_texts, lp_scores):
                key = key_bbox(det)
                if key is None: 
                    continue
                oid = idbank.get(key)
                tr = tracks.get(oid)
                if tr is None:
                    tr = tracks[oid] = Track(oid)
                cy = center_y(det)
                if cy is None: 
                    continue

                tr.last_seen_f = frame_idx
                seen.add(oid)

                # lock plate text once per entry if confident
                tr.update_text_once(text, score)

                # crossing check
                if tr.prev_y is not None:
                    crossed = (tr.prev_y - gate_y) * (cy - gate_y) < 0
                    if crossed:
                        moving_up = cy < tr.prev_y
                        direction = "in" if ((ENTRY_IS_UP and moving_up) or (not ENTRY_IS_UP and not moving_up)) else "out"

                        if direction == "in":
                            # acquire slot if needed
                            if tr.slot is None:
                                slot = slot_pool.acquire()
                                if slot is not None:
                                    tr.slot = slot
                                    tr.state = "in"
                            # log once on IN if we have a slot
                            if tr.slot is not None:
                                suffix = tr.num_suffix or (parse_number_suffix(tr.best_text) or "")
                                object_id = f"{tr.slot:02d}({suffix})" if suffix else f"{tr.slot:02d}"
                                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                city, kana = parse_city_kana(tr.best_text)
                                write_csv_row(csv_writer, csv_file, [ts, object_id, "car", "in", city or "", kana or ""])
                                cv2.putText(frame, f"IN  {object_id}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2, cv2.LINE_AA)
                                tr.text_locked = True

                        else:  # OUT
                            if tr.slot is not None:
                                suffix = tr.num_suffix or (parse_number_suffix(tr.best_text) or "")
                                object_id = f"{tr.slot:02d}({suffix})" if suffix else f"{tr.slot:02d}"
                                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                city, kana = parse_city_kana(tr.best_text)
                                write_csv_row(csv_writer, csv_file, [ts, object_id, "car", "out", city or "", kana or ""])
                                cv2.putText(frame, f"OUT {object_id}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2, cv2.LINE_AA)
                                # free & reset
                                slot_pool.release(tr.slot)
                                tr.slot = None
                                tr.state = "new"
                                tr.text_locked = False

                tr.prev_y = cy

            # prune long-missed tracks with no slot
            to_del = []
            for oid, tr in tracks.items():
                if tr.slot is None and (frame_idx - tr.last_seen_f) > MISS_GRACE_FRAMES*3:
                    to_del.append(oid)
            for oid in to_del:
                del tracks[oid]

            # draw boxes (optional)
            if SHOW_BOXES:
                for det in lp_dets:
                    xy = xyxy_from_det(det)
                    if not xy: continue
                    x1,y1,x2,y2 = map(int, xy)
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,200,255), 2)

            # show / write
            cv2.imshow("FAST Parking (region04)", frame)
            if WRITE_VIDEO and out is not None:
                out.write(frame)

            k = cv2.waitKey(1) & 0xFF
            if k in (27, ord('q')):   # ESC or q
                break

    finally:
        cap.release()
        if out is not None: out.release()
        csv_file.close()
        cv2.destroyAllWindows()
        print("[INFO] CSV saved to", csv_path)

if __name__ == "__main__":
    try:
        cv2.setNumThreads(1)  # macOS OpenCV stability
    except Exception:
        pass
    main()
