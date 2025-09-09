import cv2
import os
import csv
import re
from datetime import datetime
from detections import CarDetection, LicencePlateDetection

# ---------- ID manager keyed by OCR plate text ----------
class IDBank:
    def __init__(self):
        self._map = {}
        self._next = 100
    def get_id(self, plate_text: str) -> int:
        if plate_text not in self._map:
            self._map[plate_text] = self._next
            self._next += 1
        return self._map[plate_text]

# ---------- Very light JP plate parser ----------
PLATE_RE = re.compile(r"^\s*([^\s\d]+)\s+[0-9]{3,4}\s*([ぁ-ゖァ-ヿ一-龯A-Za-z])", re.UNICODE)
def parse_city_kana(plate_text: str):
    m = PLATE_RE.search(plate_text or "")
    if not m:
        return None, None
    return m.group(1), m.group(2)

# ---------- CSV helpers ----------
def open_daily_csv():
    date_str = datetime.now().strftime("%Y%m%d")
    os.makedirs("logs", exist_ok=True)
    path = os.path.join("logs", f"parking_log_{date_str}.csv")
    new_file = not os.path.exists(path)
    f = open(path, "a", newline="", encoding="utf-8")
    w = csv.writer(f)
    if new_file:
        w.writerow(["timestamp","object_id","vehicle_type","direction","city","kana"])
    return f, w, path

# ---------- Virtual gate settings ----------
GATE_Y_RATIO = 0.55   # 0(top) .. 1(bottom) — tweak this
ENTRY_IS_UP   = True  # True: up-cross = 'in', down-cross = 'out'

# Track last Y center per object
last_y_by_id = {}

def det_center_y(det):
    """
    Try to extract bbox and return center Y.
    Supported shapes:
      [x1,y1,x2,y2,*rest] OR dict with 'bbox' or 'xyxy'
    """
    if isinstance(det, (list, tuple)) and len(det) >= 4:
        x1, y1, x2, y2 = det[:4]
    elif isinstance(det, dict):
        if "bbox" in det and len(det["bbox"]) >= 4:
            x1, y1, x2, y2 = det["bbox"][:4]
        elif "xyxy" in det and len(det["xyxy"]) >= 4:
            x1, y1, x2, y2 = det["xyxy"][:4]
        else:
            return None
    else:
        return None
    return (y1 + y2) / 2.0

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open USB camera (index 0)")

    car_detector = CarDetection(model_path="yolo11n.pt")
    lp_detector  = LicencePlateDetection(model_path="models/best.pt")

    os.makedirs("output_videos", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    fps    = cap.get(cv2.CAP_PROP_FPS) or 20.0
    w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out    = cv2.VideoWriter("output_videos/output_video.avi", fourcc, fps, (w, h))

    csv_file, csv_writer, csv_path = open_daily_csv()
    id_bank = IDBank()
    print(f"[INFO] Logging to: {csv_path}")

    gate_y = int(h * GATE_Y_RATIO)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Detections
            car_dets = car_detector.detect_frames([frame], read_from_stub=False)[0]
            all_lp_dets, all_lp_texts = lp_detector.detect_frames([frame])
            lp_dets, lp_texts = all_lp_dets[0], all_lp_texts[0]

            # Drawing
            frame = car_detector.draw_bboxes([frame], [car_dets])[0]
            frame = lp_detector.draw_bboxes([frame], [lp_dets], [lp_texts])[0]

            # Draw virtual gate
            cv2.line(frame, (0, gate_y), (w, gate_y), (0, 255, 255), 2)
            label = f"GATE y={gate_y} ({'up=in' if ENTRY_IS_UP else 'down=in'})"
            cv2.putText(frame, label, (10, max(25, gate_y-10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2, cv2.LINE_AA)

            # Direction events (log only on crossing)
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            for det, text in zip(lp_dets, lp_texts):
                if not text:
                    continue
                oid = id_bank.get_id(text)
                cy  = det_center_y(det)
                if cy is None:
                    continue

                # previous Y?
                prev = last_y_by_id.get(oid)
                last_y_by_id[oid] = cy

                if prev is None:
                    continue  # need a previous position to determine direction

                crossed = (prev - gate_y) * (cy - gate_y) < 0  # sign changed → crossed
                if not crossed:
                    continue

                moving_up = cy < prev  # y smaller means moving up
                if ENTRY_IS_UP:
                    direction = "in" if moving_up else "out"
                else:
                    direction = "in" if not moving_up else "out"

                city, kana = parse_city_kana(text)
                csv_writer.writerow([ts, oid, "car", direction, city or "", kana or ""])
                print(f"[EVENT] {ts} id={oid} dir={direction} text='{text}'")

            # Show & save
            cv2.imshow("Live Detection", frame)
            out.write(frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")

    finally:
        cap.release()
        out.release()
        csv_file.close()
        cv2.destroyAllWindows()
        print("[INFO] Video saved to output_videos/output_video.avi")
        print(f"[INFO] CSV saved to {csv_path}")

if __name__ == "__main__":
    main()
