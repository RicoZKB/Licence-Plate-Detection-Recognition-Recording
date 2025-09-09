import cv2
import os
import csv
from datetime import datetime
import re
from detections import CarDetection, LicencePlateDetection

# --- simple object-id manager (keyed by plate text) ---
class IDBank:
    def __init__(self):
        self._map = {}
        self._next = 100  # start at 100 like your example
    def get_id(self, plate_text: str) -> int:
        if plate_text not in self._map:
            self._map[plate_text] = self._next
            self._next += 1
        return self._map[plate_text]

# --- very light JP plate parser: returns (city, kana) when possible ---
PLATE_RE = re.compile(r"^\s*([^\s\d]+)\s+[0-9]{3,4}\s*([ぁ-ゖァ-ヿ一-龯A-Za-z])", re.UNICODE)
def parse_city_kana(plate_text: str):
    """
    Expected styles like: '神戸 500 わ 12-34', '大阪 330 ふ 12-34'
    Returns ('神戸','わ') or (None,None) if not found.
    """
    m = PLATE_RE.search(plate_text or "")
    if not m:
        return None, None
    city, kana = m.group(1), m.group(2)
    # Normalize katakana to hiragana if needed (optional simple map)
    return city, kana

def open_daily_csv():
    date_str = datetime.now().strftime("%Y%m%d")
    os.makedirs("logs", exist_ok=True)
    path = os.path.join("logs", f"parking_log_{date_str}.csv")
    new_file = not os.path.exists(path)
    f = open(path, "a", newline="", encoding="utf-8")
    writer = csv.writer(f)
    if new_file:
        writer.writerow(["timestamp","object_id","vehicle_type","direction","city","kana"])
    return f, writer, path

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open USB camera (index 0)")

    car_detector = CarDetection(model_path="yolo11n.pt")
    lp_detector  = LicencePlateDetection(model_path="models/best.pt")

    os.makedirs("output_videos", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")  # AVI, QuickTime-friendly
    fps    = cap.get(cv2.CAP_PROP_FPS) or 20.0
    w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out    = cv2.VideoWriter("output_videos/output_video.avi", fourcc, fps, (w, h))

    # CSV setup
    csv_file, csv_writer, csv_path = open_daily_csv()
    id_bank = IDBank()

    print(f"[INFO] Logging to: {csv_path}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Car detections (kept for drawing)
            car_dets = car_detector.detect_frames([frame], read_from_stub=False)[0]

            # Plate detections + OCR
            all_lp_dets, all_lp_texts = lp_detector.detect_frames([frame])
            lp_dets, lp_texts = all_lp_dets[0], all_lp_texts[0]

            # Draw
            frame = car_detector.draw_bboxes([frame], [car_dets])[0]
            frame = lp_detector.draw_bboxes([frame], [lp_dets], [lp_texts])[0]

            # --- CSV logging: one row per recognized plate text (once per frame) ---
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            for text in lp_texts:
                if not text:  # skip empty OCR results
                    continue
                obj_id = id_bank.get_id(text)
                city, kana = parse_city_kana(text)
                # direction: unknown without entry/exit logic → put 'in' as placeholder or ''
                direction = ""  # TODO: set 'in'/'out' if you add zone/track logic
                csv_writer.writerow([ts, obj_id, "car", direction, city or "", kana or ""])

            # Show & save
            cv2.imshow("Live Detection", frame)
            out.write(frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user. Saving and closing...")

    finally:
        cap.release()
        out.release()
        csv_file.close()
        cv2.destroyAllWindows()
        print("[INFO] Video saved to output_videos/output_video.avi")
        print(f"[INFO] CSV saved to {csv_path}")

if __name__ == "__main__":
    main()
