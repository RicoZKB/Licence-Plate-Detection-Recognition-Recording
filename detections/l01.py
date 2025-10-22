import cv2
from ultralytics import YOLO
from paddleocr import PaddleOCR

class LicencePlateDetection:
    def __init__(self, model_path):
        # load your YOLO model
        self.model = YOLO(model_path)
        # initialize PaddleOCR
        self.ocr = PaddleOCR(use_angle_cls=True, lang='japan')

    def detect_frames(self, frames):
        all_bboxes = []
        all_texts = []
        for frame in frames:
            bboxes, texts = self.detect_frame(frame)
            all_bboxes.append(bboxes)
            all_texts.append(texts)
        return all_bboxes, all_texts

    def detect_frame(self, frame):
        results = self.model.predict(frame)[0]
        detections = []
        texts = []

        for box in results.boxes:
            cls_id = int(box.cls.tolist()[0])
            if results.names[cls_id] != "License_Plate":
                continue

            # get bounding box coords
            x1, y1, x2, y2 = map(int, box.xyxy.tolist()[0])
            crop = frame[y1:y2, x1:x2]

            # preprocess for OCR
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            up = cv2.resize(gray, None, fx=2, fy=2)
            prep = cv2.cvtColor(up, cv2.COLOR_GRAY2BGR)

            # RUN OCR — no cls=True
            ocr_res = self.ocr.ocr(prep)
            print("DEBUG OCR_RES:", ocr_res)

            # normalize each line into (y_top, text)
            lines = []
            for entry in ocr_res:
                # Case A: [pts, txt, conf]
                if isinstance(entry, (list, tuple)) and len(entry) == 3 and isinstance(entry[1], str):
                    pts, txt = entry[0], entry[1]

                # Case B: [ [pts], (txt, conf) ]
                elif isinstance(entry, (list, tuple)) and len(entry) >= 2 and isinstance(entry[1], (list, tuple)):
                    pts = entry[0][0] if isinstance(entry[0][0][0], (list, tuple)) else entry[0]
                    txt = entry[1][0]

                # Case C: dict-based (PaddleOCR v3+)
                elif isinstance(entry, dict):
                    pts = entry.get("box") or entry.get("bbox") or entry.get("boxes") or []
                    recs = entry.get("rec_texts") or entry.get("rec_text") or entry.get("text")
                    if isinstance(recs, list):
                        txt = recs[0] if recs else ""
                    else:
                        txt = recs or ""
                else:
                    continue

                # skip if no text
                if not txt or not isinstance(pts, (list, tuple)):
                    continue

                # get the top-most y coordinate
                try:
                    y_top = min(p[1] for p in pts)
                except Exception:
                    continue
                lines.append((y_top, txt))

            # sort by vertical position and join
            lines.sort(key=lambda x: x[0])
            plate_str = " ".join(txt for _, txt in lines)

            detections.append((x1, y1, x2, y2))
            texts.append(plate_str)

        return detections, texts

    def draw_bboxes(self, video_frames, licence_plate_detections, licence_plate_texts):
        output_frames = []
        for frame, bbox_list, text_list in zip(video_frames, licence_plate_detections, licence_plate_texts):
            for (x1, y1, x2, y2), text in zip(bbox_list, text_list):
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,0), 2)
            output_frames.append(frame)
        return output_frames

