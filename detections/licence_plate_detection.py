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
            # print("DEBUG OCR_RES:", ocr_res)

            # normalize each line into (y_top, text)
            lines = []

            # Newer PaddleOCR (dict style with batched lines)
            if (
                isinstance(ocr_res, list)
                and len(ocr_res) >= 1
                and isinstance(ocr_res[0], dict)
                and ("rec_texts" in ocr_res[0] or "text" in ocr_res[0])
            ):
                entry = ocr_res[0]
                rec_texts = entry.get("rec_texts")
                rec_boxes = entry.get("rec_boxes")  # Nx4 [x1,y1,x2,y2]
                rec_polys = entry.get("rec_polys")  # list of 4-point polys

                if isinstance(rec_texts, list) and rec_texts:
                    for i, txt in enumerate(rec_texts):
                        y_top = 0
                        if rec_boxes is not None and len(rec_boxes) > i:
                            try:
                                # rec_boxes[i] => [x1,y1,x2,y2]
                                box = rec_boxes[i]
                                y1, y2 = int(box[1]), int(box[3])
                                y_top = min(y1, y2)
                            except Exception:
                                y_top = 0
                        elif rec_polys is not None and len(rec_polys) > i:
                            try:
                                poly = rec_polys[i]
                                y_top = min(int(p[1]) for p in poly)
                            except Exception:
                                y_top = 0
                        lines.append((y_top, str(txt)))
                else:
                    # Fallback single text
                    single = entry.get("rec_text") or entry.get("text") or ""
                    if single:
                        lines.append((0, str(single)))

            else:
                # Older styles: list entries
                for entry in ocr_res:
                    # Case A: [pts, txt, conf]
                    if isinstance(entry, (list, tuple)) and len(entry) == 3 and isinstance(entry[1], str):
                        pts, txt = entry[0], entry[1]

                    # Case B: [ [pts], (txt, conf) ]
                    elif isinstance(entry, (list, tuple)) and len(entry) >= 2 and isinstance(entry[1], (list, tuple)):
                        pts = entry[0][0] if isinstance(entry[0][0][0], (list, tuple)) else entry[0]
                        txt = entry[1][0]
                    else:
                        continue

                    if not txt:
                        continue
                    try:
                        y_top = min(p[1] for p in pts)
                    except Exception:
                        y_top = 0
                    lines.append((y_top, str(txt)))

            # sort by vertical position and join
            lines.sort(key=lambda x: x[0])
            plate_str = " ".join(txt for _, txt in lines).strip()

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
