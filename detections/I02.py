import cv2
from ultralytics import YOLO
from paddleocr import PaddleOCR

class LicencePlateDetection:
    def __init__(self, model_path, verbose=False, debug_ocr=False):
        self.model = YOLO(model_path)
        self.ocr = PaddleOCR(use_angle_cls=True, lang='japan')
        self.verbose = bool(verbose)
        self.debug_ocr = bool(debug_ocr)

    def detect_frames(self, frames):
        all_bboxes, all_texts = [], []
        for frame in frames:
            bboxes, texts = self.detect_frame(frame)
            all_bboxes.append(bboxes)
            all_texts.append(texts)
        return all_bboxes, all_texts

    def _parse_ocr_lines(self, ocr_res):
        """Return list[(y_top, text)], robust to PaddleOCR output variants."""
        lines = []

        # Newer dict style
        if (
            isinstance(ocr_res, list) and len(ocr_res) >= 1 and
            isinstance(ocr_res[0], dict) and ("rec_texts" in ocr_res[0] or "text" in ocr_res[0])
        ):
            entry = ocr_res[0]
            rec_texts = entry.get("rec_texts")
            rec_boxes = entry.get("rec_boxes")
            rec_polys = entry.get("rec_polys")

            if isinstance(rec_texts, list) and rec_texts:
                for i, txt in enumerate(rec_texts):
                    y_top = 0
                    if rec_boxes is not None and len(rec_boxes) > i:
                        try:
                            box = rec_boxes[i]  # [x1,y1,x2,y2]
                            y1b, y2b = int(box[1]), int(box[3])
                            y_top = min(y1b, y2b)
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
                single = entry.get("rec_text") or entry.get("text") or ""
                if single:
                    lines.append((0, str(single)))
            return lines

        # Older list styles
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

        return lines

    def _ocr_once(self, img_bgr):
        try:
            return self.ocr.ocr(img_bgr)
        except Exception:
            return None

    def detect_frame(self, frame):
        results = self.model.predict(frame)[0]
        detections, texts = [], []

        names = getattr(results, "names", {})
        try:
            num_classes = len(names)
        except Exception:
            num_classes = None

        for box in results.boxes:
            # class filtering
            cls_id = None
            try:
                cls_id = int(box.cls[0].item())
            except Exception:
                try:
                    cls_id = int(box.cls.tolist()[0])
                except Exception:
                    cls_id = None

            cls_name = ""
            if isinstance(names, dict) and cls_id is not None:
                cls_name = str(names.get(cls_id, ""))
            elif isinstance(names, list) and cls_id is not None and 0 <= cls_id < len(names):
                cls_name = str(names[cls_id])

            keep = False
            if num_classes == 1:
                keep = True
            else:
                nm = cls_name.lower()
                if ("plate" in nm) or ("licence" in nm) or ("license" in nm) or (nm == "lp") or ("number" in nm):
                    keep = True
            if not keep:
                continue

            # bbox + crop
            x1, y1, x2, y2 = map(int, box.xyxy.tolist()[0])
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # ---------- preprocess (adaptive upscaling + contrast + mild sharpen) ----------
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

            # CLAHE
            try:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                gray = clahe.apply(gray)
            except Exception:
                pass

            # mild unsharp
            try:
                blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
                gray = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)
            except Exception:
                pass

            # Adaptive scale: boost tiny crops more, keep big ones modest
            h, w = gray.shape[:2]
            target_h = 96 if (w*h) < (120*120) else 128
            scale = max(1.5, min(3.0, float(target_h) / max(1.0, float(h))))
            up = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

            # Variant A (default)
            prepA = cv2.cvtColor(up, cv2.COLOR_GRAY2BGR)

            # Variant B (cheap binarization for low contrast)
            try:
                th = cv2.adaptiveThreshold(up, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, 31, 5)
                prepB = cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)
            except Exception:
                prepB = None

            # ---------- OCR (two variants, pick best) ----------
            resA = self._ocr_once(prepA)
            resB = self._ocr_once(prepB) if prepB is not None else None

            # choose by (#lines, total text length)
            candidates = []
            for rr in (resA, resB):
                if not rr:
                    continue
                # count lines for newer dict style
                try:
                    n = len(rr[0].get("rec_texts") or [])
                    s = " ".join((rr[0].get("rec_texts") or [])) if n > 0 else (rr[0].get("rec_text") or rr[0].get("text") or "")
                    candidates.append((n, len(s), rr))
                except Exception:
                    # older list style
                    flat = []
                    for e in rr:
                        if isinstance(e, (list, tuple)) and len(e) >= 2:
                            if isinstance(e[1], str):
                                flat.append(e[1])
                            elif isinstance(e[1], (list, tuple)) and len(e[1]) > 0:
                                flat.append(str(e[1][0]))
                    s = " ".join(flat)
                    candidates.append((len(flat), len(s), rr))

            chosen = None
            if candidates:
                candidates.sort(key=lambda z: (z[0], z[1]))  # prefer more lines, then longer text
                chosen = candidates[-1][2]

            ocr_res = chosen if chosen is not None else resA
            if self.debug_ocr:
                try:
                    print("DEBUG OCR_RES:", ocr_res)
                except Exception:
                    pass

            # ---------- parse -> final joined string ----------
            lines = self._parse_ocr_lines(ocr_res or [])
            lines.sort(key=lambda x: x[0])
            plate_str = " ".join(txt for _, txt in lines).strip()

            if self.verbose:
                try:
                    print(f"LP DET: bbox=({x1},{y1},{x2},{y2}) text='{plate_str}'")
                except Exception:
                    pass

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
