# -*- coding: utf-8 -*-
import re
import cv2
from ultralytics import YOLO
from paddleocr import PaddleOCR

# kana range for quick scoring
KANA_RE = re.compile(u"[ぁ-ゖァ-ヿ]")

class LicencePlateDetection:
    def __init__(self, model_path, verbose=False, debug_ocr=False):
        """
        model_path: path to a YOLO model trained to detect license plates
        verbose: print basic detections
        debug_ocr: print raw OCR results
        """
        # load YOLO plate model
        self.model = YOLO(model_path)
        # PaddleOCR for Japanese
        self.ocr = PaddleOCR(use_angle_cls=True, lang='japan')
        self.verbose = bool(verbose)
        self.debug_ocr = bool(debug_ocr)

    # --------- public API ---------
    def detect_frames(self, frames):
        all_bboxes = []
        all_texts = []
        for frame in frames:
            bboxes, texts = self.detect_frame(frame)
            all_bboxes.append(bboxes)
            all_texts.append(texts)
        return all_bboxes, all_texts

    def detect_frame(self, frame):
        """
        Returns:
            detections: list of (x1,y1,x2,y2)
            texts:      list of recognized strings per detection
        """
        results = self.model.predict(frame)[0]
        detections = []
        texts = []

        # Resolve class names robustly
        names = getattr(results, "names", {})
        try:
            num_classes = len(names)
        except Exception:
            num_classes = None

        for box in results.boxes:
            # Class id/name resolution
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

            # bbox
            x1, y1, x2, y2 = map(int, box.xyxy.tolist()[0])
            crop = frame[y1:y2, x1:x2]

            # --- OCR with an ensemble of preprocessing variants (kana-friendly) ---
            plate_str = self._ocr_plate_ensemble(crop)

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

    # --------- helpers ---------
    def _ocr_plate_ensemble(self, crop_bgr):
        """
        Build several preprocessed variants that make kana more legible and
        choose the best OCR result by a simple kana+length score.
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return ""

        # base gray
        try:
            gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        except Exception:
            return ""

        # 1) CLAHE (contrast)
        try:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            g1 = clahe.apply(gray)
        except Exception:
            g1 = gray

        # 2) unsharp (edge boost)
        try:
            blur = cv2.GaussianBlur(g1, (0,0), 1.0)
            sharp = cv2.addWeighted(g1, 1.6, blur, -0.6, 0)
        except Exception:
            sharp = g1

        # 3) upscale aggressively (3x, kana is tiny)
        up = cv2.resize(sharp, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

        # 4) binarizations
        try:
            adap = cv2.adaptiveThreshold(up, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                         cv2.THRESH_BINARY, 31, 10)
        except Exception:
            adap = up
        try:
            _, otsu = cv2.threshold(up, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        except Exception:
            otsu = up

        # 5) gentle morphology to thicken kana strokes
        try:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
            adap2 = cv2.morphologyEx(adap, cv2.MORPH_CLOSE, kernel, iterations=1)
            otsu2 = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel, iterations=1)
        except Exception:
            adap2, otsu2 = adap, otsu

        # candidates for OCR (3-channel)
        candidates = [
            cv2.cvtColor(up,    cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(adap2, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(otsu2, cv2.COLOR_GRAY2BGR),
        ]

        best_lines = []
        best_score = -1

        for img in candidates:
            ocr_res = self.ocr.ocr(img)
            if self.debug_ocr:
                try:
                    print("DEBUG OCR_RES:", ocr_res)
                except Exception:
                    pass

            lines = self._parse_ocr_result(ocr_res)  # [(y_top, text), ...]
            # kana-weighted score
            kana_count = sum(1 for _, t in lines if KANA_RE.search(t or ""))
            total_len  = sum(len(t or "") for _, t in lines)
            score = kana_count * 3 + total_len

            if score > best_score:
                best_score = score
                best_lines = lines

        best_lines.sort(key=lambda x: x[0])
        return " ".join(t for _, t in best_lines).strip()

    def _parse_ocr_result(self, ocr_res):
        """
        Normalize PaddleOCR outputs across versions into a list of (y_top, text).
        """
        lines = []

        # Newer PaddleOCR dict style
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
                            box = rec_boxes[i]
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
                # Fallback single text
                single = entry.get("rec_text") or entry.get("text") or ""
                if single:
                    lines.append((0, str(single)))
            return lines

        # Older styles: list entries
        for entry in ocr_res or []:
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
