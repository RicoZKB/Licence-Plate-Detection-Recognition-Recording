# -*- coding: utf-8 -*-
import os
import re
import cv2
import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR

# --- Pillow overlay (Japanese-safe text drawing) ---
try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

# Set to a Japanese-capable TTF/OTF (NotoSansCJK etc.)
FONT_PATH = os.environ.get("JP_FONT_PATH", "")  # e.g. "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc"

KANA_RE = re.compile(u"[ぁ-ゖァ-ヿ]")

class LicencePlateDetection:
    def __init__(self, model_path, verbose=False, debug_ocr=False):
        """
        model_path: path to a YOLO model trained to detect license plates
        verbose: print basic detections
        debug_ocr: print raw OCR results to terminal
        """
        self.model = YOLO(model_path)
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

        names = getattr(results, "names", {})
        try:
            num_classes = len(names)
        except Exception:
            num_classes = None

        for box in results.boxes:
            # Resolve class id -> name
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

            x1, y1, x2, y2 = map(int, box.xyxy.tolist()[0])
            crop = frame[y1:y2, x1:x2]

            # OCR with kana-friendly ensemble
            plate_str = self._ocr_plate_ensemble(crop)

            if self.verbose:
                try:
                    print(f"[DET] bbox=({x1},{y1},{x2},{y2}) text='{plate_str}'")
                except Exception:
                    pass

            detections.append((x1, y1, x2, y2))
            texts.append(plate_str)

        return detections, texts

    def draw_bboxes(
        self,
        video_frames,
        licence_plate_detections,
        licence_plate_texts,
        use_pillow=True,                 # <—— accept keyword (fixes your error)
        font_path: str = FONT_PATH,
        text_scale=0.9,
        text_thickness=2
    ):
        """
        Draw bboxes + text. If Pillow with JP font is available, we use it (kana-safe).
        Else we fallback to cv2.putText (may not render kana on some systems).
        """
        output_frames = []
        do_pillow = bool(use_pillow and PIL_AVAILABLE and font_path and os.path.exists(font_path))

        for frame, bbox_list, text_list in zip(video_frames, licence_plate_detections, licence_plate_texts):
            if do_pillow:
                frame = self._draw_with_pillow(frame, bbox_list, text_list, font_path)
            else:
                # Fallback to OpenCV
                for (x1, y1, x2, y2), text in zip(bbox_list, text_list):
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                    cv2.putText(frame, text, (x1, max(10, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, text_scale, (255,255,0), text_thickness, cv2.LINE_AA)
            output_frames.append(frame)
        return output_frames

    # --------- helpers ---------
    def _ocr_plate_ensemble(self, crop_bgr):
        """
        Build several preprocessed variants that make kana more legible and
        choose the best OCR result by a kana-weighted score.
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return ""

        # base gray
        try:
            gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        except Exception:
            return ""

        # 1) CLAHE
        try:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            g1 = clahe.apply(gray)
        except Exception:
            g1 = gray

        # 2) unsharp
        try:
            blur = cv2.GaussianBlur(g1, (0,0), 1.0)
            sharp = cv2.addWeighted(g1, 1.6, blur, -0.6, 0)
        except Exception:
            sharp = g1

        # 3) 3× upscale
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

        # 5) thin morphology CLOSE (thicken strokes)
        try:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
            adap2 = cv2.morphologyEx(adap, cv2.MORPH_CLOSE, kernel, iterations=1)
            otsu2 = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel, iterations=1)
        except Exception:
            adap2, otsu2 = adap, otsu

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
                    print("[OCR-RAW]", ocr_res)
                except Exception:
                    pass

            lines = self._parse_ocr_result(ocr_res)  # [(y_top, text), ...]
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
        Normalize PaddleOCR outputs into a list[(y_top, text)] across versions:
        - Case A: [pts, txt, conf]
        - Case B: [[pts], (txt, conf)]
        - Case C: dict-based (PaddleOCR v3+ with rec_texts/rec_text/text)
        """
        lines = []

        # Case C: dict style batch
        if (
            isinstance(ocr_res, list)
            and len(ocr_res) >= 1
            and isinstance(ocr_res[0], dict)
            and ("rec_texts" in ocr_res[0] or "rec_text" in ocr_res[0] or "text" in ocr_res[0])
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
                # single text
                single = entry.get("rec_text") or entry.get("text") or ""
                if single:
                    lines.append((0, str(single)))
            return lines

        # Older list styles
        for entry in ocr_res or []:
            # Case A: [pts, txt, conf]
            if isinstance(entry, (list, tuple)) and len(entry) == 3 and isinstance(entry[1], str):
                pts, txt = entry[0], entry[1]
            # Case B: [[pts], (txt, conf)]
            elif isinstance(entry, (list, tuple)) and len(entry) >= 2 and isinstance(entry[1], (list, tuple)):
                pts = entry[0][0] if isinstance(entry[0][0][0], (list, tuple)) else entry[0]
                txt = entry[1][0]
            # Extra dict fallbacks
            elif isinstance(entry, dict):
                pts = entry.get("box") or entry.get("bbox") or entry.get("boxes") or []
                recs = entry.get("rec_texts") or entry.get("rec_text") or entry.get("text")
                if isinstance(recs, list):
                    txt = recs[0] if recs else ""
                else:
                    txt = recs or ""
            else:
                continue

            if not txt:
                continue
            try:
                y_top = min(p[1] for p in pts) if pts else 0
            except Exception:
                y_top = 0
            lines.append((y_top, str(txt)))

        return lines

    def _draw_with_pillow(self, frame_bgr, bbox_list, text_list, font_path: str):
        """Kana-safe overlay using Pillow; returns BGR frame."""
        # Convert to PIL Image (RGB)
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(pil_img)

        # Choose font size by height heuristic
        H = frame_bgr.shape[0]
        base_size = max(14, int(H * 0.03))
        try:
            font = ImageFont.truetype(font_path, base_size)
        except Exception:
            # Fallback to default PIL font (may not have JP glyphs)
            font = ImageFont.load_default()

        for (x1, y1, x2, y2), text in zip(bbox_list, text_list):
            # rectangle
            draw.rectangle([x1, y1, x2, y2], outline=(0,255,0), width=2)
            # text background for readability
            label = str(text or "")
            bbox = draw.textbbox((0,0), label, font=font)
            tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
            ty = max(0, y1 - th - 2)
            draw.rectangle([x1, ty, x1 + tw + 4, ty + th + 2], fill=(0,0,0,180))
            draw.text((x1 + 2, ty + 1), label, fill=(255,255,0), font=font)

        # Back to BGR numpy
        out_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return out_bgr
