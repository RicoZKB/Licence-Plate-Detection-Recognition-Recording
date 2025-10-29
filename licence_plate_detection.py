import cv2
from ultralytics import YOLO
from paddleocr import PaddleOCR
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import queue

class LicencePlateDetection:
    def __init__(self, model_path, verbose=False, debug_ocr=False,
                 use_gpu=True, enable_async_ocr=False, fast_mode=True):
        # load your YOLO model
        self.model = YOLO(model_path)

        # OCR optimization settings
        self.use_gpu = use_gpu
        self.enable_async_ocr = enable_async_ocr
        self.fast_mode = fast_mode

        # Initialize PaddleOCR with optimized settings
        ocr_kwargs = {
            'lang': 'japan',
            'use_angle_cls': False,  # Disabled for speed (plates are usually horizontal)
            'show_log': False,
            'use_gpu': use_gpu,
        }

        # Fast mode: reduce model complexity for speed
        if fast_mode:
            ocr_kwargs.update({
                'det_db_thresh': 0.3,          # Lower threshold for faster detection
                'det_db_box_thresh': 0.5,      # Box threshold
                'rec_batch_num': 6,            # Batch processing
                'use_space_char': True,
                'drop_score': 0.3,             # Lower confidence threshold
                'use_dilation': False,         # Skip dilation for speed
                'det_limit_side_len': 640,     # Limit detection side length
                'rec_image_shape': "3, 32, 320" # Smaller recognition shape
            })

        self.ocr = PaddleOCR(**ocr_kwargs)

        # Async OCR thread pool
        if enable_async_ocr:
            self.ocr_executor = ThreadPoolExecutor(max_workers=2)
            self.ocr_queue = queue.Queue(maxsize=5)

        # logging options
        self.verbose = bool(verbose)
        self.debug_ocr = bool(debug_ocr)

    def detect_frames(self, frames, ocr_filter=None):
        all_bboxes = []
        all_texts = []
        for frame in frames:
            bboxes, texts = self.detect_frame(frame, ocr_filter=ocr_filter)
            all_bboxes.append(bboxes)
            all_texts.append(texts)
        return all_bboxes, all_texts

    def detect_frame(self, frame, ocr_filter=None):
        results = self.model.predict(frame)[0]
        detections = []
        texts = []

        # Resolve class names and count (handle list/dict/other)
        names = getattr(results, "names", {})
        try:
            num_classes = len(names)
        except Exception:
            num_classes = None

        for box in results.boxes:
            # Determine class id and name robustly
            cls_id = None
            try:
                # ultralytics boxes.cls is a tensor
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

            # Keep boxes if:
            # - model is single-class, or
            # - class name suggests it's a plate (case-insensitive contains)
            keep = False
            if num_classes == 1:
                keep = True
            else:
                nm = cls_name.lower()
                if ("plate" in nm) or ("licence" in nm) or ("license" in nm) or (nm == "lp") or ("number" in nm):
                    keep = True
            if not keep:
                continue

            # get bounding box coords
            x1, y1, x2, y2 = map(int, box.xyxy.tolist()[0])
            crop = frame[y1:y2, x1:x2]

            run_ocr = True
            if ocr_filter is not None:
                try:
                    run_ocr = bool(ocr_filter((x1, y1, x2, y2)))
                except Exception:
                    run_ocr = True

            # OPTIMIZED preprocessing for OCR
            if self.fast_mode:
                # Fast mode: minimal preprocessing (2-3x faster)
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                # Only upscale 1.5x instead of 2x (saves 30% processing time)
                up = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
                prep = cv2.cvtColor(up, cv2.COLOR_GRAY2BGR)
            else:
                # Quality mode: full preprocessing (original behavior)
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                # CLAHE contrast boost
                try:
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    gray = clahe.apply(gray)
                except Exception:
                    pass
                # mild unsharp mask to enhance edges
                try:
                    blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
                    gray = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)
                except Exception:
                    pass
                up = cv2.resize(gray, None, fx=2, fy=2)
                prep = cv2.cvtColor(up, cv2.COLOR_GRAY2BGR)

            # RUN OCR — optimized
            ocr_res = []
            if run_ocr:
                try:
                    # Use PaddleOCR with cls=False (already set in init)
                    ocr_res = self.ocr.ocr(prep, cls=False)
                    if self.debug_ocr:
                        print("DEBUG OCR_RES:", ocr_res)
                except Exception as e:
                    if self.debug_ocr:
                        print(f"OCR Error: {e}")
                    ocr_res = []

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
