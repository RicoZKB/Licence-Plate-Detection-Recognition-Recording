import cv2
from detections import CarDetection, LicencePlateDetection

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open USB camera (index 0)")

    car_detector = CarDetection(model_path="yolo11n.pt")
    lp_detector  = LicencePlateDetection(model_path="models/best.pt")

    # Optional: save output
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    fps    = cap.get(cv2.CAP_PROP_FPS) or 20.0
    w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out    = cv2.VideoWriter("output.avi", fourcc, fps, (w, h))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Car detections
        car_dets = car_detector.detect_frames([frame], read_from_stub=False)[0]

        # Plate detections + OCR
        all_lp_dets, all_lp_texts = lp_detector.detect_frames([frame])
        lp_dets, lp_texts         = all_lp_dets[0], all_lp_texts[0]

        # Draw car boxes
        frame = car_detector.draw_bboxes([frame], [car_dets])[0]

        # Draw plate boxes & text
        frame = lp_detector.draw_bboxes([frame], [lp_dets], [lp_texts])[0]

        # Show and save
        cv2.imshow("Live Detection", frame)
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

