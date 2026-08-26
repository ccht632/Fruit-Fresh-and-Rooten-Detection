"""
camera_app_yolo.py

Real-time fruit freshness detection using the trained YOLOv8 model over a webcam feed.
Draws bounding boxes, labels (Fresh/Rotten + fruit type), and confidence scores on
each frame. Background blur is a purely cosmetic display effect applied after
detection and does not affect the detection results themselves.

Usage:
    python camera_app_yolo.py
    python camera_app_yolo.py --threshold 0.3       # lower the threshold on the spot to check for missed low-confidence detections
    python camera_app_yolo.py --no-blur              # disable background blur

Press 'q' to quit.
"""

import argparse
import os

import cv2
import numpy as np
from ultralytics import YOLO

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(
    SCRIPT_DIR, "..", "runs", "detect", "fruit_yolo", "fruit_freshness_v1", "weights", "best.pt"
)


def blur_background(frame, boxes):
    blurred = cv2.GaussianBlur(frame, (35, 35), 0)
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        mask[y1:y2, x1:x2] = 255

    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    result = np.where(mask_3ch == 255, frame, blurred)
    return result


def draw_detections(frame_bgr, boxes, names):
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        name = names[cls_id]
        color_bgr = (0, 255, 0) if name.startswith("Fresh") else (0, 0, 255)

        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color_bgr, 2)

        text = f"{name} {conf:.2f}"
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame_bgr, (x1, y1 - text_h - 8), (x1 + text_w + 4, y1), color_bgr, -1)
        cv2.putText(frame_bgr, text, (x1 + 2, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    return frame_bgr


def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            f"Check the path is correct, or use --model-path to point to the actual location of best.pt"
        )
    return YOLO(model_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH,
                         help="Path to the YOLO weights file (best.pt)")
    parser.add_argument("--camera-index", type=int, default=0,
                         help="Camera device index, usually 0 is the default camera")
    parser.add_argument("--threshold", type=float, default=0.7,
                         help="Confidence threshold; lower values detect more but may also increase false positives")
    parser.add_argument("--no-blur", action="store_true",
                         help="Disable background blur (enabled by default). Note: this is purely a display effect and does not affect the detection results themselves")
    parser.add_argument("--max-hold-frames", type=int, default=5,
                         help="When no fruit is detected, how many frames to keep showing the previous boxes for, to reduce flicker")
    args = parser.parse_args()

    blur_enabled = not args.no_blur

    print("Loading model...")
    try:
        model = load_model(args.model_path)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return
    except Exception as e:
        print(f"ERROR: failed to load model: {e}")
        return

    print(f"Opening camera (index {args.camera_index})...")
    cap = cv2.VideoCapture(args.camera_index)

    if not cap.isOpened():
        print("ERROR: Could not open camera/video source.")
        print("If you don't have a webcam, change --camera-index to a video file path instead.")
        return

    print(f"Starting real-time detection (threshold={args.threshold}, blur={blur_enabled}). Press 'q' to quit.")

    last_annotated_frame = None
    frames_since_detection = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("End of video / camera feed lost.")
                break

            results = model(frame, conf=args.threshold, verbose=False)

            has_detection = len(results[0].boxes) > 0

            if has_detection:
                display_frame = blur_background(frame, results[0].boxes) if blur_enabled else frame.copy()
                annotated_frame = draw_detections(display_frame, results[0].boxes, model.names)

                last_annotated_frame = annotated_frame
                frames_since_detection = 0
            else:
                if last_annotated_frame is not None and frames_since_detection < args.max_hold_frames:
                    annotated_frame = last_annotated_frame
                    frames_since_detection += 1
                else:
                    annotated_frame = frame
                    last_annotated_frame = None

            cv2.imshow("Fruit Freshness Detector (Press 'q' to quit)", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camera closed.")


if __name__ == "__main__":
    main()
