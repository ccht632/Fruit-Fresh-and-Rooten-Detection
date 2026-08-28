# YOLOv8 inference helpers used by the Streamlit app.

import os

import cv2
from ultralytics import YOLO

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(
    SCRIPT_DIR, "..", "runs", "detect", "fruit_yolo", "fruit_freshness_v1", "weights", "best.pt"
)


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
            f"Check the path is correct, or point to the actual location of best.pt"
        )
    return YOLO(model_path)
