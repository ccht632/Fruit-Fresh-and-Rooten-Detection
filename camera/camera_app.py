"""
camera_app.py

用训练好的YOLOv8模型，通过摄像头实时检测水果，
判断是Fresh还是Rotten，并在画面上画框+显示标签+置信度。

用法：
    python camera_app.py

按 'q' 退出。
"""

import cv2
from ultralytics import YOLO

# ---------------- 配置区，按你实际路径改 ----------------
MODEL_PATH = r"C:\Users\Cht632\Asgmnt RSW\RSWY2S1\AI - Fresh and Rooten Fruit Classification\runs\detect\fruit_yolo\fruit_freshness_v1\weights\best.pt"
CAMERA_INDEX = 0  # 0 = 电脑默认摄像头。如果没有摄像头，可以改成一个视频文件路径，比如 r"test_video.mp4"
CONFIDENCE_THRESHOLD = 0.5
# ---------------------------------------------------------


def main():
    print("Loading model...")
    model = YOLO(MODEL_PATH)

    print(f"Opening camera (index {CAMERA_INDEX})...")
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print("ERROR: Could not open camera/video source.")
        print("If you don't have a webcam, change CAMERA_INDEX to a video file path instead.")
        return

    print("Starting real-time detection. Press 'q' to quit.")

    last_annotated_frame = None
    frames_since_detection = 0
    MAX_FRAMES_TO_HOLD = 5  # 没检测到的时候，最多保留之前的框几帧，减少闪烁

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video / camera feed lost.")
            break

        # 跑YOLO检测
        results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)

        has_detection = len(results[0].boxes) > 0

        if has_detection:
            annotated_frame = results[0].plot()
            last_annotated_frame = annotated_frame
            frames_since_detection = 0
        else:
            # 没检测到，短暂保留上一次的画面，避免闪烁
            if last_annotated_frame is not None and frames_since_detection < MAX_FRAMES_TO_HOLD:
                annotated_frame = last_annotated_frame
                frames_since_detection += 1
            else:
                annotated_frame = frame
                last_annotated_frame = None

        cv2.imshow("Fruit Freshness Detector (Press 'q' to quit)", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Camera closed.")


if __name__ == "__main__":
    main()