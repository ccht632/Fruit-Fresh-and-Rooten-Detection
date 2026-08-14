"""
camera_demo.py

实时摄像头 Fresh/Rotten Fruit Detection demo。
打开电脑摄像头，实时框出画面中的水果并显示 "Fresh Apple" / "Rotten Banana" 等标签。

用法：
    python camera_demo.py --checkpoint ../ssd/ssd_fruit_model.pth

按 'q' 键退出。

注意：这是CPU推理，SSD300对每一帧做完整推理会有明显延迟，
所以默认每隔几帧才重新推理一次（中间帧复用上一次的检测框），
让画面看起来更流畅，同时不会因为逐帧推理拖慢摄像头显示速度。
"""

import argparse
import os
import sys
import time

import cv2
import torch
from PIL import Image

# camera/ 和 ssd/ 是同级文件夹，需要把 ssd/ 加入模块搜索路径才能 import
SSD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ssd")
sys.path.insert(0, SSD_DIR)

from inference import load_model, predict_image, get_box_color  # noqa: E402
from dataset import CATEGORY_ID_TO_NAME  # noqa: E402


def draw_on_frame(frame_bgr, boxes, labels, scores):
    """
    直接在 OpenCV 的 BGR numpy frame 上画框，
    比每帧转成PIL画完再转回来更快（减少CPU开销，摄像头场景下流畅度更重要）。
    """
    for box, label, score in zip(boxes, labels, scores):
        xmin, ymin, xmax, ymax = [int(v) for v in box.tolist()]
        category_id = int(label.item())
        name = CATEGORY_ID_TO_NAME.get(category_id, f"id_{category_id}")

        # PIL用的颜色名字("lime"/"red")，OpenCV要用BGR数值，这里单独映射一次
        color_bgr = (0, 255, 0) if name.startswith("Fresh") else (0, 0, 255)

        cv2.rectangle(frame_bgr, (xmin, ymin), (xmax, ymax), color_bgr, 2)

        text = f"{name} {score:.2f}"
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame_bgr, (xmin, ymin - text_h - 8), (xmin + text_w + 4, ymin), color_bgr, -1)
        cv2.putText(frame_bgr, text, (xmin + 2, ymin - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    return frame_bgr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="../ssd/ssd_fruit_model.pth")
    parser.add_argument("--threshold", type=float, default=0.5, help="置信度阈值")
    parser.add_argument("--nms-thresh", type=float, default=0.2,
                         help="NMS去重阈值，默认0.2（经测试对紧贴水果的框重叠问题有改善，同时不影响独立水果的检测）")
    parser.add_argument("--camera-index", type=int, default=0, help="摄像头设备编号，通常0是默认摄像头")
    parser.add_argument("--skip-frames", type=int, default=2,
                         help="每隔多少帧重新推理一次，数值越大画面越流畅但检测框更新越慢（CPU推理建议保留>=2）")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    model = load_model(args.checkpoint, device, nms_thresh=args.nms_thresh)

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        print(f"⚠️ 无法打开摄像头(index={args.camera_index})，请检查摄像头连接或换一个index试试")
        return

    print("摄像头已打开，按 'q' 键退出")

    frame_count = 0
    last_boxes, last_labels, last_scores = torch.empty(0, 4), torch.empty(0), torch.empty(0)
    fps_start_time = time.time()
    fps_frame_count = 0
    fps_display = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ 读取摄像头画面失败，退出")
            break

        # 每隔 skip_frames 帧才重新推理一次，中间帧复用上一次的检测结果
        if frame_count % (args.skip_frames + 1) == 0:
            # OpenCV是BGR，模型需要RGB，这里转换后交给推理函数
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)

            last_boxes, last_labels, last_scores = predict_image(
                model, pil_image, device, score_threshold=args.threshold
            )

        frame = draw_on_frame(frame, last_boxes, last_labels, last_scores)

        # 简单计算并显示FPS，方便你判断当前设备的实时性是否够用
        fps_frame_count += 1
        elapsed = time.time() - fps_start_time
        if elapsed >= 1.0:
            fps_display = fps_frame_count / elapsed
            fps_frame_count = 0
            fps_start_time = time.time()
        cv2.putText(frame, f"FPS: {fps_display:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Fruit Freshness Detection (press 'q' to quit)", frame)
        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()