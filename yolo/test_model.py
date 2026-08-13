"""
test_model.py

用训练好的YOLOv8模型，对整个test文件夹的图片做批量检测，
并把检测结果（画好框+标签的图片）保存到本地文件夹，方便逐张查看。

用法：
    python test_model.py
"""

from ultralytics import YOLO

# ---------------- 配置区，按你实际路径改 ----------------
MODEL_PATH = r"C:\Users\Cht632\Asgmnt RSW\RSWY2S1\AI - Fresh and Rooten Fruit Classification\runs\detect\fruit_yolo\fruit_freshness_v1\weights\best.pt"
TEST_IMAGES_DIR = r"C:\Users\Cht632\Asgmnt RSW\RSWY2S1\AI - Fresh and Rooten Fruit Classification\data\fresh and rotten fruits.v3i.yolov8\test\images"
CONFIDENCE_THRESHOLD = 0.5  # 置信度阈值，低于这个不显示（可以调整）
# ---------------------------------------------------------


def main():
    print(f"Loading model from: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    print(f"Running detection on all images in: {TEST_IMAGES_DIR}")
    results = model.predict(
        source=TEST_IMAGES_DIR,
        conf=CONFIDENCE_THRESHOLD,
        save=True,        # 自动保存画好框的结果图片
        show_labels=True,
        show_conf=True,
    )

    print(f"\nDone! Tested {len(results)} images.")
    print("Results (annotated images) are saved automatically.")
    print("Check the terminal output above for the exact save location")
    print('(usually something like: runs\\detect\\predict\\)')

    # 打印每张图检测到的结果摘要
    print("\n--- Detection Summary ---")
    for r in results:
        image_name = r.path.split("\\")[-1]
        print(f"\n📷 {image_name}")

        if len(r.boxes) == 0:
            print("   No detections")
        else:
            for i, box in enumerate(r.boxes, start=1):
                class_id = int(box.cls)
                class_name = model.names[class_id]
                confidence = float(box.conf) * 100  # 转成百分比更直观
                print(f"   Fruit {i}: {class_name} — Confidence: {confidence:.1f}%")


if __name__ == "__main__":
    main()