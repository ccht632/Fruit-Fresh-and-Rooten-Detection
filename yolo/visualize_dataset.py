"""
visualize_dataset.py

从YOLO格式的训练数据里，随机抽样几张图片，画出标注框+类别名字，
用于人工核对标注是否正确（框的位置、类别是否匹配）。

用法：
    python visualize_dataset.py
"""

import os
import random
import glob
import yaml
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# ---------------- 配置区，按你实际路径改 ----------------
DATA_YAML = r"C:\Users\Cht632\Asgmnt RSW\RSWY2S1\AI - Fresh and Rooten Fruit Classification\data\fresh and rotten fruits.v3i.yolov8\data.yaml"
SPLIT = "valid"  # 检查哪个部分：train / valid / test
NUM_SAMPLES = 6  # 随机抽几张
# ---------------------------------------------------------


def load_class_names(data_yaml_path):
    with open(data_yaml_path, "r") as f:
        data = yaml.safe_load(f)
    return data["names"]


def find_image_label_pairs(data_yaml_path, split):
    base_dir = os.path.dirname(data_yaml_path)
    images_dir = os.path.join(base_dir, split, "images")
    labels_dir = os.path.join(base_dir, split, "labels")

    image_paths = glob.glob(os.path.join(images_dir, "*.jpg")) + \
                  glob.glob(os.path.join(images_dir, "*.png"))

    pairs = []
    for img_path in image_paths:
        basename = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(labels_dir, basename + ".txt")
        if os.path.exists(label_path):
            pairs.append((img_path, label_path))

    return pairs


def visualize_samples(pairs, class_names, num_samples=6):
    samples = random.sample(pairs, min(num_samples, len(pairs)))

    cols = 3
    rows = (len(samples) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten() if len(samples) > 1 else [axes]

    for ax, (img_path, label_path) in zip(axes, samples):
        image = Image.open(img_path).convert("RGB")
        img_w, img_h = image.size
        ax.imshow(image)

        with open(label_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            class_id = int(parts[0])
            x_center, y_center, box_w, box_h = map(float, parts[1:])

            # YOLO格式是比例坐标(0-1)，需要转换成实际像素坐标
            xmin = (x_center - box_w / 2) * img_w
            ymin = (y_center - box_h / 2) * img_h
            width = box_w * img_w
            height = box_h * img_h

            rect = patches.Rectangle(
                (xmin, ymin), width, height,
                linewidth=2, edgecolor="lime", facecolor="none"
            )
            ax.add_patch(rect)

            label_name = class_names[class_id]
            ax.text(
                xmin, ymin - 5, label_name,
                color="white", fontsize=9,
                bbox=dict(facecolor="green", alpha=0.7, pad=1)
            )

        ax.set_title(os.path.basename(img_path))
        ax.axis("off")

    # 隐藏多余的空subplot
    for ax in axes[len(samples):]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    class_names = load_class_names(DATA_YAML)
    print(f"Classes: {class_names}")

    pairs = find_image_label_pairs(DATA_YAML, SPLIT)
    print(f"Found {len(pairs)} image-label pairs in '{SPLIT}' split")

    if len(pairs) == 0:
        print("WARNING: No image-label pairs found. Check your DATA_YAML and SPLIT paths.")
    else:
        visualize_samples(pairs, class_names, num_samples=NUM_SAMPLES)