"""
visualize_dataset.py

Randomly samples a few images from the YOLO-format training data and draws
their annotation boxes and class names, for manually checking that the
annotations (box position, class match) are correct.

Usage:
    python visualize_dataset.py
    python visualize_dataset.py --split test --num-samples 10
"""

import argparse
import os
import random
import glob
import yaml
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Default path is computed relative to this file's location, so no code changes
# are needed when switching computers or usernames
DEFAULT_DATA_YAML = os.path.join(
    SCRIPT_DIR, "..", "data", "fresh and rotten fruits.v3i.yolov8", "data.yaml"
)


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

            # YOLO format uses normalised coordinates (0-1), need to convert to actual pixel coordinates
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

    # hide any unused empty subplots
    for ax in axes[len(samples):]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-yaml", type=str, default=DEFAULT_DATA_YAML,
                         help="Path to the dataset's data.yaml")
    parser.add_argument("--split", type=str, default="valid", choices=["train", "valid", "test"],
                         help="Which split to check")
    parser.add_argument("--num-samples", type=int, default=6, help="Number of samples to draw")
    args = parser.parse_args()

    if not os.path.exists(args.data_yaml):
        print(f"ERROR: data.yaml not found: {args.data_yaml}")
        print("Use --data-yaml to point to the actual path")
        return

    class_names = load_class_names(args.data_yaml)
    print(f"Classes: {class_names}")

    pairs = find_image_label_pairs(args.data_yaml, args.split)
    print(f"Found {len(pairs)} image-label pairs in '{args.split}' split")

    if len(pairs) == 0:
        print("WARNING: No image-label pairs found. Check your --data-yaml and --split.")
    else:
        visualize_samples(pairs, class_names, num_samples=args.num_samples)


if __name__ == "__main__":
    main()
