"""
visualize_dataset.py

从 FruitDataset 中随机抽样几张图片，画出标注框 + 类别标签，
用于人工核对标注是否正确（框的位置、类别是否匹配）。

用法：
    python visualize_dataset.py
"""

import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from dataset import FruitDataset, get_transform, CATEGORY_ID_TO_NAME


def visualize_samples(dataset, num_samples=6):
    # 随机抽样几个不重复的索引
    indices = random.sample(range(len(dataset)), num_samples)

    cols = 3
    rows = (num_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten()

    for ax, idx in zip(axes, indices):
        image, target = dataset[idx]

        # image是ToTensor后的tensor，shape为[C, H, W]，画图前要转回[H, W, C]
        img_np = image.permute(1, 2, 0).numpy()
        ax.imshow(img_np)

        boxes = target["boxes"]
        labels = target["labels"]

        for box, label in zip(boxes, labels):
            xmin, ymin, xmax, ymax = box.tolist()
            width = xmax - xmin
            height = ymax - ymin

            rect = patches.Rectangle(
                (xmin, ymin), width, height,
                linewidth=2, edgecolor="lime", facecolor="none"
            )
            ax.add_patch(rect)

            label_name = CATEGORY_ID_TO_NAME[label.item()]
            ax.text(
                xmin, ymin - 5, label_name,
                color="white", fontsize=9,
                bbox=dict(facecolor="green", alpha=0.7, pad=1)
            )

        ax.set_title(f"image_id: {target['image_id'].item()}")
        ax.axis("off")

    # 隐藏多余的空subplot
    for ax in axes[len(indices):]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    dataset = FruitDataset(root_dir="../data", split="train", transforms=get_transform(train=True))
    visualize_samples(dataset, num_samples=6)