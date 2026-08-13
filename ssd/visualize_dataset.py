import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from dataset import FruitDataset, get_transform, CATEGORY_ID_TO_NAME


def visualize_samples(dataset, num_samples=6):
    indices = random.sample(range(len(dataset)), num_samples)

    cols = 3
    rows = (num_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten()

    for ax, idx in zip(axes, indices):
        image, target = dataset[idx]

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

    for ax in axes[len(indices):]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    dataset = FruitDataset(root_dir="../data", split="train", transforms=get_transform(train=True))
    visualize_samples(dataset, num_samples=6)