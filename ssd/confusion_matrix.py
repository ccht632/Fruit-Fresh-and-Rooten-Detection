# Builds the SSD confusion matrix on the test set (6 classes + background, matrix[pred][true]).

import argparse
import os

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from dataset import FruitDataset, get_transform, collate_fn, NUM_CLASSES, CATEGORY_ID_TO_NAME
from inference import load_model, predict_image


CLASS_IDS = list(CATEGORY_ID_TO_NAME.keys())  # [1..6]
CLASS_NAMES = [CATEGORY_ID_TO_NAME[i] for i in CLASS_IDS] + ["background"]
BACKGROUND_INDEX = len(CLASS_IDS)  # background row/col index


def box_iou(box1, box2):
    # IoU of two [xmin, ymin, xmax, ymax] boxes.
    xmin = max(box1[0], box2[0])
    ymin = max(box1[1], box2[1])
    xmax = min(box1[2], box2[2])
    ymax = min(box1[3], box2[3])

    inter_area = max(0, xmax - xmin) * max(0, ymax - ymin)
    if inter_area == 0:
        return 0.0

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area

    return inter_area / union_area


def class_id_to_matrix_index(category_id):
    # category_id (1-6) -> matrix index (0-5)
    return CLASS_IDS.index(category_id)


def match_predictions_to_gt(pred_boxes, pred_labels, gt_boxes, gt_labels, iou_thresh=0.5):
    # Greedy IoU matching; returns matched pairs, unmatched preds (FP), unmatched GT (FN).
    num_pred = len(pred_boxes)
    num_gt = len(gt_boxes)

    if num_pred == 0:
        return [], [], list(gt_labels)
    if num_gt == 0:
        return [], list(pred_labels), []

    # Pairwise IoU
    iou_matrix = np.zeros((num_pred, num_gt))
    for i in range(num_pred):
        for j in range(num_gt):
            iou_matrix[i, j] = box_iou(pred_boxes[i], gt_boxes[j])

    matched_pairs = []
    used_pred = set()
    used_gt = set()

    # Greedy: highest-IoU pair first
    while True:
        i, j = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
        if iou_matrix[i, j] < iou_thresh:
            break
        matched_pairs.append((pred_labels[i], gt_labels[j]))
        used_pred.add(i)
        used_gt.add(j)
        iou_matrix[i, :] = -1
        iou_matrix[:, j] = -1

    unmatched_pred_labels = [pred_labels[i] for i in range(num_pred) if i not in used_pred]
    unmatched_gt_labels = [gt_labels[j] for j in range(num_gt) if j not in used_gt]

    return matched_pairs, unmatched_pred_labels, unmatched_gt_labels


def build_confusion_matrix(model, test_loader, device, score_thresh=0.5, iou_thresh=0.5):
    matrix = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=int)

    for images, targets in test_loader:
        for image_tensor, target in zip(images, targets):
            from PIL import Image
            import torchvision.transforms.functional as TF
            pil_image = TF.to_pil_image(image_tensor)

            pred_boxes, pred_labels, pred_scores = predict_image(
                model, pil_image, device, score_threshold=score_thresh
            )
            pred_boxes = pred_boxes.tolist()
            pred_labels = pred_labels.tolist()

            gt_boxes = target["boxes"].tolist()
            gt_labels = target["labels"].tolist()

            matched_pairs, unmatched_pred, unmatched_gt = match_predictions_to_gt(
                pred_boxes, pred_labels, gt_boxes, gt_labels, iou_thresh
            )

            for pred_label, gt_label in matched_pairs:
                p_idx = class_id_to_matrix_index(pred_label)
                g_idx = class_id_to_matrix_index(gt_label)
                matrix[p_idx, g_idx] += 1

            for pred_label in unmatched_pred:
                p_idx = class_id_to_matrix_index(pred_label)
                matrix[p_idx, BACKGROUND_INDEX] += 1  # false positive

            for gt_label in unmatched_gt:
                g_idx = class_id_to_matrix_index(gt_label)
                matrix[BACKGROUND_INDEX, g_idx] += 1  # false negative

    return matrix


def plot_confusion_matrix(matrix, output_path):
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matrix, cmap="Blues")

    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=90)
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("True")
    ax.set_title("Confusion Matrix (SSD300-VGG16)")

    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            value = matrix[i, j]
            if value > 0:
                color = "white" if value > matrix.max() / 2 else "black"
                ax.text(j, i, str(value), ha="center", va="center", color=color)

    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Confusion matrix saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="../data")
    parser.add_argument("--checkpoint", type=str, default="ssd_fruit_model.pth")
    parser.add_argument("--score-thresh", type=float, default=0.5)
    parser.add_argument("--iou-thresh", type=float, default=0.5,
                         help="Minimum IoU for a prediction/GT box to count as matched")
    parser.add_argument("--output", type=str, default="confusion_matrix.png")
    args = parser.parse_args()

    if not os.path.isdir(args.data_root):
        print(f"ERROR: data directory not found: {args.data_root}")
        print("Use --data-root to point to the correct dataset path")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    try:
        test_dataset = FruitDataset(args.data_root, split="test", transforms=get_transform(train=False))
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return
    test_loader = DataLoader(
        test_dataset, batch_size=4, shuffle=False, num_workers=0, collate_fn=collate_fn
    )
    print(f"Test set size: {len(test_dataset)}")

    try:
        model = load_model(args.checkpoint, device)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"ERROR: {e}")
        return

    print("Computing confusion matrix...")
    matrix = build_confusion_matrix(
        model, test_loader, device,
        score_thresh=args.score_thresh, iou_thresh=args.iou_thresh
    )

    print("\nConfusion matrix (rows=Predicted, cols=True):")
    print(" " * 16 + "".join(f"{n[:10]:>12}" for n in CLASS_NAMES))
    for i, row_name in enumerate(CLASS_NAMES):
        print(f"{row_name:<16}" + "".join(f"{v:>12}" for v in matrix[i]))

    plot_confusion_matrix(matrix, args.output)


if __name__ == "__main__":
    main()
