"""
confusion_matrix.py

在test set上生成SSD模型的混淆矩阵，格式对齐队友YOLO那张图：
7x7矩阵（6个水果类别 + background），
matrix[pred][true] 记录"预测类别 vs 真实类别"的匹配数量。

匹配逻辑（和Ultralytics/YOLO内部逻辑一致的标准做法）：
- 每张图片里，预测框和真实框(GT)按IoU做贪心匹配（IoU >= 阈值才算匹配上）
- 匹配上的一对 (预测类别, 真实类别) -> matrix[pred][true] += 1
  （如果预测类别和真实类别一样，就落在对角线上，即正确检测）
- 没有匹配到任何GT的预测框（误报/false positive）-> matrix[pred][background] += 1
- 没有被任何预测框匹配到的GT（漏检/false negative） -> matrix[background][true] += 1

用法：
    python confusion_matrix.py --checkpoint ssd_fruit_model.pth
"""

import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from dataset import FruitDataset, get_transform, collate_fn, NUM_CLASSES, CATEGORY_ID_TO_NAME
from inference import load_model, predict_image


CLASS_IDS = list(CATEGORY_ID_TO_NAME.keys())  # [1,2,3,4,5,6]
CLASS_NAMES = [CATEGORY_ID_TO_NAME[i] for i in CLASS_IDS] + ["background"]
BACKGROUND_INDEX = len(CLASS_IDS)  # 矩阵里background所在的行/列索引 (=6)


def box_iou(box1, box2):
    """计算两个box的IoU，box格式为[xmin, ymin, xmax, ymax]"""
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
    """category_id(1-6) -> 矩阵索引(0-5)"""
    return CLASS_IDS.index(category_id)


def match_predictions_to_gt(pred_boxes, pred_labels, gt_boxes, gt_labels, iou_thresh=0.5):
    """
    贪心匹配：按IoU从高到低，依次把预测框配对给最匹配的真实框。
    返回：
        matched_pairs: [(pred_label, gt_label), ...] 匹配上的(预测类别,真实类别)对
        unmatched_pred_labels: 没匹配到任何GT的预测框类别列表 (false positive)
        unmatched_gt_labels: 没被任何预测框匹配到的GT类别列表 (false negative / 漏检)
    """
    num_pred = len(pred_boxes)
    num_gt = len(gt_boxes)

    if num_pred == 0:
        return [], [], list(gt_labels)
    if num_gt == 0:
        return [], list(pred_labels), []

    # 算出所有预测框和GT框两两之间的IoU
    iou_matrix = np.zeros((num_pred, num_gt))
    for i in range(num_pred):
        for j in range(num_gt):
            iou_matrix[i, j] = box_iou(pred_boxes[i], gt_boxes[j])

    matched_pairs = []
    used_pred = set()
    used_gt = set()

    # 贪心：每次找全局IoU最大的一对，配对后移除，直到没有IoU>=阈值的对为止
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
                matrix[p_idx, BACKGROUND_INDEX] += 1  # 误报：预测了但实际是background

            for gt_label in unmatched_gt:
                g_idx = class_id_to_matrix_index(gt_label)
                matrix[BACKGROUND_INDEX, g_idx] += 1  # 漏检：真实存在但预测成了background

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
    print(f"混淆矩阵图已保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="../data")
    parser.add_argument("--checkpoint", type=str, default="ssd_fruit_model.pth")
    parser.add_argument("--score-thresh", type=float, default=0.5)
    parser.add_argument("--iou-thresh", type=float, default=0.5,
                         help="判定预测框和真实框算“匹配上”的最低IoU要求")
    parser.add_argument("--output", type=str, default="confusion_matrix.png")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    test_dataset = FruitDataset(args.data_root, split="test", transforms=get_transform(train=False))
    test_loader = DataLoader(
        test_dataset, batch_size=4, shuffle=False, num_workers=0, collate_fn=collate_fn
    )
    print(f"Test集图片数量: {len(test_dataset)}")

    model = load_model(args.checkpoint, device)

    print("正在计算混淆矩阵...")
    matrix = build_confusion_matrix(
        model, test_loader, device,
        score_thresh=args.score_thresh, iou_thresh=args.iou_thresh
    )

    print("\n混淆矩阵 (行=预测类别, 列=真实类别):")
    print(" " * 16 + "".join(f"{n[:10]:>12}" for n in CLASS_NAMES))
    for i, row_name in enumerate(CLASS_NAMES):
        print(f"{row_name:<16}" + "".join(f"{v:>12}" for v in matrix[i]))

    plot_confusion_matrix(matrix, args.output)


if __name__ == "__main__":
    main()