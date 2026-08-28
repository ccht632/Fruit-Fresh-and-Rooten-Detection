# Evaluates the trained SSD model on the test set with pycocotools (mAP, per-class AP).

import argparse
import io
import os
import contextlib

import torch
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader

from dataset import FruitDataset, get_transform, collate_fn, NUM_CLASSES, CATEGORY_ID_TO_NAME
from model import get_model
from confusion_matrix import build_confusion_matrix, CLASS_IDS, BACKGROUND_INDEX
from inference import load_model


@torch.no_grad()
def run_inference(model, data_loader, device):
    # Runs the test set through the model, formats results as COCO detections.
    model.eval()
    results = []

    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        outputs = model(images)

        for target, output in zip(targets, outputs):
            image_id = target["image_id"].item()
            boxes = output["boxes"].cpu()
            labels = output["labels"].cpu()
            scores = output["scores"].cpu()

            for box, label, score in zip(boxes, labels, scores):
                xmin, ymin, xmax, ymax = box.tolist()
                # xyxy -> xywh for COCO
                w = xmax - xmin
                h = ymax - ymin
                results.append({
                    "image_id": image_id,
                    "category_id": int(label.item()),
                    "bbox": [xmin, ymin, w, h],
                    "score": float(score.item()),
                })

    return results


def compute_mean_precision_recall(matrix):
    # Per-class P/R from the confusion matrix, averaged (Ultralytics-style).
    num_classes = len(CLASS_IDS)
    precisions, recalls = [], []

    for c in range(num_classes):
        tp = matrix[c, c]
        fp = matrix[c, :].sum() - tp
        fn = matrix[:, c].sum() - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)

    return sum(precisions) / num_classes, sum(recalls) / num_classes


def print_per_category_ap(coco_gt, coco_dt, cat_ids):
    # Runs COCOeval per class; returns {category_id: (ap_5095, ap_50)}.
    per_class_results = {}

    print("\nPer-class AP:")
    print(f"{'Class':<18}{'AP@[.5:.95]':<15}{'AP@0.5'}")

    for cat_id in cat_ids:
        eval_single = COCOeval(coco_gt, coco_dt, iouType="bbox")
        eval_single.params.catIds = [cat_id]

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            eval_single.evaluate()
            eval_single.accumulate()
            eval_single.summarize()

        ap_5095 = eval_single.stats[0]  # AP @ IoU=0.50:0.95
        ap_50 = eval_single.stats[1]    # AP @ IoU=0.50
        name = CATEGORY_ID_TO_NAME.get(cat_id, f"id_{cat_id}")
        per_class_results[cat_id] = (ap_5095, ap_50)
        print(f"{name:<18}{ap_5095:<15.3f}{ap_50:.3f}")

    return per_class_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="../data")
    parser.add_argument("--checkpoint", type=str, default="ssd_fruit_model.pth")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--score-threshold", type=float, default=0.0,
                         help="Filters low-confidence boxes; keep 0 for evaluation")
    parser.add_argument("--pr-score-thresh", type=float, default=0.5,
                         help="Fixed confidence threshold used for Precision/Recall (Ultralytics-style)")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"ERROR: checkpoint not found: {args.checkpoint}")
        print("Use --checkpoint to point to the correct .pth file")
        return
    if not os.path.isdir(args.data_root):
        print(f"ERROR: data directory not found: {args.data_root}")
        print("Use --data-root to point to the correct dataset path")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    try:
        test_dataset = FruitDataset(args.data_root, split="test", transforms=get_transform(train=False))
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, collate_fn=collate_fn,
    )
    print(f"Test set size: {len(test_dataset)}")

    # Model
    model = get_model(num_classes=NUM_CLASSES)
    try:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    except Exception as e:
        print(f"ERROR: failed to load checkpoint ({args.checkpoint}): {e}")
        return
    model.to(device)

    # Inference
    print("Running inference...")
    results = run_inference(model, test_loader, device)
    print(f"Generated {len(results)} predictions")

    if len(results) == 0:
        print("WARNING: no predictions generated; check the checkpoint loaded correctly")
        return

    # COCOeval
    coco_gt = test_dataset.coco
    coco_dt = coco_gt.loadRes(results)

    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()

    print("\n===== Overall evaluation =====")
    coco_eval.summarize()

    # Per-class AP
    cat_ids = list(CATEGORY_ID_TO_NAME.keys())
    per_class_results = print_per_category_ap(coco_gt, coco_dt, cat_ids)

    # Mean Precision / Recall
    print(f"\nComputing Precision/Recall (threshold={args.pr_score_thresh})...")
    model_for_pr = load_model(args.checkpoint, device)
    matrix = build_confusion_matrix(
        model_for_pr, test_loader, device,
        score_thresh=args.pr_score_thresh, iou_thresh=0.5
    )
    mean_precision, mean_recall = compute_mean_precision_recall(matrix)

    # Summary
    map_5095 = coco_eval.stats[0]
    map_50 = coco_eval.stats[1]
    map_75 = coco_eval.stats[2]

    print("\n" + "=" * 50)
    print("Summary table")
    print("=" * 50)
    print(f"{'Metric':<20}{'Value'}")
    print(f"{'mAP@0.5':<20}{map_50:.3f}")
    print(f"{'mAP@[0.5:0.95]':<20}{map_5095:.3f}")
    print(f"{'mAP@0.75':<20}{map_75:.3f}")
    print(f"{'Precision, mean':<20}{mean_precision:.3f}")
    print(f"{'Recall, mean':<20}{mean_recall:.3f}")

    print(f"\n{'Class':<18}{'AP@0.5':<12}{'AP@[0.5:0.95]'}")
    for cat_id in cat_ids:
        ap_5095, ap_50 = per_class_results[cat_id]
        name = CATEGORY_ID_TO_NAME[cat_id]
        print(f"{name:<18}{ap_50:<12.3f}{ap_5095:.3f}")


if __name__ == "__main__":
    main()
