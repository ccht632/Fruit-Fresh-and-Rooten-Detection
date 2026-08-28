# Evaluates YOLOv8 on the test set; regenerates eval_results.txt.

import argparse
import os

from ultralytics import YOLO

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(
    SCRIPT_DIR, "..", "runs", "detect", "fruit_yolo", "fruit_freshness_v1", "weights", "best.pt"
)
DEFAULT_DATA_YAML = os.path.join(
    SCRIPT_DIR, "..", "data", "Yolov8n", "fresh and rotten fruits.v3i.yolov8", "data.yaml"
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data-yaml", type=str, default=DEFAULT_DATA_YAML)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--output", type=str, default=os.path.join(SCRIPT_DIR, "eval_results.txt"))
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"ERROR: model file not found: {args.model_path}")
        print("Use --model-path to point to the actual location of best.pt")
        return
    if not os.path.exists(args.data_yaml):
        print(f"ERROR: data.yaml not found: {args.data_yaml}")
        print("Use --data-yaml to point to the actual location of the dataset config")
        return

    model = YOLO(args.model_path)
    results = model.val(data=args.data_yaml, split=args.split)

    box = results.box
    names = results.names
    nt_per_image = results.nt_per_image
    nt_per_class = results.nt_per_class

    lines = []
    lines.append(f"{'Class':<16}{'Images':<10}{'Instances':<12}{'P':<9}{'R':<9}{'mAP50':<9}{'mAP50-95'}")
    lines.append(
        f"{'all':<16}{sum(nt_per_image):<10}{sum(nt_per_class):<12}"
        f"{box.mp:<9.3f}{box.mr:<9.3f}{box.map50:<9.3f}{box.map:.3f}"
    )
    for idx, cls_idx in enumerate(box.ap_class_index):
        cls_idx = int(cls_idx)
        name = names[cls_idx]
        lines.append(
            f"{name:<16}{nt_per_image[cls_idx]:<10}{nt_per_class[cls_idx]:<12}"
            f"{box.p[idx]:<9.3f}{box.r[idx]:<9.3f}{box.ap50[idx]:<9.3f}{box.ap[idx]:.3f}"
        )
    lines.append("")

    lines.append("=" * 50)
    lines.append("Summary table (Metric / Value)")
    lines.append("=" * 50)
    lines.append(f"{'Metric':<20}{'Value'}")
    lines.append(f"{'mAP@0.5':<20}{box.map50:.3f}")
    lines.append(f"{'mAP@[0.5:0.95]':<20}{box.map:.3f}")
    lines.append(f"{'mAP@0.75':<20}{box.map75:.3f}")
    lines.append(f"{'Precision, mean':<20}{box.mp:.3f}")
    lines.append(f"{'Recall, mean':<20}{box.mr:.3f}")
    lines.append("")
    lines.append(f"{'Class':<18}{'AP@0.5':<12}{'AP@[0.5:0.95]'}")
    for idx, cls_idx in enumerate(box.ap_class_index):
        name = names[int(cls_idx)]
        lines.append(f"{name:<18}{box.ap50[idx]:<12.3f}{box.ap[idx]:.3f}")
    lines.append("")
    speed = results.speed
    total_ms = sum(speed.values())
    lines.append(
        f"Speed: {speed['preprocess']:.1f}ms preprocess, {speed['inference']:.1f}ms inference, "
        f"{speed['loss']:.1f}ms loss, {speed['postprocess']:.1f}ms postprocess per image "
        f"({total_ms:.1f}ms total)"
    )

    summary = "\n".join(lines)
    print("\n" + summary)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(summary + "\n")
    print(f"\nSaved summary to: {args.output}")


if __name__ == "__main__":
    main()
