# Builds YOLOv8's confusion matrix (same layout as ssd/confusion_matrix.py).

import argparse
import os
import shutil

from ultralytics import YOLO

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(
    SCRIPT_DIR, "..", "runs", "detect", "fruit_yolo", "fruit_freshness_v1", "weights", "best.pt"
)
DEFAULT_DATA_YAML = os.path.join(
    SCRIPT_DIR, "..", "data", "Yolov8n", "fresh and rotten fruits.v3i.yolov8", "data.yaml"
)


def format_matrix(matrix, class_names):
    lines = []
    lines.append(" " * 16 + "".join(f"{n[:10]:>12}" for n in class_names))
    for i, row_name in enumerate(class_names):
        lines.append(f"{row_name:<16}" + "".join(f"{v:>12}" for v in matrix[i]))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data-yaml", type=str, default=DEFAULT_DATA_YAML)
    parser.add_argument("--output-png", type=str, default=os.path.join(SCRIPT_DIR, "confusion_matrix.png"))
    parser.add_argument("--output-txt", type=str, default=os.path.join(SCRIPT_DIR, "confusion_matrix.txt"))
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
    results = model.val(data=args.data_yaml, split="test", verbose=False)

    cm = results.confusion_matrix
    class_names = [model.names[i] for i in sorted(model.names)] + ["background"]
    matrix = cm.matrix.astype(int)

    text_output = format_matrix(matrix, class_names)
    print("\nConfusion matrix (rows=Predicted, cols=True):")
    print(text_output)

    with open(args.output_txt, "w", encoding="utf-8") as f:
        f.write("Confusion matrix (rows=Predicted, cols=True)\n\n")
        f.write(text_output + "\n")
    print(f"\nSaved matrix as text to: {args.output_txt}")

    # copy plot next to script
    generated_png = os.path.join(results.save_dir, "confusion_matrix.png")
    if os.path.exists(generated_png):
        shutil.copy(generated_png, args.output_png)
        print(f"Copied plot to: {args.output_png}")


if __name__ == "__main__":
    main()
