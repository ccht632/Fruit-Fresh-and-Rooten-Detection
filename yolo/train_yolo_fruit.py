# Trains a YOLOv8 model to detect fruit freshness/rottenness.

import argparse
import os
from ultralytics import YOLO

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_YAML = os.path.join(
    SCRIPT_DIR, "..", "data", "fresh and rotten fruits.v3i.yolov8", "data.yaml"
)


def train(data_yaml, epochs, img_size, batch_size, project_name, run_name):
    if not os.path.exists(data_yaml):
        raise FileNotFoundError(
            f"data.yaml not found: {data_yaml}\n"
            f"Check whether the dataset has been downloaded into the data/ folder, "
            f"or use --data-yaml to point to the actual path"
        )

    model = YOLO("yolov8n.pt")  # pretrained start

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        project=project_name,
        name=run_name,

        # Overfitting prevention
        patience=20,
        dropout=0.2,
        weight_decay=0.0005,

        # Data augmentation
        hsv_h=0.015,
        hsv_s=0.7,                 # kept mild to preserve fresh/rotten color cue
        hsv_v=0.4,
        fliplr=0.5,
        flipud=0.0,                 # fruit orientation is never upside down
        degrees=15,
        translate=0.1,
        scale=0.5,

        save=True,
        save_period=5,
    )

    print("Training done.")
    best_model_path = os.path.join(project_name, run_name, "weights", "best.pt")
    print(f"Best model saved at: {best_model_path}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-yaml", type=str, default=DEFAULT_DATA_YAML,
                         help="Path to the dataset's data.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--project-name", type=str, default="fruit_yolo")
    parser.add_argument("--run-name", type=str, default="fruit_freshness_v1")
    args = parser.parse_args()

    try:
        train(args.data_yaml, args.epochs, args.img_size, args.batch_size,
              args.project_name, args.run_name)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
    except Exception as e:
        print(f"ERROR: training failed: {e}")
        raise


if __name__ == "__main__":
    main()
