"""
train_yolo_fruit.py

Trains a YOLOv8 model to detect fruit freshness/rottenness.
This version is meant to be run locally in VS Code (CPU or local GPU), and
includes several measures to reduce overfitting.

Usage:
    1. Make sure the dataset folder (containing data.yaml) is already in the project
    2. Run this script

Run:
    python train_yolo_fruit.py
    python train_yolo_fruit.py --data-yaml "D:\\xxx\\data.yaml"   # specify if the dataset is not in the default location
"""

import argparse
import os
from ultralytics import YOLO

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Default path is computed relative to this file's location (yolo/ and data/ are sibling folders),
# so no code changes are needed when switching computers or usernames
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

    model = YOLO("yolov8n.pt")  # start from pretrained weights, faster and less prone to overfitting than training from scratch

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        project=project_name,
        name=run_name,

        # ---- overfitting-prevention settings ----
        patience=20,             # early stopping: stop automatically after 20 epochs with no improvement, to stop the model from "memorising" the training set
        dropout=0.2,              # dropout: randomly drop some neurons during training, forcing the model to learn general patterns rather than memorising details
        weight_decay=0.0005,      # weight decay: penalises overly large weights, reducing overfitting to the training data

        # ---- data augmentation (further reduces overfitting risk by exposing the model to more variation) ----
        hsv_h=0.015,               # random hue jitter
        hsv_s=0.7,                 # random saturation jitter (freshness/rottenness relies heavily on colour, so this is kept but not too strong, to avoid distorting the key fresh/rotten colour cue)
        hsv_v=0.4,                 # random brightness jitter
        fliplr=0.5,                 # 50% chance of horizontal flip
        flipud=0.0,                 # no vertical flip (fruit is not normally placed upside down, so this is unnecessary)
        degrees=15,                 # random rotation angle (matches the +/-15 degree augmentation already present in the dataset)
        translate=0.1,               # random translation
        scale=0.5,                   # random scaling

        save=True,
        save_period=5,               # save a checkpoint every 5 epochs, so progress is not lost if something goes wrong mid-training
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
