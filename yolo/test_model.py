"""
test_model.py

Runs the trained YOLOv8 model on every image in the test folder as a batch,
and saves the annotated (boxes + labels) results to a local folder for review.

Usage:
    python test_model.py
    python test_model.py --confidence 0.3
"""

import argparse
import os

from ultralytics import YOLO

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Default paths are computed relative to this file's location, so no code changes
# are needed when switching computers or usernames
DEFAULT_MODEL_PATH = os.path.join(
    SCRIPT_DIR, "..", "runs", "detect", "fruit_yolo", "fruit_freshness_v1", "weights", "best.pt"
)
DEFAULT_TEST_IMAGES_DIR = os.path.join(
    SCRIPT_DIR, "..", "data", "fresh and rotten fruits.v3i.yolov8", "test", "images"
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH,
                         help="Path to the YOLO weights file (best.pt)")
    parser.add_argument("--test-images-dir", type=str, default=DEFAULT_TEST_IMAGES_DIR,
                         help="Folder containing the images to run detection on")
    parser.add_argument("--confidence", type=float, default=0.5,
                         help="Confidence threshold; detections below this are not shown")
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"ERROR: model file not found: {args.model_path}")
        print("Use --model-path to point to the actual location of best.pt")
        return

    if not os.path.isdir(args.test_images_dir):
        print(f"ERROR: test images folder not found: {args.test_images_dir}")
        print("Use --test-images-dir to point to the actual location")
        return

    print(f"Loading model from: {args.model_path}")
    model = YOLO(args.model_path)

    print(f"Running detection on all images in: {args.test_images_dir}")
    results = model.predict(
        source=args.test_images_dir,
        conf=args.confidence,
        save=True,        # automatically save the annotated result images
        show_labels=True,
        show_conf=True,
    )

    print(f"\nDone! Tested {len(results)} images.")
    print("Results (annotated images) are saved automatically.")
    print("Check the terminal output above for the exact save location")
    print('(usually something like: runs\\detect\\predict\\)')

    # Print a detection summary for each image
    print("\n--- Detection Summary ---")
    for r in results:
        image_name = os.path.basename(r.path)
        print(f"\n📷 {image_name}")

        if len(r.boxes) == 0:
            print("   No detections")
        else:
            for i, box in enumerate(r.boxes, start=1):
                class_id = int(box.cls)
                class_name = model.names[class_id]
                confidence = float(box.conf) * 100  # convert to a percentage for readability
                print(f"   Fruit {i}: {class_name} — Confidence: {confidence:.1f}%")


if __name__ == "__main__":
    main()
