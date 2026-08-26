# Fresh and Rotten Fruit Detection

Object detection and classification project: implemented with two algorithms, **SSD300-VGG16** and **YOLOv8n**, to detect fruit (Apple / Banana / Orange) in images or a live camera feed and determine whether it is Fresh or Rotten, then compare the two algorithms' performance.

## Project structure

```
ssd/            SSD300-VGG16 (torchvision) training/evaluation/inference
yolo/           YOLOv8n (ultralytics) training/evaluation/inference
camera/         Real-time camera detection demo (both models can be switched between)
data/           Dataset (.gitignore, needs to be downloaded separately, see "Dataset" below)
runs/           YOLO training outputs (weights, training curves, etc.)
documentation/  Documentation draft, figures, and other assets for the written report
comparison_results.md   Comparison of the two algorithms on the same test set
```

## Environment setup

```
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

> `torch`/`torchvision` install the CPU version by default. If you have an NVIDIA GPU and want to train with it, go to https://pytorch.org/get-started/locally/ and reinstall the matching torch/torchvision for your CUDA version.

## Dataset

Uses a public dataset from Roboflow (exported in YOLOv8 format):
https://universe.roboflow.com/alrz-nt-wcyue/fresh-and-rotten-fruits-aa5ae/dataset/3

After downloading, place it at:
```
data/fresh and rotten fruits.v3i.yolov8/
├── data.yaml
├── train/images, train/labels
├── valid/images, valid/labels
└── test/images, test/labels
```

6 classes: `Fresh Apple / Fresh Banana / Fresh Orange / Rotten Banana / Rotten Orange / Rotten Apple`

**Note**: this Roboflow export only contains YOLO-format annotations (`labels/*.txt`), not the COCO json required by the SSD side. Before running the SSD scripts, run the following once:

```
cd ssd
python build_coco_annotations.py --data-root "../data/fresh and rotten fruits.v3i.yolov8"
```

This regenerates each split's `_annotations.coco.json` from the YOLO labels (only needs to be run once, unless the dataset is updated to a new version).

## SSD (`ssd/`)

```
cd ssd

# 1. Generate COCO annotations (before the first run, see "Dataset" above)
python build_coco_annotations.py --data-root "../data/fresh and rotten fruits.v3i.yolov8"

# 2. Train
python train.py --data-root "../data/fresh and rotten fruits.v3i.yolov8" --epochs 10

# 3. Evaluate on the test set (outputs mAP / per-class AP)
python evaluate.py --data-root "../data/fresh and rotten fruits.v3i.yolov8" --checkpoint ssd_fruit_model.pth

# 4. Run inference on a single image
python inference.py --image "../data/fresh and rotten fruits.v3i.yolov8/test/images/xxx.jpg" --checkpoint ssd_fruit_model.pth
```

Pretrained weights: `ssd/ssd_fruit_model.pth` (already included in the repo, so you can skip training and go straight to evaluation/inference).

## YOLO (`yolo/`)

```
cd yolo

# Train (default path is computed automatically from this file's location, so it usually doesn't
# need changing between computers; --data-yaml is only needed if the dataset is not in the default location)
python train_yolo_fruit.py

# Batch inference on the whole test folder, result images saved to runs/detect/predict/
python test_model.py
python test_model.py --confidence 0.3       # lower the threshold on the spot to check for missed detections

# Sample and check whether the annotation boxes are correct
python visualize_dataset.py --split test --num-samples 10
```

Evaluate on the test set (using Ultralytics' built-in `val`):

```python
from ultralytics import YOLO
model = YOLO("runs/detect/fruit_yolo/fruit_freshness_v1/weights/best.pt")
model.val(data="data/fresh and rotten fruits.v3i.yolov8/data.yaml", split="test")
```

Pretrained weights: `runs/detect/fruit_yolo/fruit_freshness_v1/weights/best.pt`.

All path arguments in the scripts under `yolo/` (`--data-yaml`, `--model-path`, etc.) default to relative paths computed from each script's own location, so under normal circumstances no code changes are needed between computers or usernames. Manually overriding them is only needed if the dataset or model weights are stored in a different location from the repository structure.

## Real-time camera demo (`camera/`)

```
cd camera
python camera_app.py --model ssd     # or --model yolo

# Additional arguments supported on the YOLO side (passed through to camera_app_yolo.py):
python camera_app.py --model yolo --threshold 0.3    # lower the threshold on the spot to check for missed detections
python camera_app.py --model yolo --no-blur          # disable background blur (a display-only effect, does not affect detection results)
```

`camera_app.py` is the unified entry point, dispatching to `camera_app_ssd.py` / `camera_app_yolo.py` based on the `--model` argument. Their dependencies (torchvision vs ultralytics) are kept independent, so both environments don't need to be installed at once to run either one.

## Results comparison

Evaluation results, per-class AP, and a brief analysis for both models on the same 96-image test set are in [`comparison_results.md`](comparison_results.md). Raw evaluation logs: `ssd/eval_results.txt`, `yolo/eval_results.txt`.

| Metric | SSD300-VGG16 | YOLOv8n |
|---|---|---|
| mAP@0.5 | 0.919 | 0.924 |
| mAP@[0.5:0.95] | 0.675 | 0.674 |

## Troubleshooting

- **`evaluate.py` / `inference.py` hangs downloading pretrained weights, or reports an SSL certificate error**: these two scripts load a trained checkpoint, so there is no need to download torchvision's official pretrained weights. This is already skipped in `model.py` via `pretrained=False`. If the issue persists, check whether the place calling `get_model()` is passing `pretrained=False`.
- **`ModuleNotFoundError: No module named 'pycocotools'` / `'ultralytics'` / `'yaml'`**: make sure `pip install -r requirements.txt` was run inside the `venv`.
