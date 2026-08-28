# Fresh and Rotten Fruit Detection

Object detection and classification project: implemented with two algorithms, **SSD300-VGG16** and **YOLOv8n**, to detect fruit (Apple / Banana / Orange) in images or a live camera feed and determine whether it is Fresh or Rotten, then compare the two algorithms' performance.

## Project structure

```
ssd/            SSD300-VGG16 (torchvision) training/evaluation/inference
yolo/           YOLOv8n (ultralytics) training/evaluation/inference
camera/         Streamlit web app: upload a photo or take one with the device camera (both models can be switched between)
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

Uses a public dataset from Roboflow, exported twice in two different formats (one per model, since YOLOv8 needs YOLO-format `.txt` labels and SSD needs a COCO-format `_annotations.coco.json`):
https://universe.roboflow.com/alrz-nt-wcyue/fresh-and-rotten-fruits-aa5ae/dataset/3

```
data/
├── Yolov8n/fresh and rotten fruits.v3i.yolov8/    (used by yolo/, camera/'s YOLO path)
│   ├── data.yaml
│   ├── train/images, train/labels
│   ├── valid/images, valid/labels
│   └── test/images, test/labels
└── SSD/fresh and rotten fruits.v3i.coco/          (used by ssd/, camera/'s SSD path)
    ├── train/ (images + _annotations.coco.json)
    ├── valid/
    └── test/
```

6 classes: `Fresh Apple / Fresh Banana / Fresh Orange / Rotten Banana / Rotten Orange / Rotten Apple`

## SSD (`ssd/`)

```
cd ssd

# 1. Train
python train.py --data-root "../data/SSD/fresh and rotten fruits.v3i.coco" --epochs 10

# 2. Evaluate on the test set (outputs mAP / per-class AP to the console and eval_results.txt if redirected)
python evaluate.py --data-root "../data/SSD/fresh and rotten fruits.v3i.coco" --checkpoint ssd_fruit_model.pth

# 3. Generate the confusion matrix (saved as confusion_matrix.png)
python confusion_matrix.py --data-root "../data/SSD/fresh and rotten fruits.v3i.coco" --checkpoint ssd_fruit_model.pth

# 4. Run inference on a single image
python inference.py --image "../data/SSD/fresh and rotten fruits.v3i.coco/test/xxx.jpg" --checkpoint ssd_fruit_model.pth
```

> The console scripts print Chinese status messages; if redirecting output to a file on Windows and the text comes out garbled, run with `PYTHONIOENCODING=utf-8` set first (e.g. `set PYTHONIOENCODING=utf-8` on cmd, or prefix the command on Git Bash/PowerShell).

Pretrained weights: `ssd/ssd_fruit_model.pth` (already included in the repo, so you can skip training and go straight to evaluation/inference).

## YOLO (`yolo/`)

```
cd yolo

# Train (default path is computed automatically from this file's location, so it usually doesn't
# need changing between computers; --data-yaml is only needed if the dataset is not in the default location)
python train_yolo_fruit.py

# Evaluate on the test set (outputs mAP / per-class AP, saved to eval_results.txt)
python evaluate.py

# Generate the confusion matrix (saved as confusion_matrix.png + confusion_matrix.txt)
python confusion_matrix.py

# Batch inference on the whole test folder, result images saved to runs/detect/predict/
python test_model.py
python test_model.py --confidence 0.3       # lower the threshold on the spot to check for missed detections

# Sample and check whether the annotation boxes are correct
python visualize_dataset.py --split test --num-samples 10
```

Pretrained weights: `runs/detect/fruit_yolo/fruit_freshness_v1/weights/best.pt`.

All path arguments in the scripts under `yolo/` (`--data-yaml`, `--model-path`, etc.) default to relative paths computed from each script's own location, so under normal circumstances no code changes are needed between computers or usernames. Manually overriding them is only needed if the dataset or model weights are stored in a different location from the repository structure.

## Web UI (`camera/streamlit_app.py`)

A Streamlit page: upload a photo, or take one with your device's camera (this satisfies the "real-time camera" requirement without needing a separate continuous-video pipeline), pick SSD300-VGG16 or YOLOv8n, and view the detection result with bounding boxes and per-detection confidence scores.

```
cd camera
streamlit run streamlit_app.py
```

Needs both `ssd/ssd_fruit_model.pth` and the trained YOLO weights (`runs/detect/fruit_yolo/fruit_freshness_v1/weights/best.pt`) present, since either model can be picked from the page.

## Results comparison

Evaluation results, per-class AP, and a brief analysis for both models on the same test set (155 annotated fruits; SSD's COCO-format export has 93 images, YOLO's YOLO-format export has 96 images — same underlying dataset, same GT instance counts per class) are in [`comparison_results.md`](comparison_results.md). Raw evaluation logs: `ssd/eval_results.txt`, `yolo/eval_results.txt`. Confusion matrices: `ssd/confusion_matrix.png`, `yolo/confusion_matrix.png` (+ `yolo/confusion_matrix.txt` for the raw numbers).

| Metric | SSD300-VGG16 | YOLOv8n |
|---|---|---|
| mAP@0.5 | 0.920 | 0.924 |
| mAP@[0.5:0.95] | 0.688 | 0.674 |
| mAP@0.75 | 0.768 | 0.694 |
| Precision (mean) | 0.911 | 0.899 |
| Recall (mean) | 0.906 | 0.849 |

## Troubleshooting

- **`get_model()` (in `ssd/model.py`) fails or hangs on first run**: it downloads torchvision's ImageNet-pretrained SSD300-VGG16 backbone weights the first time it's called, so it needs internet access. This only affects the backbone init; `evaluate.py`/`inference.py` immediately overwrite it with your trained checkpoint, so it has no effect on results — it just needs to succeed once (subsequent runs use torch's local cache).
- **`ModuleNotFoundError: No module named 'pycocotools'` / `'ultralytics'` / `'yaml'`**: make sure `pip install -r requirements.txt` was run inside the `venv`.
- **Garbled/mojibake console output when redirecting a `ssd/` script's output to a file on Windows**: those scripts print Chinese status text; set `PYTHONIOENCODING=utf-8` before running (see the SSD section above).
