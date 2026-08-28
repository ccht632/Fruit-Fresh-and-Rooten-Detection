"""Static evaluation results for both models, used by the developer-mode comparison view.

Numbers come from the held-out test set (96 images, 155 annotated fruits):
yolo/eval_results.txt (Ultralytics val) and ssd/eval_results.txt (pycocotools COCOeval).
"""

import math

BEST_MODEL = "YOLOv8n"

OVERALL_METRICS = {
    "YOLOv8n": {
        "mAP@0.5": 0.924,
        "mAP@[0.5:0.95]": 0.674,
        "mAP@0.75": 0.694,
        "Precision": 0.899,
        "Recall": 0.849,
    },
    "SSD300-VGG16": {
        "mAP@0.5": 0.919,
        "mAP@[0.5:0.95]": 0.675,
        "mAP@0.75": 0.705,
        "Precision": math.nan,  # pycocotools does not report a single precision figure
        "Recall": 0.726,  # Average Recall (AR) @ IoU=0.50:0.95, maxDets=100
    },
}

PER_CLASS_AP50 = {
    "Fresh Apple": {"YOLOv8n": 0.835, "SSD300-VGG16": 0.923},
    "Fresh Banana": {"YOLOv8n": 0.822, "SSD300-VGG16": 0.827},
    "Fresh Orange": {"YOLOv8n": 0.975, "SSD300-VGG16": 0.904},
    "Rotten Apple": {"YOLOv8n": 0.995, "SSD300-VGG16": 1.000},
    "Rotten Orange": {"YOLOv8n": 0.995, "SSD300-VGG16": 1.000},
    "Rotten Banana": {"YOLOv8n": 0.921, "SSD300-VGG16": 0.857},
}
