# Static eval results for developer-mode comparison view.

BEST_MODEL = "SSD300-VGG16"

OVERALL_METRICS = {
    "YOLOv8n": {
        "mAP@0.5": 0.924,
        "mAP@[0.5:0.95]": 0.674,
        "mAP@0.75": 0.694,
        "Precision": 0.899,
        "Recall": 0.849,
    },
    "SSD300-VGG16": {
        "mAP@0.5": 0.920,
        "mAP@[0.5:0.95]": 0.688,
        "mAP@0.75": 0.768,
        "Precision": 0.885,
        "Recall": 0.934,
    },
}

PER_CLASS_AP50 = {
    "Fresh Apple": {"YOLOv8n": 0.835, "SSD300-VGG16": 0.873},
    "Fresh Banana": {"YOLOv8n": 0.822, "SSD300-VGG16": 0.794},
    "Fresh Orange": {"YOLOv8n": 0.975, "SSD300-VGG16": 0.994},
    "Rotten Apple": {"YOLOv8n": 0.995, "SSD300-VGG16": 1.000},
    "Rotten Orange": {"YOLOv8n": 0.995, "SSD300-VGG16": 1.000},
    "Rotten Banana": {"YOLOv8n": 0.921, "SSD300-VGG16": 0.860},
}
