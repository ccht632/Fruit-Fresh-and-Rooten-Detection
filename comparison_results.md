# SSD vs YOLOv8 — Test Set Comparison Results

Evaluation setup: both models were evaluated on the same test set (`data/fresh and rotten fruits.v3i.yolov8/test/`, 96 images, 155 annotated boxes), so the results are directly comparable.

- SSD: `ssd/evaluate.py` (pycocotools COCOeval), full output in `ssd/eval_results.txt`
- YOLOv8: Ultralytics' built-in `model.val(split='test')`, full output in `yolo/eval_results.txt`

## Overall metrics

| Metric | SSD300-VGG16 | YOLOv8n |
|---|---|---|
| mAP@0.5 | 0.919 | 0.924 |
| mAP@[0.5:0.95] | 0.675 | 0.674 |
| mAP@0.75 | 0.705 | 0.694 |
| Precision (mean) | not reported separately by COCOeval | 0.899 |
| Recall (mean / AR@100) | 0.726 | 0.849 |
| Inference speed (CPU, per image) | not measured yet, can be added using the same method | approximately 67ms per image |

## Per-class AP@0.5

| Class | SSD | YOLOv8n |
|---|---|---|
| Fresh Apple | 0.923 | 0.835 |
| Fresh Banana | 0.827 | 0.822 |
| Fresh Orange | 0.904 | 0.975 |
| Rotten Banana | 0.857 | 0.921 |
| Rotten Orange | 1.000 | 0.995 |
| Rotten Apple | 1.000 | 0.995 |

## Preliminary observations (ready to use directly in the documentation's Results & Discussion)

- The two models' overall mAP@0.5 and mAP@[0.5:0.95] are very close, within one percentage point of each other, indicating comparable performance on this dataset.
- SSD clearly outperforms YOLOv8n on **Fresh Apple / Fresh Banana**, while YOLOv8n clearly outperforms SSD on **Fresh Orange / Rotten Banana**. This pattern could be explained in the documentation by referring to each class's sample size and colour-feature differences.
- YOLOv8n's overall recall (0.849) is higher than SSD's AR (0.726), indicating YOLOv8n misses fewer detections. However, both models are close to a perfect score on Rotten Orange / Rotten Apple, suggesting that "rotten" characteristics (distinct colour and texture changes) are easier to learn than the "fresh" state.
- YOLOv8n already has per-image inference timing available (approximately 67ms per image, CPU). If an "inference speed" comparison is needed in the documentation, SSD would need to be timed under the same conditions (either by adding timing code to `ssd/evaluate.py`, or by running `ssd/inference.py` over the same batch of images and averaging the elapsed time).
