# SSD vs YOLOv8 — Test Set Comparison Results

Evaluation setup: both models were evaluated on the same test set (`data/fresh and rotten fruits.v3i.yolov8/test/`, 96 images, 155 annotated boxes), so the results are directly comparable.

- SSD: `ssd/evaluate.py` (pycocotools COCOeval), full output in `ssd/eval_results.txt`
- YOLOv8: Ultralytics' built-in `model.val(split='test')`, full output in `yolo/eval_results.txt`

## Overall metrics

| Metric | SSD300-VGG16 | YOLOv8n |
|---|---|---|
| mAP@0.5 | 0.920 | 0.924 |
| mAP@[0.5:0.95] | 0.688 | 0.674 |
| mAP@0.75 | 0.768 | 0.694 |
| Precision (mean) | 0.885 | 0.899 |
| Recall (mean) | 0.934 | 0.849 |
| Inference speed (CPU, per image) | approximately 1004ms per image | approximately 67ms per image |

SSD's Precision/Recall are computed at a fixed 0.25 confidence threshold (`ssd/evaluate.py`). YOLOv8n's come from Ultralytics' own validation, which finds each class's best-F1 confidence point rather than using one fixed threshold. The two columns are not computed the same way and should not be compared directly; mAP@0.5, mAP@[0.5:0.95] and mAP@0.75 above are threshold-independent and remain directly comparable.

## Per-class AP@0.5

| Class | SSD | YOLOv8n |
|---|---|---|
| Fresh Apple | 0.873 | 0.835 |
| Fresh Banana | 0.794 | 0.822 |
| Fresh Orange | 0.994 | 0.975 |
| Rotten Banana | 0.860 | 0.921 |
| Rotten Orange | 1.000 | 0.995 |
| Rotten Apple | 1.000 | 0.995 |

## Per-class Precision/Recall (fair comparison, both at a 0.25 confidence threshold)

Both models' confusion matrices default to a 0.25 confidence threshold (Ultralytics' own default for YOLOv8n, and an explicit `--score-thresh 0.25` for SSD), so Precision/Recall computed from these two confusion matrices use the same operating point and can be compared directly, unlike the mean Precision/Recall rows above.

| Class | SSD Precision | SSD Recall | YOLOv8n Precision | YOLOv8n Recall |
|---|---|---|---|---|
| Fresh Apple | 0.800 | 1.000 | 0.650 | 0.813 |
| Fresh Banana | 0.722 | 0.765 | 0.520 | 0.765 |
| Fresh Orange | 1.000 | 1.000 | 0.889 | 0.889 |
| Rotten Banana | 0.788 | 0.839 | 0.773 | 0.935 |
| Rotten Orange | 1.000 | 1.000 | 1.000 | 1.000 |
| Rotten Apple | 1.000 | 1.000 | 1.000 | 1.000 |

SSD is ahead or tied on Precision for five of the six classes; Recall is mixed, with YOLOv8n ahead specifically on Rotten Banana.

## Preliminary observations (ready to use directly in the documentation's Results & Discussion)

- The two models' overall mAP@0.5 is nearly tied (within half a percentage point), but SSD is ahead on the stricter, threshold-independent metrics: mAP@[0.5:0.95] (0.688 vs 0.674) and mAP@0.75 (0.768 vs 0.694).
- SSD clearly outperforms YOLOv8n on **Fresh Apple**, while YOLOv8n clearly outperforms SSD on **Fresh Banana / Rotten Banana**. This pattern could be explained in the documentation by referring to each class's sample size and colour-feature differences.
- Both models are close to a perfect score on Rotten Orange / Rotten Apple, suggesting that "rotten" characteristics (distinct colour and texture changes) are easier to learn than the "fresh" state.
- Inference speed is the clearest differentiator: YOLOv8n runs roughly 15x faster per image on CPU (~67ms vs ~1004ms), making it the more practical choice for real-time or edge deployment even though SSD has a slight edge on strict-IoU accuracy metrics.
