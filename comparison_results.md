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
| Precision (mean) | 0.911 | 0.899 |
| Recall (mean) | 0.906 | 0.849 |
| Inference speed (CPU, per image) | approximately 1004ms per image | approximately 67ms per image |

Precision/Recall for both models are now computed the same way (IoU>=0.5 greedy matching at a 0.5 confidence threshold, averaged across classes), so the two columns are directly comparable.

## Per-class AP@0.5

| Class | SSD | YOLOv8n |
|---|---|---|
| Fresh Apple | 0.873 | 0.835 |
| Fresh Banana | 0.794 | 0.822 |
| Fresh Orange | 0.994 | 0.975 |
| Rotten Banana | 0.860 | 0.921 |
| Rotten Orange | 1.000 | 0.995 |
| Rotten Apple | 1.000 | 0.995 |

## Preliminary observations (ready to use directly in the documentation's Results & Discussion)

- The two models' overall mAP@0.5 is nearly tied (within half a percentage point), but SSD is ahead on the stricter metrics: mAP@[0.5:0.95] (0.688 vs 0.674), mAP@0.75 (0.768 vs 0.694), and mean Precision/Recall (0.911/0.906 vs 0.899/0.849).
- SSD clearly outperforms YOLOv8n on **Fresh Apple**, while YOLOv8n clearly outperforms SSD on **Fresh Banana / Rotten Banana**. This pattern could be explained in the documentation by referring to each class's sample size and colour-feature differences.
- Both models are close to a perfect score on Rotten Orange / Rotten Apple, suggesting that "rotten" characteristics (distinct colour and texture changes) are easier to learn than the "fresh" state.
- Inference speed is the clearest differentiator: YOLOv8n runs roughly 15x faster per image on CPU (~67ms vs ~1004ms), making it the more practical choice for real-time or edge deployment even though SSD has a slight edge on strict-IoU accuracy metrics.
