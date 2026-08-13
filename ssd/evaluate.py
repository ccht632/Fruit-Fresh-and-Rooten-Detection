"""
evaluate.py

用 pycocotools 在 test set 上评估训练好的SSD模型，
输出标准COCO指标(mAP@[0.5:0.95], mAP@0.5等)，
以及每个类别(Fresh Apple, Rotten Banana...)各自的AP，
方便判断哪些类别学得好、哪些类别(比如样本较少的orange类)表现较差。

用法：
    python evaluate.py --checkpoint ssd_fruit_model.pth
"""

import argparse
import copy

import torch
from pycocotools.cocoeval import COCOeval

from dataset import FruitDataset, get_transform, collate_fn, NUM_CLASSES, CATEGORY_ID_TO_NAME
from model import get_model
from torch.utils.data import DataLoader


@torch.no_grad()
def run_inference(model, data_loader, device):
    """
    对整个test set跑推理，把预测结果转换成COCO格式:
    [{"image_id": ..., "category_id": ..., "bbox": [x,y,w,h], "score": ...}, ...]
    """
    model.eval()
    results = []

    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        outputs = model(images)  # eval模式下返回预测框，而不是loss

        for target, output in zip(targets, outputs):
            image_id = target["image_id"].item()
            boxes = output["boxes"].cpu()
            labels = output["labels"].cpu()
            scores = output["scores"].cpu()

            for box, label, score in zip(boxes, labels, scores):
                xmin, ymin, xmax, ymax = box.tolist()
                # COCO格式要求bbox是 [x, y, width, height]，这里要转回去
                # (我们Dataset类之前是从xywh转成xyxy给模型用，现在评估要转回xywh给pycocotools用)
                w = xmax - xmin
                h = ymax - ymin
                results.append({
                    "image_id": image_id,
                    "category_id": int(label.item()),
                    "bbox": [xmin, ymin, w, h],
                    "score": float(score.item()),
                })

    return results


def print_per_category_ap(coco_gt, coco_dt, cat_ids):
    """
    pycocotools默认的summarize()只给整体mAP，
    这里对每个类别单独跑一次COCOeval，拿到各自的AP，
    方便判断哪个水果类别检测效果差。
    """
    print("\n各类别 AP：")
    print(f"{'类别':<18}{'AP@[.5:.95]':<15}{'AP@0.5'}")

    for cat_id in cat_ids:
        eval_single = COCOeval(coco_gt, coco_dt, iouType="bbox")
        eval_single.params.catIds = [cat_id]
        eval_single.evaluate()
        eval_single.accumulate()

        # 屏蔽summarize()自带的print，只取数值
        import io
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            eval_single.summarize()

        ap_5095 = eval_single.stats[0]  # AP @ IoU=0.50:0.95
        ap_50 = eval_single.stats[1]    # AP @ IoU=0.50
        name = CATEGORY_ID_TO_NAME.get(cat_id, f"id_{cat_id}")
        print(f"{name:<18}{ap_5095:<15.3f}{ap_50:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="../data")
    parser.add_argument("--checkpoint", type=str, default="ssd_fruit_model.pth")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--score-threshold", type=float, default=0.0,
                         help="过滤掉低于此置信度的预测框，评估阶段建议保持0（让pycocotools自己处理阈值曲线）")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # ---- 数据 ----
    test_dataset = FruitDataset(args.data_root, split="test", transforms=get_transform(train=False))
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, collate_fn=collate_fn,
    )
    print(f"Test集图片数量: {len(test_dataset)}")

    # ---- 模型 ----
    model = get_model(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)

    # ---- 推理，收集预测结果 ----
    print("正在跑推理...")
    results = run_inference(model, test_loader, device)
    print(f"共产生 {len(results)} 个预测框")

    if len(results) == 0:
        print("⚠️ 没有任何预测框，模型可能有问题，请检查checkpoint是否正确加载")
        return

    # ---- 用pycocotools评估 ----
    coco_gt = test_dataset.coco
    coco_dt = coco_gt.loadRes(results)

    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()

    print("\n===== 整体评估结果 =====")
    coco_eval.summarize()

    # ---- 每个类别单独的AP ----
    cat_ids = list(CATEGORY_ID_TO_NAME.keys())
    print_per_category_ap(coco_gt, coco_dt, cat_ids)


if __name__ == "__main__":
    main()