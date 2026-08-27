"""
inference.py

用训练好的SSD模型对一张图片做推理：
框出图中的水果，并显示 "Fresh Apple" / "Rotten Banana" 等标签。

用法：
    python inference.py --image ../data/test/some_image.jpg --checkpoint ssd_fruit_model.pth
    python inference.py --image ../data/test/some_image.jpg --checkpoint ssd_fruit_model.pth --threshold 0.3 --output result.jpg
"""

import argparse

import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import functional as F
from torchvision.ops import nms

from dataset import NUM_CLASSES, CATEGORY_ID_TO_NAME
from model import get_model


# Fresh类别用绿色框，Rotten类别用红色框，方便一眼区分
def get_box_color(category_id: int):
    name = CATEGORY_ID_TO_NAME[category_id]
    return "lime" if name.startswith("Fresh") else "red"


def load_model(checkpoint_path, device, nms_thresh=None):
    model = get_model(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    if nms_thresh is not None:
        # nms_thresh 是模型的普通属性（默认0.45），推理时可以直接覆盖，
        # 不需要重新训练。数值越小，NMS去重越激进（重叠框更容易被合并成一个）。
        model.nms_thresh = nms_thresh
        print(f"已将 nms_thresh 覆盖为: {nms_thresh} (默认值0.45)")

    return model


@torch.no_grad()
def predict_image(model, image, device, score_threshold=0.5, class_agnostic_nms_thresh=0.8):
    """
    核心推理逻辑：接收一个已经打开的PIL Image（而不是文件路径），
    这样摄像头视频流的每一帧也能直接调用这个函数，不需要先存成文件再读。

    class_agnostic_nms_thresh:
        torchvision SSD内部的NMS是"分类别"做的，如果同一个物理位置
        (比如一颗腐烂橙子)被同时判断成"Rotten Orange"和"Rotten Banana"
        且都超过置信度阈值，两个框不会互相去重，会同时显示。
        这里额外做一次"跨类别"的NMS，把物理位置高度重叠的框，
        只保留置信度最高的那一个，其余（不管是什么类别标签）都丢弃。
        设为 None 可以关闭这个后处理，恢复模型原始输出。

        默认值从0.5调高到0.8：诊断发现多水果紧挨/部分遮挡同框时
        (如banana贴着apple/orange)，不同水果的框IoU也可能达到0.5，
        0.5的阈值会把置信度较低的那个水果框（比如banana）跨类别抹掉，
        造成漏检。调到0.8后只去重真正高度重叠的"同一个物体被判成
        两个类别"的重复框，不再误伤紧挨在一起的不同水果。
    """
    image_tensor = F.to_tensor(image).to(device)
    output = model([image_tensor])[0]

    boxes = output["boxes"].cpu()
    labels = output["labels"].cpu()
    scores = output["scores"].cpu()

    keep = scores >= score_threshold
    boxes = boxes[keep]
    labels = labels[keep]
    scores = scores[keep]

    if class_agnostic_nms_thresh is not None and len(boxes) > 0:
        # nms()本身不看类别标签，只看boxes+scores，天然就是"跨类别"去重
        keep_idx = nms(boxes, scores, iou_threshold=class_agnostic_nms_thresh)
        boxes = boxes[keep_idx]
        labels = labels[keep_idx]
        scores = scores[keep_idx]

    return boxes, labels, scores


def predict(model, image_path, device, score_threshold=0.5, class_agnostic_nms_thresh=0.8):
    """
    对单张图片文件做推理，读取文件后调用 predict_image()
    """
    image = Image.open(image_path).convert("RGB")
    boxes, labels, scores = predict_image(
        model, image, device, score_threshold, class_agnostic_nms_thresh
    )
    return image, boxes, labels, scores


def draw_predictions(image, boxes, labels, scores):
    """
    在图片上画出预测框 + 类别标签 + 置信度
    """
    draw_image = image.copy()
    draw = ImageDraw.Draw(draw_image)

    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        # 找不到指定字体时退回默认字体（不同系统字体路径不一样，这样更稳妥）
        font = ImageFont.load_default()

    for box, label, score in zip(boxes, labels, scores):
        xmin, ymin, xmax, ymax = box.tolist()
        category_id = int(label.item())
        name = CATEGORY_ID_TO_NAME.get(category_id, f"id_{category_id}")
        color = get_box_color(category_id)

        draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=3)

        text = f"{name} {score:.2f}"
        text_bbox = draw.textbbox((xmin, ymin), text, font=font)
        draw.rectangle(text_bbox, fill=color)
        draw.text((xmin, ymin), text, fill="black", font=font)

    return draw_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="输入图片路径")
    parser.add_argument("--checkpoint", type=str, default="ssd_fruit_model.pth")
    parser.add_argument("--threshold", type=float, default=0.5, help="置信度阈值，低于此值的预测框会被过滤掉")
    parser.add_argument("--nms-thresh", type=float, default=None,
                         help="NMS去重阈值，默认使用模型自带的0.45。数值越小，重叠框越容易被合并成一个，"
                              "可以用来改善多个同类水果紧贴在一起时框重叠的问题")
    parser.add_argument("--agnostic-nms-thresh", type=float, default=0.8,
                         help="跨类别NMS阈值，解决同一物理位置被不同类别(如Rotten Orange和Rotten Banana)"
                              "同时检测的问题。数值越小去重越激进，但也更容易误伤紧挨在一起的不同水果"
                              "(比如banana贴着apple/orange)。设为负数(如-1)可关闭此后处理")
    parser.add_argument("--output", type=str, default="prediction_result.jpg")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agnostic_thresh = args.agnostic_nms_thresh if args.agnostic_nms_thresh >= 0 else None

    model = load_model(args.checkpoint, device, nms_thresh=args.nms_thresh)
    image, boxes, labels, scores = predict(
        model, args.image, device,
        score_threshold=args.threshold,
        class_agnostic_nms_thresh=agnostic_thresh,
    )

    print(f"检测到 {len(boxes)} 个水果 (阈值={args.threshold}):")
    for label, score in zip(labels, scores):
        name = CATEGORY_ID_TO_NAME.get(int(label.item()), f"id_{label.item()}")
        print(f"  {name}  置信度={score:.3f}")

    result_image = draw_predictions(image, boxes, labels, scores)
    result_image.save(args.output)
    print(f"结果已保存到: {args.output}")


if __name__ == "__main__":
    main()