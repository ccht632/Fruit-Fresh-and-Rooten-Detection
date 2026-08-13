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

from dataset import NUM_CLASSES, CATEGORY_ID_TO_NAME
from model import get_model


# Fresh类别用绿色框，Rotten类别用红色框，方便一眼区分
def get_box_color(category_id: int):
    name = CATEGORY_ID_TO_NAME[category_id]
    return "lime" if name.startswith("Fresh") else "red"


def load_model(checkpoint_path, device):
    model = get_model(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict(model, image_path, device, score_threshold=0.5):
    """
    对单张图片做推理，返回过滤掉低置信度框之后的结果
    """
    image = Image.open(image_path).convert("RGB")
    image_tensor = F.to_tensor(image).to(device)

    output = model([image_tensor])[0]

    boxes = output["boxes"].cpu()
    labels = output["labels"].cpu()
    scores = output["scores"].cpu()

    # 过滤掉低置信度的预测框
    keep = scores >= score_threshold
    boxes = boxes[keep]
    labels = labels[keep]
    scores = scores[keep]

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
    parser.add_argument("--output", type=str, default="prediction_result.jpg")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(args.checkpoint, device)
    image, boxes, labels, scores = predict(model, args.image, device, score_threshold=args.threshold)

    print(f"检测到 {len(boxes)} 个水果 (阈值={args.threshold}):")
    for label, score in zip(labels, scores):
        name = CATEGORY_ID_TO_NAME.get(int(label.item()), f"id_{label.item()}")
        print(f"  {name}  置信度={score:.3f}")

    result_image = draw_predictions(image, boxes, labels, scores)
    result_image.save(args.output)
    print(f"结果已保存到: {args.output}")


if __name__ == "__main__":
    main()