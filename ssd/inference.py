# SSD inference: detect fruit and label Fresh/Rotten on an image.

import argparse
import os

import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import functional as F
from torchvision.ops import nms

from dataset import NUM_CLASSES, CATEGORY_ID_TO_NAME
from model import get_model


# Fresh = green, Rotten = red
def get_box_color(category_id: int):
    name = CATEGORY_ID_TO_NAME[category_id]
    return "lime" if name.startswith("Fresh") else "red"


def load_model(checkpoint_path, device, nms_thresh=None):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"SSD checkpoint not found: {checkpoint_path}\n"
            f"Check the path is correct, or use --checkpoint to point to the actual location of the .pth file"
        )

    model = get_model(num_classes=NUM_CLASSES)
    try:
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint {checkpoint_path}: {e}") from e
    model.to(device)
    model.eval()

    if nms_thresh is not None:
        model.nms_thresh = nms_thresh
        print(f"nms_thresh overridden to: {nms_thresh} (default 0.45)")

    return model


@torch.no_grad()
def predict_image(model, image, device, score_threshold=0.5, class_agnostic_nms_thresh=0.8):
    # Runs inference on a PIL Image; class_agnostic_nms_thresh dedupes overlapping boxes across classes.
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
        # Cross-class NMS: dedupe by box overlap only
        keep_idx = nms(boxes, scores, iou_threshold=class_agnostic_nms_thresh)
        boxes = boxes[keep_idx]
        labels = labels[keep_idx]
        scores = scores[keep_idx]

    return boxes, labels, scores


def predict(model, image_path, device, score_threshold=0.5, class_agnostic_nms_thresh=0.8):
    # Runs inference on an image file.
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        raise OSError(f"Could not read image {image_path}: {e}") from e
    boxes, labels, scores = predict_image(
        model, image, device, score_threshold, class_agnostic_nms_thresh
    )
    return image, boxes, labels, scores


def draw_predictions(image, boxes, labels, scores):
    # Draws boxes + class label + confidence on the image.
    draw_image = image.copy()
    draw = ImageDraw.Draw(draw_image)

    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
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
    parser.add_argument("--image", type=str, required=True, help="Path to the input image")
    parser.add_argument("--checkpoint", type=str, default="ssd_fruit_model.pth")
    parser.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold; boxes below this are dropped")
    parser.add_argument("--nms-thresh", type=float, default=None,
                         help="NMS threshold; defaults to the model's built-in 0.45")
    parser.add_argument("--agnostic-nms-thresh", type=float, default=0.8,
                         help="Cross-class NMS threshold; set negative (e.g. -1) to disable")
    parser.add_argument("--output", type=str, default="prediction_result.jpg")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agnostic_thresh = args.agnostic_nms_thresh if args.agnostic_nms_thresh >= 0 else None

    try:
        model = load_model(args.checkpoint, device, nms_thresh=args.nms_thresh)
        image, boxes, labels, scores = predict(
            model, args.image, device,
            score_threshold=args.threshold,
            class_agnostic_nms_thresh=agnostic_thresh,
        )
    except (FileNotFoundError, OSError, RuntimeError) as e:
        print(f"ERROR: {e}")
        return

    print(f"Detected {len(boxes)} fruit (threshold={args.threshold}):")
    for label, score in zip(labels, scores):
        name = CATEGORY_ID_TO_NAME.get(int(label.item()), f"id_{label.item()}")
        print(f"  {name}  confidence={score:.3f}")

    result_image = draw_predictions(image, boxes, labels, scores)
    result_image.save(args.output)
    print(f"Result saved to: {args.output}")


if __name__ == "__main__":
    main()
