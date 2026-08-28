import os
import random
import torch
from PIL import Image
from pycocotools.coco import COCO
from torchvision.transforms import functional as F
from torchvision.transforms import ColorJitter


# category_id -> display name
CATEGORY_ID_TO_NAME = {
    1: "Fresh Apple",
    2: "Fresh Banana",
    3: "Fresh Orange",
    4: "Rotten Banana",
    5: "Rotten Orange",
    6: "Rotten Apple",
}
NUM_CLASSES = len(CATEGORY_ID_TO_NAME) + 1  # +1 background

# category_id -> base fruit (fresh/rotten merged)
CATEGORY_ID_TO_BASE_FRUIT = {
    1: "Apple", 2: "Banana", 3: "Orange",
    4: "Banana", 5: "Orange", 6: "Apple",
}


class FruitDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, split="train", transforms=None):
        self.root_dir = os.path.join(root_dir, split)
        self.annotation_path = os.path.join(self.root_dir, "_annotations.coco.json")
        self.transforms = transforms

        if not os.path.exists(self.annotation_path):
            raise FileNotFoundError(
                f"COCO annotation file not found: {self.annotation_path}\n"
                f"Check --data-root points to the folder containing train/valid/test subfolders."
            )
        self.coco = COCO(self.annotation_path)

        # keep only annotated images
        self.image_ids = [
            img_id for img_id in self.coco.imgs.keys()
            if len(self.coco.getAnnIds(imgIds=img_id)) > 0
        ]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]

        # Load image
        img_info = self.coco.imgs[image_id]
        img_path = os.path.join(self.root_dir, img_info["file_name"])
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image referenced in annotations but not found on disk: {img_path}")
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise OSError(f"Could not read image {img_path}: {e}") from e

        # Load annotations
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)

        boxes = []
        labels = []
        for ann in anns:
            category_id = ann["category_id"]
            if category_id == 0:
                continue

            # xywh -> xyxy
            x, y, w, h = ann["bbox"]
            xmin, ymin = x, y
            xmax, ymax = x + w, y + h
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(category_id)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([image_id]),
        }

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target


def get_multi_fruit_flags(dataset: "FruitDataset"):
    """Flags images containing >=2 different fruit types."""
    flags = []
    for image_id in dataset.image_ids:
        ann_ids = dataset.coco.getAnnIds(imgIds=image_id)
        anns = dataset.coco.loadAnns(ann_ids)
        base_fruits = {
            CATEGORY_ID_TO_BASE_FRUIT[ann["category_id"]]
            for ann in anns if ann["category_id"] != 0
        }
        flags.append(len(base_fruits) >= 2)
    return flags


class ComposeDouble:
    """Runs transforms that need both image and target."""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensorDouble:
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


class RandomHorizontalFlipDouble:
    """Flip image and mirror box x-coords."""
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            width, _ = image.size
            image = F.hflip(image)

            boxes = target["boxes"]
            if boxes.numel() > 0:
                new_boxes = boxes.clone()
                new_boxes[:, 0] = width - boxes[:, 2]
                new_boxes[:, 2] = width - boxes[:, 0]
                target["boxes"] = new_boxes

        return image, target


class RandomScaleCropDouble:
    """Random crop; breaks box-area/freshness shortcut."""
    def __init__(self, prob=0.5, scale_range=(0.7, 1.0), min_box_keep_ratio=0.3):
        self.prob = prob
        self.scale_range = scale_range
        self.min_box_keep_ratio = min_box_keep_ratio

    def __call__(self, image, target):
        if random.random() >= self.prob:
            return image, target

        width, height = image.size
        scale = random.uniform(*self.scale_range)
        crop_w = max(1, int(width * scale))
        crop_h = max(1, int(height * scale))

        crop_x = random.randint(0, width - crop_w)
        crop_y = random.randint(0, height - crop_h)

        boxes = target["boxes"]
        labels = target["labels"]

        if boxes.numel() == 0:
            return image, target

        new_boxes = boxes.clone()
        new_boxes[:, [0, 2]] = new_boxes[:, [0, 2]].clamp(crop_x, crop_x + crop_w) - crop_x
        new_boxes[:, [1, 3]] = new_boxes[:, [1, 3]].clamp(crop_y, crop_y + crop_h) - crop_y

        orig_areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        new_areas = (new_boxes[:, 2] - new_boxes[:, 0]) * (new_boxes[:, 3] - new_boxes[:, 1])
        # drop boxes cropped down too much
        keep = (new_areas / orig_areas.clamp(min=1e-6)) >= self.min_box_keep_ratio

        if keep.sum().item() == 0:
            return image, target

        image = F.crop(image, crop_y, crop_x, crop_h, crop_w)
        target = dict(target)
        target["boxes"] = new_boxes[keep]
        target["labels"] = labels[keep]
        return image, target


class ColorJitterDouble:
    """Color-only augmentation; boxes unaffected."""
    def __init__(self, brightness=0.15, contrast=0.15, saturation=0.15, hue=0.02):
        self.jitter = ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)

    def __call__(self, image, target):
        image = self.jitter(image)
        return image, target


def get_transform(train: bool):
    """Train: flip + crop + color jitter. Eval: tensor only."""
    if train:
        transform_list = [
            RandomHorizontalFlipDouble(prob=0.5),
            RandomScaleCropDouble(prob=0.5, scale_range=(0.7, 1.0)),
            ColorJitterDouble(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.02),
            ToTensorDouble(),
        ]
    else:
        transform_list = [ToTensorDouble()]

    return ComposeDouble(transform_list)


def collate_fn(batch):
    """Variable-length targets; no stacking."""
    return tuple(zip(*batch))


if __name__ == "__main__":
    dataset = FruitDataset(root_dir="../data", split="train", transforms=get_transform(train=True))
    print(f"Train set size: {len(dataset)}")

    image, target = dataset[0]
    print(f"Image tensor shape: {image.shape}")
    print(f"Boxes: {target['boxes']}")
    print(f"Labels: {target['labels']}")
    print(f"Label names: {[CATEGORY_ID_TO_NAME[l.item()] for l in target['labels']]}")
