"""
dataset.py

定义 Fresh/Rotten Fruit Detection 的 PyTorch Dataset。
读取 COCO JSON 格式标注（Roboflow 导出），供 torchvision SSD300 训练使用。

目录结构假设：
Fruit-Fresh-and-Rooten-Detection/
├── data/
│   ├── train/  (images + _annotations.coco.json)
│   ├── valid/
│   └── test/
└── ssd/
    └── dataset.py   <- 当前文件
"""

import os
import random
import torch
from PIL import Image
from pycocotools.coco import COCO
from torchvision.transforms import functional as F


# -----------------------------
# 类别映射（跳过 category_id=0 "else" 占位类别）
# -----------------------------
# 注意：Roboflow导出的原始命名不统一 (apple / rotten_apple / rotten-banana)
# 这里统一整理成人类可读的显示名称，方便后续推理阶段直接使用
CATEGORY_ID_TO_NAME = {
    1: "Fresh Apple",
    2: "Fresh Banana",
    3: "Fresh Orange",
    4: "Rotten Banana",
    5: "Rotten Orange",
    6: "Rotten Apple",
}
# 训练用的 label 数量 = 6个真实类别 + 1个 background(id=0，torchvision SSD保留给背景)
NUM_CLASSES = len(CATEGORY_ID_TO_NAME) + 1  # = 7


class FruitDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, split="train", transforms=None):
        """
        root_dir: 指向 data/ 文件夹的路径 (例如 "../data")
        split: "train" / "valid" / "test"
        transforms: 数据增强/预处理 transform（见下方 get_transform）
        """
        self.root_dir = os.path.join(root_dir, split)
        self.annotation_path = os.path.join(self.root_dir, "_annotations.coco.json")
        self.transforms = transforms

        self.coco = COCO(self.annotation_path)

        # 只保留有标注的图片id，且过滤掉category_id=0（保险起见，即使目前数据里没有这个类别的标注）
        self.image_ids = [
            img_id for img_id in self.coco.imgs.keys()
            if len(self.coco.getAnnIds(imgIds=img_id)) > 0
        ]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]

        # ---- 读取图片 ----
        img_info = self.coco.imgs[image_id]
        img_path = os.path.join(self.root_dir, img_info["file_name"])
        image = Image.open(img_path).convert("RGB")

        # ---- 读取标注 ----
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)

        boxes = []
        labels = []
        for ann in anns:
            category_id = ann["category_id"]
            if category_id == 0:
                # 跳过 "else" 占位类别（保险处理，目前数据集里不会触发）
                continue

            # COCO bbox 格式: [x, y, width, height] -> 转成 [xmin, ymin, xmax, ymax]
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


# -----------------------------
# 自定义 transform：需要同时接收 image 和 target，
# 因为水平翻转图片时，boxes 的 x 坐标也必须跟着镜像，
# 不能像标准 torchvision transforms 那样只处理 image。
# -----------------------------
class ComposeDouble:
    """依次执行一组同时处理 (image, target) 的transform"""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensorDouble:
    """把 PIL Image 转成 Tensor，target 原样返回（boxes坐标不受影响）"""
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


class RandomHorizontalFlipDouble:
    """
    以给定概率对图片做水平翻转，并同步翻转 boxes 的 x 坐标。
    boxes 格式为 [xmin, ymin, xmax, ymax]，图片宽度为 width 时：
    新的 xmin = width - 旧xmax
    新的 xmax = width - 旧xmin
    """
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            width, _ = image.size  # PIL Image: (width, height)
            image = F.hflip(image)

            boxes = target["boxes"]
            if boxes.numel() > 0:
                new_boxes = boxes.clone()
                new_boxes[:, 0] = width - boxes[:, 2]  # new xmin = width - old xmax
                new_boxes[:, 2] = width - boxes[:, 0]  # new xmax = width - old xmin
                target["boxes"] = new_boxes

        return image, target


def get_transform(train: bool):
    """
    baseline数据增强:
    - train: 水平翻转(概率0.5，同步翻转boxes坐标) + ToTensor
    - valid/test: 只做 ToTensor（保持原始状态用于公平评估，不做增强）
    """
    if train:
        transform_list = [
            RandomHorizontalFlipDouble(prob=0.5),
            ToTensorDouble(),
        ]
    else:
        transform_list = [ToTensorDouble()]

    return ComposeDouble(transform_list)


def collate_fn(batch):
    """
    因为每张图片的目标(水果)数量不同，无法直接stack成一个tensor，
    需要自定义collate_fn，返回tuple形式让模型自己处理变长的target列表。
    """
    return tuple(zip(*batch))


if __name__ == "__main__":
    # 简单自测：确认Dataset能正常读取，不报错
    dataset = FruitDataset(root_dir="../data", split="train", transforms=get_transform(train=True))
    print(f"训练集图片数量: {len(dataset)}")

    image, target = dataset[0]
    print(f"图片tensor shape: {image.shape}")
    print(f"boxes: {target['boxes']}")
    print(f"labels: {target['labels']}")
    print(f"labels对应名称: {[CATEGORY_ID_TO_NAME[l.item()] for l in target['labels']]}")