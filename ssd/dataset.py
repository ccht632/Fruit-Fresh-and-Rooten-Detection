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
from torchvision.transforms import ColorJitter


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

# category_id -> 水果种类（不分fresh/rotten），用于识别"多水果同框"的图片
CATEGORY_ID_TO_BASE_FRUIT = {
    1: "Apple", 2: "Banana", 3: "Orange",
    4: "Banana", 5: "Orange", 6: "Apple",
}


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


def get_multi_fruit_flags(dataset: "FruitDataset"):
    """
    返回一个长度为 len(dataset) 的 bool 列表，标记每张图片是否同时包含
    >=2 种不同水果（不区分fresh/rotten，比如同时有apple和orange就算）。

    诊断发现这类"多水果同框"图片在train set里占比很低(~2%)，且几乎都来自
    同一批底图的翻转/旋转复制，训练时对它们的曝光次数太少，是apple/orange
    同框互相误检、banana在三果同框场景下检测差这两个问题的共同根因之一。
    train.py 用这个函数的结果构造 WeightedRandomSampler，对这批图片做过采样。
    """
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


class RandomScaleCropDouble:
    """
    以给定概率随机裁剪图片的一个子区域(保留原图 scale_range 比例的宽高)，
    并同步裁剪/过滤 boxes 坐标。

    诊断发现：训练集里 rotten apple/orange 的框普遍占满画面(~70-77%面积)，
    而fresh apple/orange的框普遍较小(~27-28%面积)——这是采集时留下的偏差，
    模型很容易学到"画面里橙色/苹果色区域占比大 => rotten"这种shortcut，
    而不是真正学斑点、褶皱等新鲜度纹理特征。这个transform随机改变同一个
    物体在画面里的占比，用来打破"占比"和"新鲜度标签"之间的伪相关。

    scale_range=(0.7, 1.0) 是保守设置：每次裁剪保留原图70%~100%的宽高，
    幅度不大，避免破坏图片内容或过度拖慢收敛。
    """
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

        # 把boxes坐标裁剪到新窗口范围内，再平移到新窗口的局部坐标系
        new_boxes = boxes.clone()
        new_boxes[:, [0, 2]] = new_boxes[:, [0, 2]].clamp(crop_x, crop_x + crop_w) - crop_x
        new_boxes[:, [1, 3]] = new_boxes[:, [1, 3]].clamp(crop_y, crop_y + crop_h) - crop_y

        orig_areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        new_areas = (new_boxes[:, 2] - new_boxes[:, 0]) * (new_boxes[:, 3] - new_boxes[:, 1])
        # 裁剪后残留面积占原面积太少的box(比如被裁掉了大半)直接丢弃，
        # 避免训练时看到一个"新鲜度关键纹理都被裁没了"的残缺框
        keep = (new_areas / orig_areas.clamp(min=1e-6)) >= self.min_box_keep_ratio

        if keep.sum().item() == 0:
            # 这次裁剪会让所有box都不满足保留条件，放弃裁剪，返回原图
            return image, target

        image = F.crop(image, crop_y, crop_x, crop_h, crop_w)
        target = dict(target)
        target["boxes"] = new_boxes[keep]
        target["labels"] = labels[keep]
        return image, target


class ColorJitterDouble:
    """
    只对图片做颜色扰动(brightness/contrast/saturation/hue)，boxes不受影响。

    诊断发现训练pipeline里完全没有颜色相关的增强，这会让模型倾向于依赖
    局部颜色小线索去区分fresh apple / fresh orange，泛化性差；同时也让
    "画面颜色区域占比"这种shortcut(见RandomScaleCropDouble的说明)更容易
    被学到。hue幅度刻意设得很小(默认0.02)，是为了不抹掉apple(红/绿)和
    orange(橙)之间本身就该用的合法颜色区分线索，只是让模型对光照/色调的
    小幅扰动更鲁棒。
    """
    def __init__(self, brightness=0.15, contrast=0.15, saturation=0.15, hue=0.02):
        self.jitter = ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)

    def __call__(self, image, target):
        image = self.jitter(image)
        return image, target


def get_transform(train: bool):
    """
    train数据增强:
    - 水平翻转(概率0.5，同步翻转boxes坐标)
    - 随机scale/crop(概率0.5，同步裁剪boxes坐标，见RandomScaleCropDouble说明)
    - 颜色扰动(brightness/contrast/saturation/hue，仅作用于图片，见ColorJitterDouble说明)
    - ToTensor
    valid/test: 只做 ToTensor（保持原始状态用于公平评估，不做增强）
    """
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