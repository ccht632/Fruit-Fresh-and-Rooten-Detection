"""
train_yolo_fruit.py

训练 YOLOv8 模型，检测水果新鲜/腐烂状态。
本版本适合本地 VS Code 运行（CPU或本地GPU），已加入多项防止过拟合的措施。

使用方法：
    1. 确认数据集文件夹（含 data.yaml）已经在项目里
    2. 运行这个脚本

运行:
    python train_yolo_fruit.py
"""

import os
from ultralytics import YOLO

# ---------------- 配置区，按你实际路径改 ----------------
DATA_YAML = r"C:\Users\Cht632\Asgmnt RSW\RSWY2S1\AI - Fresh and Rooten Fruit Classification\data\fresh and rotten fruits.v3i.yolov8\data.yaml"  
EPOCHS = 100
IMG_SIZE = 640
BATCH_SIZE = 16
PROJECT_NAME = "fruit_yolo"
RUN_NAME = "fruit_freshness_v1"
# ---------------------------------------------------------


def train():
    model = YOLO("yolov8n.pt")  # 预训练权重作为起点，比从零训练快、也更不容易过拟合

    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        project=PROJECT_NAME,
        name=RUN_NAME,

        # ---- 防止过拟合的设置 ----
        patience=20,           # Early stopping：20轮没进步就自动停止，不让模型继续"死记硬背"
        dropout=0.2,            # Dropout：训练时随机丢弃部分神经元，逼模型学通用规律而不是记细节
        weight_decay=0.0005,    # 权重衰减：给参数加一点"惩罚"，防止某些权重变得过大、过度拟合训练数据

        # ---- 数据增强（进一步降低过拟合风险，让模型见到更多变化） ----
        hsv_h=0.015,             # 随机调整色调
        hsv_s=0.7,               # 随机调整饱和度（水果新鲜/腐烂很大程度靠颜色判断，这个要保留但别调太猛，避免颜色失真影响fresh/rotten这个关键特征）
        hsv_v=0.4,               # 随机调整亮度
        fliplr=0.5,               # 50%概率水平翻转
        flipud=0.0,               # 不做上下翻转（水果一般不会颠倒着放，没必要）
        degrees=15,               # 随机旋转角度（呼应数据集本身已有的±15度增强）
        translate=0.1,             # 随机平移
        scale=0.5,                 # 随机缩放

        save=True,
        save_period=5,             # 每5个epoch保存一次检查点，避免中途出问题全部白费
    )

    print("Training done.")
    best_model_path = os.path.join(PROJECT_NAME, RUN_NAME, "weights", "best.pt")
    print(f"Best model saved at: {best_model_path}")

if __name__ == "__main__":
    train()

if __name__ == "__main__":
    train()