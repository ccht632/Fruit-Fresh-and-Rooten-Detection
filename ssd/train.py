import argparse
import time

import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from dataset import FruitDataset, get_transform, collate_fn, NUM_CLASSES, get_multi_fruit_flags
from model import get_model


# "多水果同框"图片(如apple+banana+orange同框)在train set里占比很低(~2%)，
# 且几乎都来自同一批底图的翻转/旋转复制。diagnose发现这类稀有场景正是
# apple/orange同框互相误检、三果同框banana检测差这两个问题的共同根因之一，
# 用WeightedRandomSampler对它们做适度过采样，让模型每个epoch多看几次。
# 倍数保守设置(5倍)，避免在这批底图数量本就很少(~12张)的样本上过拟合。
MULTI_FRUIT_OVERSAMPLE_WEIGHT = 5.0


def get_dataloaders(data_root, batch_size, subset=None, num_workers=0):
    train_dataset = FruitDataset(data_root, split="train", transforms=get_transform(train=True))
    valid_dataset = FruitDataset(data_root, split="valid", transforms=get_transform(train=False))

    multi_fruit_flags = get_multi_fruit_flags(train_dataset)

    if subset is not None:
        subset_indices = range(min(subset, len(train_dataset)))
        multi_fruit_flags = [multi_fruit_flags[i] for i in subset_indices]
        train_dataset = Subset(train_dataset, subset_indices)
        valid_dataset = Subset(valid_dataset, range(min(subset // 5 + 1, len(valid_dataset))))

    sample_weights = [
        MULTI_FRUIT_OVERSAMPLE_WEIGHT if is_multi_fruit else 1.0
        for is_multi_fruit in multi_fruit_flags
    ]
    train_sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_dataset), replacement=True)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, collate_fn=collate_fn,
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn,
    )
    return train_loader, valid_loader


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    total_loss = 0.0
    num_batches = 0
    start_time = time.time()

    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    elapsed = time.time() - start_time
    print(f"[Epoch {epoch}] train_loss={avg_loss:.4f}  time={elapsed:.1f}s")
    return avg_loss


@torch.no_grad()
def evaluate_loss(model, data_loader, device, epoch):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        total_loss += losses.item()
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    print(f"[Epoch {epoch}] valid_loss={avg_loss:.4f}")
    return avg_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="../data")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--subset", type=int, default=None,
                         help="Use only the first N images for small-scale testing; otherwise, use all data.")
    parser.add_argument("--output", type=str, default="ssd_fruit_model.pth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Use Device: {device}")
    if args.subset:
        print(f"Small scale testing mode: using only the first {args.subset} images")

    train_loader, valid_loader = get_dataloaders(
        args.data_root, args.batch_size, subset=args.subset
    )
    print(f"Training batch size: {len(train_loader)}  Validate batch number: {len(valid_loader)}")

    model = get_model(num_classes=NUM_CLASSES)
    model.to(device)


    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    best_valid_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch)
        valid_loss = evaluate_loss(model, valid_loader, device, epoch)
        lr_scheduler.step()

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), args.output)
            print(f"  Save the new best model (valid_loss={valid_loss:.4f}) -> {args.output}")

    print("Training Completed.")


if __name__ == "__main__":
    main()