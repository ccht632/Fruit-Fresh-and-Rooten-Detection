import torch
from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights
from torchvision.models.detection.ssd import SSDClassificationHead
from torchvision.models.detection._utils import retrieve_out_channels


def get_model(num_classes: int):
    weights = SSD300_VGG16_Weights.DEFAULT
    model = ssd300_vgg16(weights=weights)

    in_channels = retrieve_out_channels(model.backbone, (300, 300))
    num_anchors = model.anchor_generator.num_anchors_per_location()

    model.head.classification_head = SSDClassificationHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
        num_classes=num_classes,
    )

    model.num_classes = num_classes

    return model


if __name__ == "__main__":
    from dataset import NUM_CLASSES

    model = get_model(num_classes=NUM_CLASSES)
    model.eval()

    print(f"Model built successfully，num_classes = {NUM_CLASSES}")

    dummy_input = [torch.rand(3, 300, 300)]
    with torch.no_grad():
        output = model(dummy_input)

    print("Forward propagation successful, output structure：")
    print(f"  boxes shape: {output[0]['boxes'].shape}")
    print(f"  labels shape: {output[0]['labels'].shape}")
    print(f"  scores shape: {output[0]['scores'].shape}")