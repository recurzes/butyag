import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B3_Weights


class ButyagModel(nn.Module):
    def __init__(self, num_classes: int = 1, freeze_backbone: bool = True):
        super().__init__()

        base = models.efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)

        self.backbone = base.features
        self.pool = base.avgpool

        in_features = 1536

        if freeze_backbone:
            self._freeze_backbone()

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(512, num_classes)
        )

        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("Backbone frozen - training classifier head only")

    def unfreeze_backbone(self, from_block: int = 6):
        for i, block in enumerate(self.backbone):
            if i >= from_block:
                for param in block.parameters():
                    param.requires_grad = True

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Unfrozen from block {from_block}. Trainable params: {trainable:,}")

    def count_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total params:       {total:>12,}")
        print(f"Trainable params:   {trainable:>12,}")
        print(f"Frozen params:      {total - trainable:>12,}")


# Factory
def build_model(device: str = "cpu", freeze_backbone: bool = True) -> ButyagModel:
    model = ButyagModel(num_classes=1, freeze_backbone=freeze_backbone)
    model = model.to(device)
    model.count_params()
    return model


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")
    model = build_model(device=device)

    dummy = torch.randn(2, 3, 300, 300).to(device)
    out = model(dummy)
    print(f"\nOutput shape: {out.shape}")