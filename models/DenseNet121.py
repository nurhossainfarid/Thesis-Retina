# models/DenseNet121.py

import torch
import torch.nn as nn
from torchvision import models


class DenseNet121Medical(nn.Module):
    """
    DenseNet121 wrapper for medical image classification (RGB by default).

    - Uses torchvision DenseNet121 backbone (ImageNet weights optional).
    - Extracts penultimate features (before the original classifier).
    - Applies Dropout + new Linear classifier for `num_classes`.
    - Default input: [B, 3, H, W]  (RGB). If you really need grayscale later,
      set in_channels=1 to adapt the first conv accordingly.

    Args:
        num_classes (int): number of target classes.
        pretrained (bool): load ImageNet weights.
        dropout_rate (float): dropout before final FC.
        in_channels (int): expected input channels. Default 3 (RGB).
        freeze_backbone (bool): if True, freeze backbone params.
    """

    def __init__(
        self,
        num_classes: int = 3,
        pretrained: bool = True,
        dropout_rate: float = 0.5,
        in_channels: int = 3,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        # 1) Load backbone
        if pretrained:
            backbone = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        else:
            backbone = models.densenet121(weights=None)

        # 2) Optionally adapt first conv for non-3ch inputs
        #    (For RGB keep default 3ch conv0; for grayscale set in_channels=1)
        if in_channels != 3:
            old_conv = backbone.features.conv0  # Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            new_conv = nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None,
            )
            # If pretrained and in_channels==1, average weights across channel dim.
            if pretrained and old_conv.weight.shape[1] == 3 and in_channels == 1:
                with torch.no_grad():
                    new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
            else:
                nn.init.kaiming_normal_(new_conv.weight, nonlinearity="relu")
                if new_conv.bias is not None:
                    nn.init.zeros_(new_conv.bias)
            backbone.features.conv0 = new_conv

        # 3) Keep only feature extractor; we’ll add our own classifier
        self.features = backbone.features  # nn.Sequential(...)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # DenseNet121 classifier in_features
        num_feats = backbone.classifier.in_features

        # 4) New classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(num_feats, num_classes),
        )

        # 5) Optionally freeze backbone
        if freeze_backbone:
            for p in self.features.parameters():
                p.requires_grad = False

        # Save a reference if you need to inspect
        self.backbone = backbone  # not used in forward, but handy for debugging

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return pooled feature vectors (no classification).
        Output shape: [B, num_feats]
        """
        x = self.features(x)           # [B, C, H', W']
        x = self.relu(x)
        x = self.pool(x)               # [B, C, 1, 1]
        x = torch.flatten(x, 1)        # [B, C]
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: features -> classifier logits.
        Expects x of shape [B, 3, H, W] by default (RGB).
        """
        feats = self.extract_features(x)    # [B, C]
        logits = self.classifier(feats)     # [B, num_classes]
        return logits
