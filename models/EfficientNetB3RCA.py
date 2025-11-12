# models/EfficientNetB3RCA.py

import torch
import torch.nn as nn
from torchvision import models


class CALayer(nn.Module):
    """
    Channel Attention (Squeeze-and-Excitation style) used inside RCAB.
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, mid, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.avg_pool(x)
        w = self.fc(w)
        return x * w


class RCAB(nn.Module):
    """
    Residual Channel Attention Block (lightweight version).
    Conv(3x3) -> ReLU -> Conv(3x3) -> Channel Attention -> Residual Add.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True),
        )
        self.ca = CALayer(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.body(x)
        y = self.ca(y)
        return x + y


class EfficientNetB3RCA(nn.Module):
    """
    EfficientNet-B3 backbone + RCA head for medical eye image classification.

    - Uses torchvision EfficientNet_B3 (ImageNet weights optional).
    - Optionally adapts first conv for non-3ch inputs (e.g., grayscale).
    - Inserts N Residual Channel Attention blocks on the final feature map.
    - Global pooling + Dropout + Linear classifier.

    Args:
        num_classes (int): number of target classes.
        pretrained (bool): load ImageNet weights.
        dropout_rate (float): dropout before final FC.
        in_channels (int): input channels (default 3 for RGB).
        freeze_backbone (bool): if True, freeze EfficientNet parameters.
        rca_blocks (int): number of RCABs to apply on top features.
        rca_reduction (int): reduction ratio for channel attention.
    """

    def __init__(
        self,
        num_classes: int = 3,
        pretrained: bool = True,
        dropout_rate: float = 0.4,
        in_channels: int = 3,
        freeze_backbone: bool = False,
        rca_blocks: int = 2,
        rca_reduction: int = 16,  # kept for API compatibility; CALayer fixed to 16 above
    ):
        super().__init__()

        # 1) Load EfficientNet-B3 backbone
        if pretrained:
            backbone = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
        else:
            backbone = models.efficientnet_b3(weights=None)

        # 2) Adapt first conv if in_channels != 3
        if in_channels != 3:
            old_conv = backbone.features[0][0]  # Conv2d(3, 40, kernel_size=3, stride=2, padding=1, bias=False)
            new_conv = nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None,
            )
            if pretrained and old_conv.weight.shape[1] == 3 and in_channels == 1:
                with torch.no_grad():
                    new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
            else:
                nn.init.kaiming_normal_(new_conv.weight, nonlinearity="relu")
                if new_conv.bias is not None:
                    nn.init.zeros_(new_conv.bias)
            backbone.features[0][0] = new_conv

        # 3) Keep feature extractor
        self.features = backbone.features
        # Final channels of EfficientNet-B3 (last block output)
        # Typically 1536, but we read it from the last conv layer for safety.
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 300, 300)  # B3 default train size ~300
            if next(self.features.parameters()).is_cuda:
                dummy = dummy.cuda()
        # (We won't actually forward the dummy here to avoid CUDA/device issues.)

        # 4) RCA head on top of the final feature map
        # EfficientNet-B3 last feature channels are 1536:
        last_channels = backbone.classifier[1].in_features  # classifier[1] is Linear(in_features, out_features)
        rca = []
        for _ in range(max(1, rca_blocks)):
            rca.append(RCAB(last_channels))
        self.rca = nn.Sequential(*rca)

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool2d(1)

        # 5) Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(last_channels, num_classes),
        )

        # 6) Optional: freeze backbone
        if freeze_backbone:
            for p in self.features.parameters():
                p.requires_grad = False

        # Keep reference to full backbone (handy for debug or fine-grain unfreezing)
        self.backbone = backbone

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return pooled feature vectors after RCA (no classification).
        Output: [B, C]
        """
        x = self.features(x)   # [B, C, H', W']
        x = self.rca(x)        # RCA refinement
        x = self.relu(x)
        x = self.pool(x)       # [B, C, 1, 1]
        x = torch.flatten(x, 1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for logits. Input shape: [B, 3, H, W] (RGB by default).
        """
        feats = self.extract_features(x)
        logits = self.classifier(feats)
        return logits
