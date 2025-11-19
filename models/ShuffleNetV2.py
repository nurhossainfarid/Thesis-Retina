# shufflenetv2.py
import torch
import torch.nn as nn
from torchvision import models


class ShuffleNetV2(nn.Module):
    """
    ShuffleNetV2-x1.0 with flexible input channels (1 or 3).
    Very CPU-friendly, good for small medical image datasets.
    """

    def __init__(
        self,
        num_classes: int = 3,
        in_ch: int = 3,
        pretrained: bool = True,
        dropout_rate: float = 0.5,
    ):
        super().__init__()

        # Handle old/new torchvision APIs
        if pretrained:
            try:
                # Newer torchvision (weights API)
                from torchvision.models import ShuffleNet_V2_X1_0_Weights
                self.backbone = models.shufflenet_v2_x1_0(
                    weights=ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1
                )
            except Exception:
                # Older torchvision (pretrained flag)
                self.backbone = models.shufflenet_v2_x1_0(pretrained=True)
        else:
            try:
                self.backbone = models.shufflenet_v2_x1_0(weights=None)
            except Exception:
                self.backbone = models.shufflenet_v2_x1_0(pretrained=False)

        # ---- Adapt first conv for in_ch != 3 ----
        # In torchvision, first conv is backbone.conv1[0]
        if in_ch != 3:
            old = self.backbone.conv1[0]  # Conv2d(3, 24, 3, 2, 1, bias=False)
            new_conv = nn.Conv2d(
                in_ch,
                old.out_channels,
                kernel_size=old.kernel_size,
                stride=old.stride,
                padding=old.padding,
                bias=False,
            )

            if pretrained:
                with torch.no_grad():
                    if in_ch == 1:
                        # Average RGB weights to get single-channel filter
                        new_conv.weight.copy_(old.weight.mean(dim=1, keepdim=True))
                    else:
                        # Repeat averaged channel for arbitrary in_ch
                        rep = old.weight.mean(dim=1, keepdim=True).repeat(1, in_ch, 1, 1)
                        rep /= in_ch
                        new_conv.weight.copy_(rep)

            self.backbone.conv1[0] = new_conv

        # ---- Replace classifier head ----
        num_features = self.backbone.fc.in_features  # usually 1024 for x1.0
        self.backbone.fc = nn.Identity()  # remove original fc

        self.classifier = nn.Sequential(
            nn.BatchNorm1d(num_features),
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate * 0.25),
            nn.Linear(256, num_classes),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)  # (B, num_features)
        out = self.classifier(features)
        return out

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return penultimate features (before final linear)."""
        return self.backbone(x)
