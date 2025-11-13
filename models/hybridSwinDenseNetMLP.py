import torch
import torch.nn as nn
from torchvision import models
from timm import create_model

class HybridSwinDenseNetMLP(nn.Module):
    """
    Swin-T (RGB) + DenseNet121 (1-ch) fused classifier for grayscale CT slices.
    - Replicates grayscale to RGB for Swin.
    - Converts DenseNet first conv to 1-channel by averaging pretrained weights.
    """
    def __init__(
        self,
        num_classes: int = 4,
        pretrained: bool = True,
        swin_name: str = "swin_tiny_patch4_window7_224",
        dropout: float = 0.3,
        freeze_backbones: bool = False,
    ):
        super().__init__()

        # Swin backbone (pooled features)
        self.swin = create_model(swin_name, pretrained=pretrained, num_classes=0)
        swin_dim = getattr(self.swin, "num_features", None) or getattr(self.swin, "embed_dim")

        # DenseNet121 backbone (1-channel, features only)
        self.densenet = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None)
        # Convert first conv to 1-ch
        with torch.no_grad():
            w = self.densenet.features.conv0.weight  # [64, 3, 7, 7]
            self.densenet.features.conv0 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.densenet.features.conv0.weight.copy_(w.mean(dim=1, keepdim=True))
        dn_dim = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Identity()  # features only

        # Fusion head
        self.classifier = nn.Sequential(
            nn.Linear(swin_dim + dn_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

        if freeze_backbones:
            for p in self.swin.parameters(): p.requires_grad = False
            for p in self.densenet.parameters(): p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,1,H,W] or [B,3,H,W]
        x_rgb = x if x.size(1) == 3 else x.repeat(1, 3, 1, 1)
        swin_feat = self.swin(x_rgb)  # [B, swin_dim]

        x_gray = x if x.size(1) == 1 else x.mean(1, keepdim=True)
        dn_feat = self.densenet(x_gray)  # [B, dn_dim]

        fused = torch.cat([swin_feat, dn_feat], dim=1)
        return self.classifier(fused)

    @torch.no_grad()
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x_rgb = x if x.size(1) == 3 else x.repeat(1, 3, 1, 1)
        swin_feat = self.swin(x_rgb)
        x_gray = x if x.size(1) == 1 else x.mean(1, keepdim=True)
        dn_feat = self.densenet(x_gray)
        return torch.cat([swin_feat, dn_feat], dim=1)
