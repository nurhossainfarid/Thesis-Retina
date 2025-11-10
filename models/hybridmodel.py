import torch
import torch.nn as nn
from torchvision import models
from timm import create_model

class HybridViTCNNMLP(nn.Module):
    """
    Hybrid Vision Transformer (ViT) + ResNet18 (CNN) + MLP head
    - ViT takes 3-channel RGB (we replicate grayscale to RGB)
    - ResNet18 is adapted to 1-channel inputs (keeps ImageNet weights by mean init)
    """
    def __init__(
        self,
        num_classes: int = 3,
        pretrained: bool = True,
        vit_name: str = "vit_base_patch16_224",
        dropout_rate: float = 0.3,
        freeze_backbones: bool = False,
    ):
        super().__init__()

        # ---- ViT backbone (features only)
        self.vit = create_model(vit_name, pretrained=pretrained, num_classes=0)  # feature extractor
        vit_dim = getattr(self.vit, "num_features", None) or getattr(self.vit, "embed_dim")

        # ---- ResNet18 backbone (1-channel friendly, features only)
        self.cnn = models.resnet18(pretrained=pretrained)
        # convert first conv to 1-channel by averaging pretrained RGB weights
        with torch.no_grad():
            w = self.cnn.conv1.weight  # [64, 3, 7, 7]
            self.cnn.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.cnn.conv1.weight.copy_(w.mean(dim=1, keepdim=True))
        cnn_dim = self.cnn.fc.in_features
        self.cnn.fc = nn.Identity()

        # ---- Fusion + classifier head
        self.classifier = nn.Sequential(
            nn.Linear(vit_dim + cnn_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

        if freeze_backbones:
            for p in self.vit.parameters():
                p.requires_grad = False
            for p in self.cnn.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B,1,H,W] (grayscale) or [B,3,H,W] (already RGB)
        """
        # ViT expects 3-channel
        x_rgb = x if x.size(1) == 3 else x.repeat(1, 3, 1, 1)
        vit_feat = self.vit(x_rgb)            # [B, vit_dim]

        # ResNet path uses true grayscale (1-ch) if available
        cnn_in = x if x.size(1) == 1 else x.mean(1, keepdim=True)
        cnn_feat = self.cnn(cnn_in)           # [B, cnn_dim]

        fused = torch.cat([vit_feat, cnn_feat], dim=1)
        return self.classifier(fused)

    @torch.no_grad()
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x_rgb = x if x.size(1) == 3 else x.repeat(1, 3, 1, 1)
        vit_feat = self.vit(x_rgb)
        cnn_in = x if x.size(1) == 1 else x.mean(1, keepdim=True)
        cnn_feat = self.cnn(cnn_in)
        return torch.cat([vit_feat, cnn_feat], dim=1)
