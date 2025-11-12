# densenet121.py
import torch
import torch.nn as nn
from torchvision import models

class DenseNet121Medical(nn.Module):
    """
    DenseNet121 for medical image classification with flexible input channels (1 or 3).
    If in_ch==3 and pretrained=True, we keep the default conv0.
    If in_ch!=3, we replace conv0 and (if pretrained) initialize by channel-averaging.
    """
    def __init__(self, num_classes: int = 3, in_ch: int = 3, pretrained: bool = True, dropout_rate: float = 0.5):
        super().__init__()
        if pretrained:
            self.densenet = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        else:
            self.densenet = models.densenet121(weights=None)

        # adapt first conv if needed
        if in_ch != 3:
            with torch.no_grad():
                w = self.densenet.features.conv0.weight  # [64,3,7,7]
            self.densenet.features.conv0 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
            if pretrained:
                with torch.no_grad():
                    if in_ch == 1:
                        self.densenet.features.conv0.weight.copy_(w.mean(dim=1, keepdim=True))
                    else:
                        # repeat or interpolate channels
                        rep = torch.mean(w, dim=1, keepdim=True).repeat(1, in_ch, 1, 1) / in_ch
                        self.densenet.features.conv0.weight.copy_(rep)

        num_ftrs = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 512), nn.ReLU(inplace=True), nn.Dropout(dropout_rate),
            nn.Linear(512, 128), nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.densenet(x)
        return self.classifier(features)

    @torch.no_grad()
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.densenet(x)