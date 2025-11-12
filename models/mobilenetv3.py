# mobilenetv3.py
import torch
import torch.nn as nn
from torchvision import models

class MobileNetV3(nn.Module):
    """
    MobileNetV3-Large with flexible input channels (1 or 3).
    """
    def __init__(self, num_classes: int = 3, in_ch: int = 3, pretrained: bool = True, dropout_rate: float = 0.5):
        super().__init__()
        self.backbone = models.mobilenet_v3_large(pretrained=pretrained)

        # First conv is features[0][0] (Conv2d 3->16). Replace if needed.
        if in_ch != 3:
            old = self.backbone.features[0][0]  # Conv2d(3,16,3,2,1)
            new_conv = nn.Conv2d(in_ch, old.out_channels, kernel_size=old.kernel_size,
                                 stride=old.stride, padding=old.padding, bias=False)
            if pretrained:
                with torch.no_grad():
                    if in_ch == 1:
                        new_conv.weight.copy_(old.weight.mean(dim=1, keepdim=True))
                    else:
                        rep = torch.mean(old.weight, dim=1, keepdim=True).repeat(1, in_ch, 1, 1) / in_ch
                        new_conv.weight.copy_(rep)
            self.backbone.features[0][0] = new_conv

        # keep backbone classifier head to get features dim
        num_features = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Sequential()  # remove original fc head

        self.classifier = nn.Sequential(
            nn.BatchNorm1d(num_features),
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 512), nn.ReLU(inplace=True), nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(512, 256), nn.ReLU(inplace=True), nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate * 0.25),
            nn.Linear(256, num_classes)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)  # (B, num_features)
        return self.classifier(features)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)