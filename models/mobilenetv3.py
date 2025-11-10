import torch
import torch.nn as nn
from torchvision import models

class MobileNetV3(nn.Module):
    """
    MobileNetV3 model for Federated Learning, useful for resource-constrained devices
    like mobile phones and IoT devices.
    """
    def __init__(self, num_classes: int = 3, pretrained: bool = True, dropout_rate: float = 0.5):
        super(MobileNetV3, self).__init__()

        self.backbone = models.mobilenet_v3_large(pretrained=pretrained)

        self.backbone.features[0][0] = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False)

        num_features = self.backbone.classifier[0].in_features

        self.backbone.classifier = nn.Sequential()

        # Classifier
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
            nn.Linear(256, num_classes)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        output = self.classifier(features)
        return output

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
