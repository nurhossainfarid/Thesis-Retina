import torch
import torch.nn as nn
from torchvision import models

class DenseNet121Medical(nn.Module):
    """
    DenseNet121 model for medical image classification using ImageNet pretrained weights.
    Converts first convolution to accept 1-channel input by averaging pretrained weights.
    """
    def __init__(self, num_classes: int = 3, pretrained: bool = True, dropout_rate: float = 0.5, dataset: str = "chexpert"):
        super(DenseNet121Medical, self).__init__()

        if pretrained:
            self.densenet = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        else:
            self.densenet = models.densenet121(weights=None)
        
        num_ftrs = self.densenet.classifier.in_features 

        with torch.no_grad():
            w = self.densenet.features.conv0.weight 
            self.densenet.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.densenet.features.conv0.weight.copy_(w.mean(dim=1, keepdim=True)) 

        self.densenet.classifier = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the DenseNet121 model.
        x: [B, 1, H, W] (grayscale) is expected.
        """
        features = self.densenet(x) 
        return self.classifier(features)  

    @torch.no_grad()
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from the DenseNet backbone without classification.
        """
        return self.densenet(x) 