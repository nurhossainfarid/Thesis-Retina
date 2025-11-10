import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Optional

class ResNet50(nn.Module):
    """
    ResNet50 model for more complex medical image analysis
    """
    def __init__(self, num_classes: int = 3, pretrained: bool = True, dropout_rate: float = 0.5):
        super(ResNet50, self).__init__()
        
        self.backbone = models.resnet50(pretrained=pretrained)
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=3, bias=False)
        
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

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


# def get_model(model_name: str, num_classes: int, pretrained: bool = True, dropout_rate: float = 0.5):
#     if model_name == 'resnet50':
#         return MedicalResNet50(num_classes=num_classes, pretrained=pretrained, dropout_rate=dropout_rate)
#     elif model_name == 'customcnn':
#         return CustomCNN(num_classes=num_classes)
#     else:
#         raise ValueError(f"Model {model_name} is not supported.")

    

# class FocalLoss(nn.Module):
#     """
#     Focal Loss for addressing class imbalance in medical image classification.
#     """
#     def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
#         super(FocalLoss, self).__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.reduction = reduction

#     def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
#         BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
#         pt = torch.exp(-BCE_loss)
#         F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        
#         if self.reduction == 'mean':
#             return F_loss.mean()
#         elif self.reduction == 'sum':
#             return F_loss.sum()
#         else:
#             return F_loss
        

# class LabelSmoothingLoss(nn.Module):
#     """
#     Label smoothing loss for medical image classification
#     """
#     def __init__(self, num_classes: int, smoothing: float = 0.1):
#         super(LabelSmoothingLoss, self).__init__()
#         self.num_classes = num_classes
#         self.smoothing = smoothing
#         self.confidence = 1.0 - smoothing
    
#     def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
#         log_probs = F.log_softmax(inputs, dim=1)
#         targets_one_hot = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
#         targets_smooth = targets_one_hot * self.confidence + (1 - targets_one_hot) * self.smoothing / (self.num_classes - 1)
#         loss = (-targets_smooth * log_probs).sum(dim=1).mean()
#         return loss