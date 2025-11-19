import torch
import torch.nn as nn
import torch.nn.functional as F

from models.resnet_model import ResNet50
from models.cnn_model import CustomCNN
from models.mobilenetv3 import MobileNetV3
from models.DenseNet121 import DenseNet121Medical
from models.hybridSwinDenseNetMLP import HybridSwinDenseNetMLP
from models.EfficientNetB3RCA import EfficientNetB3RCA

def get_model(model_name: str, num_classes: int, pretrained: bool = True, dropout_rate: float = 0.5):
    
    if model_name == 'resnet50':
        return ResNet50(num_classes=num_classes, pretrained=pretrained, dropout_rate=dropout_rate)
    elif model_name == 'customcnn':
        return CustomCNN(num_classes=num_classes)
    elif model_name == "mobilenetv3":
        return MobileNetV3(num_classes=num_classes, pretrained=pretrained, dropout_rate=dropout_rate)
    elif model_name == "shufflenetv2":
        from models.ShuffleNetV2 import ShuffleNetV2
        return ShuffleNetV2(num_classes=num_classes, pretrained=pretrained, dropout_rate=dropout_rate)
    elif model_name == "densenet121":
        return DenseNet121Medical(num_classes=num_classes, pretrained=pretrained, dropout_rate=dropout_rate)
    elif model_name == "hybridswin":
        return HybridSwinDenseNetMLP(num_classes=num_classes, pretrained=pretrained, dropout=dropout_rate)
    elif model_name == "efficientnetb3rca":
        return EfficientNetB3RCA(num_classes=num_classes, pretrained=pretrained, dropout_rate=dropout_rate)
    else:
        raise ValueError(f"Model {model_name} is not supported.")



class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in medical image classification.
    """
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss


class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing loss for medical image classification
    """
    def __init__(self, num_classes: int, smoothing: float = 0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(inputs, dim=1)
        targets_one_hot = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets_smooth = targets_one_hot * self.confidence + (1 - targets_one_hot) * self.smoothing / (self.num_classes - 1)
        loss = (-targets_smooth * log_probs).sum(dim=1).mean()
        return loss
