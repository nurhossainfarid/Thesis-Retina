# # customcnn.py
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class CustomCNN(nn.Module):
#     def __init__(self, num_classes: int = 3, in_ch: int = 3):
#         super().__init__()

#         self.features = nn.Sequential(
#             nn.Conv2d(in_ch, 32, kernel_size=3, padding=1),
#             nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),

#             nn.Conv2d(32, 64, kernel_size=3, padding=1),
#             nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),

#             nn.Conv2d(64, 128, kernel_size=3, padding=1),
#             nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),

#             nn.Conv2d(128, 256, kernel_size=3, padding=1),
#             nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),

#             nn.AdaptiveAvgPool2d((1, 1))
#         )

#         self.classifier = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(256, 128),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.4),
#             nn.Linear(128, num_classes)
#         )

#         self._initialize_weights()

#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None: nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, mean=0.0, std=0.01)
#                 nn.init.constant_(m.bias, 0)

#     def forward(self, x):
#         x = self.features(x)
#         x = self.classifier(x)
#         return x



# customcnn.py  (Upgraded: residual blocks, GELU, a bit wider)
# customcnn.py
import torch
import torch.nn as nn

def conv_bn_act(in_ch, out_ch, k=3, s=1, p=1, act=True):
    layers = [
        nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
        nn.BatchNorm2d(out_ch)
    ]
    if act:
        layers.append(nn.GELU())
    return nn.Sequential(*layers)

class ResidualBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.block = nn.Sequential(
            conv_bn_act(ch, ch, k=3, s=1, p=1, act=True),
            conv_bn_act(ch, ch, k=3, s=1, p=1, act=False),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))

class CustomCNN(nn.Module):
    """
    Stronger custom CNN for 9-class retinal classification.
    - Residual blocks + GELU
    - Works with in_ch = 3 (RGB) or 1 (grayscale)
    """
    def __init__(self, num_classes: int = 9, in_ch: int = 3, drop: float = 0.3):
        super().__init__()
        widths = [48, 96, 192, 256]

        self.stem = conv_bn_act(in_ch, widths[0], k=3, s=1, p=1)

        self.stage1 = nn.Sequential(
            ResidualBlock(widths[0]),
            ResidualBlock(widths[0]),
            nn.MaxPool2d(2)  # 112x112 -> 56x56
        )
        self.stage2 = nn.Sequential(
            conv_bn_act(widths[0], widths[1]),
            ResidualBlock(widths[1]),
            ResidualBlock(widths[1]),
            nn.MaxPool2d(2)  # 56x56 -> 28x28
        )
        self.stage3 = nn.Sequential(
            conv_bn_act(widths[1], widths[2]),
            ResidualBlock(widths[2]),
            ResidualBlock(widths[2]),
            nn.MaxPool2d(2)  # 28x28 -> 14x14
        )
        self.stage4 = nn.Sequential(
            conv_bn_act(widths[2], widths[3]),
            ResidualBlock(widths[3]),
            nn.MaxPool2d(2)  # 14x14 -> 7x7
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(drop),
            nn.Linear(widths[3], 256),
            nn.GELU(),
            nn.Dropout(drop * 0.5),
            nn.Linear(256, num_classes)
        )

        self._init()

    def _init(self):
        # NOTE: torch init calculate_gain doesn't support 'gelu',
        # so we use ReLU gain (standard practice for GELU nets).
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return self.head(x)