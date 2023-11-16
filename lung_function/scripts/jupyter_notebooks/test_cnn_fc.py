import torch
import numpy as np
import torch.nn as nn

class Cnn2fc1(nn.Module):

    def __init__(self, num_classes=1000):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 2, kernel_size=5, padding=2),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)

        )
        self.avgpool = nn.AdaptiveAvgPool2d((4, 3))
        self.classifier = nn.Linear(3, num_classes)
        # 自定义权重值
        custom_weights = torch.Tensor([[1,2, 3], [3,3, 3]])
        # 将自定义权重值赋给 nn.Linear 的权重
        self.classifier.weight.data = custom_weights
            

    def forward(self, x):
        B = x.shape[0]
        x = self.features(x)
        x = self.avgpool(x)
        print(x)
        print(x.shape)
        x = self.classifier(x)
        return x

input = torch.randn(3, 1, 64, 64)
model = Cnn2fc1(2)
out = model(input)

print(out)
print(out.shape)