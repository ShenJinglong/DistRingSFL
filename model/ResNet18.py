
import sys
sys.path.append("..")
import torch

from model.ModelBase import ModelBase

class BasicBlock(torch.nn.Module):
    expansion = 1
    def __init__(self,
        in_channels: int,
        out_channels: int,
        stride: int = 1
    ) -> None:
        super().__init__()

        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(out_channels)
        
        self.shortcut = torch.nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                torch.nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self,
        x: torch.Tensor
    ) -> torch.Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNet18_Cifar(ModelBase):
    def __init__(self,
        n_classes: int = 10
    ) -> None:
        super().__init__()
        self._blocks = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
                torch.nn.BatchNorm2d(64),
                torch.nn.ReLU(inplace=True),
            ),                              # 1
            BasicBlock(64, 64, 1),          # 2
            BasicBlock(64, 64, 1),          # 3
            BasicBlock(64, 128, 2),         # 4
            BasicBlock(128, 128, 1),        # 5
            BasicBlock(128, 256, 2),        # 6
            BasicBlock(256, 256, 1),        # 7
            BasicBlock(256, 512, 2),        # 8
            BasicBlock(512, 512, 1),        # 9
            torch.nn.Sequential(
                torch.nn.AvgPool2d(kernel_size=4, stride=1),
                torch.nn.Flatten(),
                torch.nn.Linear(512, n_classes),
            ),                              # 10
        ])
        self.block_num = len(self._blocks)
        self.__initialize_weights()

    def __initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1.0)
                torch.nn.init.constant_(m.bias, 0.0)
