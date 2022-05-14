
import torch
from torch.distributed import rpc

class VGG16_Cifar(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.__blocks = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=3, out_channels=64, padding=1, kernel_size=3, stride=1),
                torch.nn.BatchNorm2d(64),
                torch.nn.ReLU(inplace=True)
            ), # 1
            torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=64, out_channels=64, padding=1, kernel_size=3, stride=1),
                torch.nn.BatchNorm2d(64),
                torch.nn.ReLU(inplace=True),
                torch.nn.MaxPool2d(kernel_size=2, stride=2)
            ), # 2
            torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=64, out_channels=128, padding=1, kernel_size=3, stride=1),
                torch.nn.BatchNorm2d(128),
                torch.nn.ReLU(inplace=True)
            ), # 3
            torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=128, out_channels=128, padding=1, kernel_size=3, stride=1),
                torch.nn.BatchNorm2d(128),
                torch.nn.ReLU(inplace=True),
                torch.nn.MaxPool2d(kernel_size=2, stride=2)
            ), # 4
            torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=128, out_channels=256, padding=1, kernel_size=3, stride=1),
                torch.nn.BatchNorm2d(256),
                torch.nn.ReLU(inplace=True)
            ), # 5
            torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=256, out_channels=256, padding=1, kernel_size=3, stride=1),
                torch.nn.BatchNorm2d(256),
                torch.nn.ReLU(inplace=True)
            ), # 6
            torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=256, out_channels=256, padding=1, kernel_size=3, stride=1),
                torch.nn.BatchNorm2d(256),
                torch.nn.ReLU(inplace=True),
                torch.nn.MaxPool2d(kernel_size=2, stride=2)
            ), # 7
            torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=256, out_channels=512, padding=1, kernel_size=3, stride=1),
                torch.nn.BatchNorm2d(512),
                torch.nn.ReLU(inplace=True)
            ), # 8
            torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=512, out_channels=512, padding=1, kernel_size=3, stride=1),
                torch.nn.BatchNorm2d(512),
                torch.nn.ReLU(inplace=True)
            ), # 9
            torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=512, out_channels=512, padding=1, kernel_size=3, stride=1),
                torch.nn.BatchNorm2d(512),
                torch.nn.ReLU(inplace=True),
                torch.nn.MaxPool2d(kernel_size=2, stride=2)
            ), # 10
            torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=512, out_channels=512, padding=1, kernel_size=3, stride=1),
                torch.nn.BatchNorm2d(512),
                torch.nn.ReLU(inplace=True)
            ), # 11
            torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=512, out_channels=512, padding=1, kernel_size=3, stride=1),
                torch.nn.BatchNorm2d(512),
                torch.nn.ReLU(inplace=True)
            ), # 12
            torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=512, out_channels=512, padding=1, kernel_size=3, stride=1),
                torch.nn.BatchNorm2d(512),
                torch.nn.ReLU(inplace=True),
                torch.nn.MaxPool2d(kernel_size=2, stride=2)
            ), # 13
            torch.nn.Sequential(
                torch.nn.AdaptiveAvgPool2d((7,7)),
                torch.nn.Flatten(1),
                torch.nn.Linear(in_features=512*7*7, out_features=4096),
                torch.nn.ReLU(True),
                torch.nn.Dropout()
            ), # 14
            torch.nn.Sequential(
                torch.nn.Linear(in_features=4096, out_features=4096),
                torch.nn.ReLU(True),
                torch.nn.Dropout()
            ), # 15
            torch.nn.Sequential(
                torch.nn.Linear(in_features=4096, out_features=10)
            )  # 16
        ])
        self.block_num = len(self.__blocks)
        self.__initialize_weights()

    def forward(self, x, start=0, stop=16):
        for block in self.__blocks[start:stop]:
            x = block(x)
        return x

    def __initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, 0, 0.01)
                torch.nn.init.constant_(m.bias, 0)


    def get_params(self, start=0, stop=16):
        return [param for param in self.__blocks[start:stop].parameters()]

    def get_rrefs(self, start=0, stop=16):
        return [rpc.RRef(param) for param in self.__blocks[start:stop].parameters()]
