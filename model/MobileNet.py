
import sys
sys.path.append("..")
import torch

from model.ModelBase import ModelBase

class MobileNet_Mnist(ModelBase): # 84
    def __init__(self) -> None:
        super().__init__()
        self._blocks = torch.nn.ModuleList([
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True),

            torch.nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),

            torch.nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True),

            torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True),

            torch.nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True),

            torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True),

            torch.nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True),

            torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True),

            torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True),

            torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True),

            torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True),

            torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True),

            torch.nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(512, 1024, kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.BatchNorm2d(1024),
            torch.nn.ReLU(inplace=True),

            torch.nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(1024),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.BatchNorm2d(1024),
            torch.nn.ReLU(inplace=True),

            torch.nn.AvgPool2d(2),
            torch.nn.Flatten(),
            torch.nn.Linear(1024, 10)
        ])
        self.block_num = len(self._blocks)

class MobileNetSimple_Mnist(ModelBase): # 12
    def __init__(self) -> None:
        super().__init__()
        self._blocks = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False),
                torch.nn.BatchNorm2d(32),
                torch.nn.ReLU(inplace=True),
            ),
            torch.nn.Sequential(
                torch.nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
                torch.nn.BatchNorm2d(32),
                torch.nn.ReLU(inplace=True),
            ),
            torch.nn.Sequential(
                torch.nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0, bias=False),
                torch.nn.BatchNorm2d(32),
                torch.nn.ReLU(inplace=True),
            ),
            torch.nn.Sequential(
                torch.nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
                torch.nn.BatchNorm2d(32),
                torch.nn.ReLU(inplace=True),
            ),
            torch.nn.Sequential(
                torch.nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=0, bias=False),
                torch.nn.BatchNorm2d(64),
                torch.nn.ReLU(inplace=True),
            ),
            torch.nn.Sequential(
                torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                torch.nn.BatchNorm2d(64),
                torch.nn.ReLU(inplace=True),
            ),
            torch.nn.Sequential(
                torch.nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=False),
                torch.nn.BatchNorm2d(64),
                torch.nn.ReLU(inplace=True),
            ),
            torch.nn.Sequential(
                torch.nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
                torch.nn.BatchNorm2d(64),
                torch.nn.ReLU(inplace=True),
            ),
            torch.nn.Sequential(
                torch.nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0, bias=False),
                torch.nn.BatchNorm2d(128),
                torch.nn.ReLU(inplace=True),
            ),
            torch.nn.Sequential(
                torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
                torch.nn.BatchNorm2d(128),
                torch.nn.ReLU(inplace=True),
            ),
            torch.nn.Sequential(
                torch.nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0, bias=False),
                torch.nn.BatchNorm2d(128),
                torch.nn.ReLU(inplace=True),
            ),
            torch.nn.Sequential(
                torch.nn.AvgPool2d(2),
                torch.nn.Flatten(),
                torch.nn.Linear(6272, 10)
            ),
        ])
        self.block_num = len(self._blocks)
        self.__initialize_weights()

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

if __name__ == "__main__":
    data = torch.randn((1, 1, 28, 28))
    model = MobileNet_Mnist()
    output = model(data)
    print(output)