
from turtle import forward
import torch
from torch.distributed import rpc

############################################

class MLP(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.__blocks = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(28 * 28, 500),
            ),
            torch.nn.Linear(500, 400),
            torch.nn.Linear(400, 300),
            torch.nn.Linear(300, 200),
            torch.nn.Linear(200, 100),
            torch.nn.Linear(100, 10)
        ])
        self.block_num = len(self.__blocks)
    
    def forward(self, x, start=0, stop=6):
        for block in self.__blocks[start:stop]:
            x = block(x)
        return x

    def get_params(self, start=0, stop=6):
        return [param for param in self.__blocks[start:stop].parameters()]
    
    def get_rrefs(self, start=0, stop=6):
        return [rpc.RRef(param) for param in self.__blocks[start:stop].parameters()]

############################################

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

############################################

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


class ResNet18_Cifar(torch.nn.Module):
    def __init__(self,
        n_classes: int = 10
    ) -> None:
        super().__init__()
        self.__blocks = torch.nn.ModuleList([
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
        self.block_num = len(self.__blocks)
        self.__initialize_weights()


    def forward(self, x, start=0, stop=10):
        for block in self.__blocks[start:stop]:
            x = block(x)
        return x

    def get_params(self, start=0, stop=10):
        return [param for param in self.__blocks[start:stop].parameters()]

    def get_rrefs(self, start=0, stop=10):
        return [rpc.RRef(param) for param in self.__blocks[start:stop].parameters()]

    def __initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1.0)
                torch.nn.init.constant_(m.bias, 0.0)

if __name__ == "__main__":
    mlp = MLP()
    print(mlp, mlp.block_num)
    dummy_data = torch.randn(size=(64, 1, 28, 28))
    print(dummy_data.shape)
    output = mlp(dummy_data)
    print(output)
    # params = mlp.get_params(5, 6)
    # for param in params:
    #     print(param)

