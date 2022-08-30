
import sys
sys.path.append("..")
import torch

from model.ModelBase import ModelBase

class CNN_Mnist(ModelBase):
    def __init__(self) -> None:
        super().__init__()
        self._blocks = torch.nn.ModuleList([ # 14
            torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(32, 64, 5, 1, 2),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(64, 128, 5, 1, 2),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(128, 256, 5, 1, 2),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Flatten(),
            torch.nn.Linear(256, 10)
        ])
        self.block_num = len(self._blocks)

if __name__ == "__main__":
    data = torch.randn((1, 1, 28, 28))
    model = CNN_Mnist()
    output = model(data)
    print(output)