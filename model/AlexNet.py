import sys
sys.path.append("..")
import torch

from model.ModelBase import ModelBase

class AlexNet_Mnist(ModelBase):
    def __init__(self) -> None:
        super().__init__()
        self._blocks = torch.nn.ModuleList([ # 17 
            torch.nn.Conv2d(1, 32, kernel_size=3, padding=1),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.Flatten(),
            torch.nn.Linear(256*3*3, 1024),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(512, 10)
        ])
        self.block_num = len(self._blocks)

if __name__ == "__main__":
    data = torch.randn((1, 1, 28, 28))
    model = AlexNet_Mnist()
    output = model(data)
    print(output)