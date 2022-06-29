import sys
sys.path.append("..")
import torch

from model.ModelBase import ModelBase

class LeNet_Mnist(ModelBase):
    def __init__(self) -> None:
        super().__init__()
        self._blocks = torch.nn.ModuleList([ # 12 
            torch.nn.Conv2d(1, 6, 5, 1, 2),
            torch.nn.ReLU(inplace=True),
            torch.nn.AvgPool2d(2, 2),
            torch.nn.Conv2d(6, 16, 5),
            torch.nn.ReLU(inplace=True),
            torch.nn.AvgPool2d(2, 2),
            torch.nn.Flatten(),
            torch.nn.Linear(5*5*16, 120),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(120, 84),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(84, 10)
        ])
        self.block_num = len(self._blocks)

if __name__ == "__main__":
    data = torch.randn((1, 1, 28, 28))
    model = LeNet_Mnist()
    output = model(data)
    print(output)