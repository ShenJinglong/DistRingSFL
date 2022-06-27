
import sys
sys.path.append("..")
import torch

from model.ModelBase import ModelBase

class MLP_Mnist(ModelBase):
    def __init__(self) -> None:
        super().__init__()
        self._blocks = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(28 * 28, 500),
            ),                              # 1
            torch.nn.Linear(500, 300),      # 2
            torch.nn.Linear(300, 100),      # 3
            torch.nn.Linear(100, 10),       # 4
        ])
        self.block_num = len(self._blocks)
