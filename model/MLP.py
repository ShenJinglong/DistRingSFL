
import torch
from torch.distributed import rpc

class MLP_Mnist(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.__blocks = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(28 * 28, 500),
            ),                              # 1
            torch.nn.Linear(500, 500),      # 2
            torch.nn.Linear(500, 500),      # 3
            torch.nn.Linear(500, 500),      # 4
            torch.nn.Linear(500, 500),      # 5
            torch.nn.Linear(500, 500),      # 6
            torch.nn.Linear(500, 500),      # 7
            torch.nn.Linear(500, 500),      # 8
            torch.nn.Linear(500, 500),      # 9
            torch.nn.Linear(500, 500),      # 10
            torch.nn.Linear(500, 500),      # 11
            torch.nn.Linear(500, 400),      # 12
            torch.nn.Linear(400, 300),      # 13
            torch.nn.Linear(300, 200),      # 14
            torch.nn.Linear(200, 100),      # 15
            torch.nn.Linear(100, 10)        # 16
        ])
        self.block_num = len(self.__blocks)
    
    def forward(self, x, start=0, stop=16):
        for block in self.__blocks[start:stop]:
            x = block(x)
        return x

    def get_params(self, start=0, stop=16):
        return [param for param in self.__blocks[start:stop].parameters()]
    
    def get_rrefs(self, start=0, stop=16):
        return [rpc.RRef(param) for param in self.__blocks[start:stop].parameters()]