
import torch
from torch.distributed import rpc

class MLP_Mnist(torch.nn.Module):
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