
from typing import Tuple
import torch
from torch.distributed import rpc

class ModelBase(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x, start=0, stop=None):
        if stop == None:
            stop = self.block_num
        for block in self._blocks[start:stop]:
            x = block(x)
        return x

    def get_params(self, start=0, stop=None):
        if stop == None:
            stop = self.block_num
        return [param for param in self._blocks[start:stop].parameters()]
    
    def get_rrefs(self, start=0, stop=None):
        if stop == None:
            stop = self.block_num
        return [rpc.RRef(param) for param in self._blocks[start:stop].parameters()]

    def get_splited_module(self,
        cut_point:int
    ) -> Tuple[torch.nn.Module, torch.nn.Module]:
        if cut_point < 0 or cut_point > self.block_num:
            raise ValueError(f"Cut point {cut_point} out of module scope [0 - {self.block_num}].")
        return (torch.nn.Sequential(*self._blocks[:cut_point]), torch.nn.Sequential(*self._blocks[cut_point:]))
