
import torch
from torch.distributed import rpc

class CNN_Mnist(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.__blocks = torch.nn.ModuleList([
            torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(32, 64, 5, 1, 2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(64, 128, 5, 1, 2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(128, 256, 5, 1, 2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Flatten(),
            torch.nn.Linear(256, 10)
        ])
        self.block_num = len(self.__blocks)
    
    def forward(self, x, start=0, stop=14):
        for block in self.__blocks[start:stop]:
            x = block(x)
        return x

    def get_params(self, start=0, stop=14):
        return [param for param in self.__blocks[start:stop].parameters()]
    
    def get_rrefs(self, start=0, stop=14):
        return [rpc.RRef(param) for param in self.__blocks[start:stop].parameters()]

if __name__ == "__main__":
    data = torch.randn((1, 1, 28, 28))
    model = CNN_Mnist()
    output = model(data)
    print(output)