
import numpy as np
import torch
import torchvision

class MNIST:
    def __init__(self,
        path:str,
        transform,
        block_num:int,
        batch_size:int
    ) -> None:
        self.__trainset = torchvision.datasets.MNIST(path, train=True, transform=transform, download=True)
        self.__testset = torchvision.datasets.MNIST(path, train=False, transform=transform, download=True)
        split_step = int(len(self.__trainset) / block_num)

        order = np.argsort(self.__trainset.targets)
        noniid_trainsubsets = [
            torch.utils.data.Subset(self.__trainset, order[split_step*i:split_step*(i+1)]) for i in range(block_num)
        ]
        self.__noniid_trainloaders = [
            torch.utils.data.DataLoader(trainsubset, batch_size=batch_size, shuffle=True) for trainsubset in noniid_trainsubsets
        ]

        iid_trainsubsets = [
            torch.utils.data.Subset(self.__trainset, range(split_step*i, split_step*(i+1))) for i in range(block_num)
        ]
        self.__iid_trainloaders = [
            torch.utils.data.DataLoader(trainsubset, batch_size=batch_size, shuffle=True) for trainsubset in iid_trainsubsets
        ]

        self.__testloader = torch.utils.data.DataLoader(self.__testset, batch_size=batch_size, shuffle=False)

    def get_iid_loader(self,
        index:int
    ) -> torch.utils.data.DataLoader:
        if index >= len(self.__iid_trainloaders) or index < 0:
            raise ValueError(f"Request index {index} out of range.")
        return self.__iid_trainloaders[index]

    def get_noniid_loader(self,
        index:int
    ) -> torch.utils.data.DataLoader:
        if index >= len(self.__noniid_trainloaders) or index < 0:
            raise ValueError(f"Request index {index} out of range.")
        return self.__noniid_trainloaders[index]

    def get_test_loader(self) -> torch.utils.data.DataLoader:
        return self.__testloader

class Cifar10:
    def __init__(self,
        path:str,
        transform,
        block_num:int,
        batch_size:int
    ) -> None:
        self.__trainset = torchvision.datasets.CIFAR10(path, train=True, transform=transform, download=True)
        self.__testset = torchvision.datasets.CIFAR10(path, train=False, transform=transform, download=True)
        split_step = int(len(self.__trainset) / block_num)

        order = np.argsort(self.__trainset.targets)
        noniid_trainsubsets = [
            torch.utils.data.Subset(self.__trainset, order[split_step*i:split_step*(i+1)]) for i in range(block_num)
        ]
        self.__noniid_trainloaders = [
            torch.utils.data.DataLoader(trainsubset, batch_size=batch_size, shuffle=True) for trainsubset in noniid_trainsubsets
        ]

        iid_trainsubsets = [
            torch.utils.data.Subset(self.__trainset, range(split_step*i, split_step*(i+1))) for i in range(block_num)
        ]
        self.__iid_trainloaders = [
            torch.utils.data.DataLoader(trainsubset, batch_size=batch_size, shuffle=True) for trainsubset in iid_trainsubsets
        ]

        self.__testloader = torch.utils.data.DataLoader(self.__testset, batch_size=batch_size, shuffle=False)

    def get_iid_loader(self,
        index:int
    ) -> torch.utils.data.DataLoader:
        if index >= len(self.__iid_trainloaders) or index < 0:
            raise ValueError(f"Request index {index} out of range.")
        return self.__iid_trainloaders[index]

    def get_noniid_loader(self,
        index:int
    ) -> torch.utils.data.DataLoader:
        if index >= len(self.__noniid_trainloaders) or index < 0:
            raise ValueError(f"Request index {index} out of range.")
        return self.__noniid_trainloaders[index]

    def get_test_loader(self) -> torch.utils.data.DataLoader:
        return self.__testloader

if __name__ == "__main__":
    data = MNIST("./datasets", torchvision.transforms.ToTensor(), 6, 64)
    print(data.get_iid_loader(2))
    print(data.get_noniid_loader(5))
    print(data.get_test_loader())
    # print(data.get_noniid_loader(7))

    cifar10 = Cifar10("./datasets", torchvision.transforms.ToTensor(), 60, 64)
    print(data.get_iid_loader(2))
    print(data.get_noniid_loader(5))
    print(data.get_test_loader())