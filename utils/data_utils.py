
import logging
from typing import List
import numpy as np
import torch
import torchvision

class DatasetManager:
    def __init__(self,
        dataset_name:str,
        path:str,
        block_num:int,
        batch_size:int
    ) -> None:
        if dataset_name == "mnist":
            transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5,), (0.5,))
            ])
            self.__trainset = torchvision.datasets.MNIST(path, train=True, transform=transform, download=True)
            self.__testset = torchvision.datasets.MNIST(path, train=False, transform=transform, download=True)
        elif dataset_name == "cifar10":
            transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            self.__trainset = torchvision.datasets.CIFAR10(path, train=True, transform=transform, download=True)
            self.__testset = torchvision.datasets.CIFAR10(path, train=False, transform=transform, download=True)
        else:
            raise ValueError(f"Unrecognized dataset name: `{dataset_name}`")
        logging.info(f"dataset [{dataset_name}] loaded: {len(self.__trainset)} samples for training, {len(self.__testset)} samples for testing.")
        split_step = int(len(self.__trainset) / block_num)

        order = np.argsort(self.__trainset.targets)
        noniid_trainsubsets = [
            torch.utils.data.Subset(self.__trainset, order[split_step*i:split_step*(i+1)]) for i in range(block_num)
        ]
        self.__noniid_trainloaders = [
            torch.utils.data.DataLoader(trainsubset, batch_size=batch_size, shuffle=True, drop_last=True) for trainsubset in noniid_trainsubsets
        ]

        iid_trainsubsets = [
            torch.utils.data.Subset(self.__trainset, range(split_step*i, split_step*(i+1))) for i in range(block_num)
        ]
        self.__iid_trainloaders = [
            torch.utils.data.DataLoader(trainsubset, batch_size=batch_size, shuffle=True, drop_last=True) for trainsubset in iid_trainsubsets
        ]

        self.__testloader = torch.utils.data.DataLoader(self.__testset, batch_size=batch_size, shuffle=False, drop_last=True)

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

    def get_iid_loaders(self,
        top: int
    ) -> List[torch.utils.data.DataLoader]:
        if top > len(self.__iid_trainloaders) or top < 0:
            raise ValueError(f"Request top {top} out of range.")
        return self.__iid_trainloaders[:top]

    def get_noniid_loaders(self,
        top: int
    ) -> List[torch.utils.data.DataLoader]:
        if top > len(self.__noniid_trainloaders) or top < 0:
            raise ValueError(f"Request range {top} out of range.")
        return self.__noniid_trainloaders[:top]

    def get_test_loader(self) -> torch.utils.data.DataLoader:
        return self.__testloader
