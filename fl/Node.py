
import sys
sys.path.append("..")
import logging
from typing import List

import torch
import torchvision
import torch.distributed as dist

from model.MLP import *
from model.VGG16 import *
from model.ResNet18 import *
from data.MNIST import *
from data.Cifar10 import *

DATASET_PATH = "~/DistRingSFL/datasets"

class Node:
    def __init__(self,
        model_type:str,
        dataset_name:str,
        dataset_type:str,
        dataset_blocknum:int,
        lr:float,
        batch_size:int,
        local_epoch:int,
    ) -> None:
        self.__lr = lr
        self.__local_epoch = local_epoch

        self.__loss_fn = torch.nn.CrossEntropyLoss()
        if model_type == "mlp":
            self.__model = MLP_Mnist()
        elif model_type == "vgg16":
            self.__model = VGG16_Cifar()
        elif model_type == "resnet18":
            self.__model = ResNet18_Cifar()
        else:
            raise ValueError(f"Unrecognized model type: `{model_type}`")
        self.__optim = torch.optim.SGD(self.__model.parameters(), lr=self.__lr)

        if dataset_name == "mnist":
            self.__trainloader = MNIST(
                DATASET_PATH, torchvision.transforms.ToTensor(), dataset_blocknum, batch_size
            ).get_iid_loader(dist.get_rank()-1) if dataset_type == "iid" else MNIST(
                DATASET_PATH, torchvision.transforms.ToTensor(), dataset_blocknum, batch_size
            ).get_noniid_loader(dist.get_rank()-1)
        elif dataset_name == "cifar10":
            self.__trainloader = Cifar10(
                DATASET_PATH, torchvision.transforms.ToTensor(), dataset_blocknum, batch_size
            ).get_iid_loader(dist.get_rank()-1) if dataset_type == "iid" else Cifar10(
                DATASET_PATH, torchvision.transforms.ToTensor(), dataset_blocknum, batch_size
            ).get_noniid_loader(dist.get_rank()-1)
        else:
            raise ValueError(f"Unrecognized dataset name: `{dataset_name}`")

    def train(self,
        global_model:dict
    ) -> dict:
        self.__model.load_state_dict(global_model)
        for epoch in range(self.__local_epoch):
            logging.info(f"Epoch: {epoch:3n}")
            for inputs, labels in self.__trainloader:
                self.__optim.zero_grad()
                outputs = self.__model(inputs)
                loss = self.__loss_fn(outputs, labels)
                print(f"loss: {loss.item():.4f}")
                loss.backward()
                self.__optim.step()
        return self.__model.state_dict()
