
import sys
import time
sys.path.append("..")
import logging
from typing import List

import torch
import torchvision
import torch.distributed as dist
import torch.distributed.autograd as dist_autograd
from torch.distributed import rpc
from torch.distributed.optim import DistributedOptimizer

from model.MLP import *
from model.VGG16 import *
from model.ResNet18 import *
from data.MNIST import *
from data.Cifar10 import *

DATASET_PATH = "~/DistRingSFL/datasets"

class Node:
    def __init__(self,
        model_type:str,                     # specify model structure
        dataset_name:str,                   # specify dataset used for training
        dataset_type:str,                   # iid or noniid
        dataset_blocknum:int,               # how many blocks to divide the dataset into
        lr:float,                           # learning rate
        batch_size:int,                     # batch size
        local_epoch:int,                    # local epoch num
        prop_len:int,                       # propagation length
        aggreg_weight:float,                # aggregation weight
    ) -> None:
        self.__prop_len = prop_len
        self.__lr = lr * aggreg_weight
        self.__local_epoch = local_epoch
        self.__label_cache = None
        self.__node_group = dist.new_group(range(1, dist.get_world_size()))

        self.__loss_fn = torch.nn.CrossEntropyLoss()
        if model_type == "mlp":
            self.__model = MLP_Mnist()
        elif model_type == "vgg16":
            self.__model = VGG16_Cifar()
        elif model_type == "resnet18":
            self.__model = ResNet18_Cifar()
        else:
            raise ValueError(f"Unrecognized model type: `{model_type}`")
        
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

    def set_next_node(self,
        next_node_rref:rpc.RRef             # remote reference of the next node in the ring
    ) -> None:
        self.__next_node = next_node_rref

    def train(self,
        global_model:dict                   # state dict of global model
    ) -> dict:                              # state dict of local model
        self.__model.load_state_dict(global_model)
        for epoch in range(self.__local_epoch):
            logging.info(f"Epoch: {epoch:3n}")
            for data, label in self.__trainloader:
                with dist_autograd.context() as context_id:
                    self.__label_cache = label
                    self.start_forward(context_id, data)
                    dist.barrier(self.__node_group)
                    self.__dist_optim.step(context_id)
                    dist.barrier(self.__node_group)
        return self.__model.state_dict()

    def start_forward(self,
        context_id:int,                     # context id for backward propagation
        x:torch.Tensor                      # a batch of input data sample
    ) -> None:
        start, stop = 0, self.__prop_len
        start_time = time.time()
        output = self.__model(x, start=start, stop=stop)
        logging.info(f"time cost: {time.time() - start_time}")
        if stop < self.__model.block_num:
            self.__next_node.rpc_sync().relay_forward(context_id, output, stop)
        elif stop == self.__model.block_num:
            self.__next_node.rpc_sync().stop_forward(context_id, output)
        else:
            raise ValueError("Stop layer output of model scope.")

    def relay_forward(self,
        context_id:int,                     # context id for backward propagation
        x:torch.Tensor,                     # feature map outputed by previous node
        start:int                           # index of the starting layer
    ) -> None:
        stop = start + self.__prop_len
        if stop > self.__model.block_num:
            raise ValueError("Stop layer output of model scope.")
        start_time = time.time()
        output = self.__model(x, start=start, stop=stop)
        logging.info(f"time cost: {time.time() - start_time}")
        if stop == self.__model.block_num:
            self.__next_node.rpc_sync().stop_forward(context_id, output)
        else:
            self.__next_node.rpc_sync().relay_forward(context_id, output, stop)

    def stop_forward(self,
        context_id:int,                     # context id for backward propagation
        x:torch.Tensor                      # model output
    ) -> None:
        loss = self.__loss_fn(x, self.__label_cache)
        logging.info(f"loss: {loss.item():.4f}")
        rpc.RRef(loss).backward(context_id)

    def start_init(self) -> None:
        start, stop = 0, self.__prop_len
        self.__rrefs = self.__model.get_rrefs(start=start, stop=stop)
        if stop < self.__model.block_num:
            ret = self.__next_node.rpc_sync().relay_init(stop)
            self.__rrefs.extend(ret)
        self.__dist_optim = DistributedOptimizer(
            torch.optim.SGD,
            self.__rrefs,
            lr = self.__lr
        )

    def relay_init(self,
        start:int                           # index of the starting layer
    ) -> List[rpc.RRef]:                    # remote reference of the parameters participated in training
        stop = start + self.__prop_len
        if stop > self.__model.block_num:
            raise ValueError("Stop layer output of model scope.")
        rrefs = self.__model.get_rrefs(start=start, stop=stop)
        if stop == self.__model.block_num:
            return rrefs
        else:
            ret = self.__next_node.rpc_sync().relay_init(stop)
            rrefs.extend(ret)
            return rrefs

