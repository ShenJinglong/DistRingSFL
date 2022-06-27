
import sys
sys.path.append("..")
import logging
from typing import List

import torch
import torch.distributed as dist
import torch.distributed.autograd as dist_autograd
from torch.distributed import rpc
from torch.distributed.optim import DistributedOptimizer

from utils.model_utils import construct_model

class Node:
    def __init__(self,
        model_type:str,                     # specify model structure
        lr:float,                           # learning rate
        local_epoch:int,                    # local epoch num
        prop_len:int,                       # propagation length
        aggreg_weight:float,                # aggregation weight
    ) -> None:
        self.__prop_len = prop_len
        self.__lr = lr * aggreg_weight
        self.__local_epoch = local_epoch
        self.__loss_fn = torch.nn.CrossEntropyLoss()
        self.__model = construct_model(model_type)
        logging.info(f"propagation length: {self.__prop_len}")
        self.__node_group = dist.new_group(range(1, dist.get_world_size()))

    def set_next_node(self,
        next_node_rref:rpc.RRef             # remote reference of the next node in the ring
    ) -> None:
        logging.info(f"setting up next node: {next_node_rref.owner_name()}")
        self.__next_node = next_node_rref

    def set_trainloader(self,
        trainloader:torch.utils.data.DataLoader
    ) -> None:
        self.__trainloader = trainloader

    def train(self,
        global_model:dict                   # state dict of global model
    ) -> dict:                              # state dict of local model
        self.__model.load_state_dict(global_model)
        dist.barrier(self.__node_group)
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
        output = self.__model(x, start=start, stop=stop)
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
        output = self.__model(x, start=start, stop=stop)
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

