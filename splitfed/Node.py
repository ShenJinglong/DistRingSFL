
import sys
sys.path.append("..")
import logging
import torch
import torch.distributed as dist
import torch.distributed.autograd as dist_autograd
from torch.distributed import rpc
from torch.distributed.optim import DistributedOptimizer

from utils.model_utils import construct_model

class Node:
    def __init__(self,
        server_rref:rpc.RRef,
        model_type:str,
        lr:float,
        local_epoch:int,
        cut_point:int
    ) -> None:
        self.__server_rref = server_rref
        self.__lr = lr
        self.__local_epoch = local_epoch
        self.__model = construct_model(model_type).get_splited_module(cut_point)[0]
        self.__node_group = dist.new_group(range(1, dist.get_world_size()))

    def set_trainloader(self,
        trainloader: torch.utils.data.DataLoader
    ) -> None:
        self.__trainloader = trainloader

    def train(self,
        global_model:dict
    ) -> dict:
        self.__model.load_state_dict(global_model)
        dist.barrier(self.__node_group)
        for epoch in range(self.__local_epoch):
            logging.info(f"epoch: {epoch:3n}")
            for inputs, labels in self.__trainloader:
                with dist_autograd.context() as context_id:
                    fm = self.__model(inputs)
                    self.__server_rref.rpc_sync().relay_forward(context_id, fm, labels)
                    dist.barrier(self.__node_group)
                    self.__dist_optim.step(context_id)
                    dist.barrier(self.__node_group)
        return self.__model.state_dict()

    def start_init(self) -> None:
        self.__rrefs = [rpc.RRef(param) for param in self.__model.parameters()] + self.__server_rref.rpc_sync().relay_init()
        self.__dist_optim = DistributedOptimizer(
            torch.optim.SGD,
            self.__rrefs,
            lr=self.__lr
        )
