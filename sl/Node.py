
import sys
sys.path.append("..")
import logging
import torch
import torch.distributed.autograd as dist_autograd
from torch.distributed import rpc
from torch.distributed.optim import DistributedOptimizer

from utils.model_utils import construct_model

class Node:
    def __init__(self,
        server_rref:rpc.RRef,
        model_type:str,
        lr:float,
        cut_point:int
    ) -> None:
        self.__server_rref = server_rref
        self.__lr = lr
        self.__model = construct_model(model_type).get_splited_module(cut_point)[0]

    def set_next_node(self,
        next_node_rref:rpc.RRef
    ) -> None:
        logging.info(f"setting up next node: {next_node_rref.owner_name()}")
        self.__next_node = next_node_rref

    def set_trainloader(self,
        trainloader: torch.utils.data.DataLoader
    ) -> None:
        self.__trainloader = trainloader
        self.__trainloader_iter = iter(self.__trainloader)

    def train(self,
        client_model:dict,
        step_counter:int,
        eval_step:int,
    ) -> None:
        logging.info(f"training {step_counter} ...")
        self.__model.load_state_dict(client_model)
        try:
            inputs, labels = self.__trainloader_iter.next()
        except StopIteration:
            self.__trainloader_iter = iter(self.__trainloader)
            inputs, labels = self.__trainloader_iter.next()
            
        with dist_autograd.context() as context_id:
            fm = self.__model(inputs)
            self.__server_rref.rpc_sync().relay_forward(context_id, fm, labels)
            self.__dist_optim.step(context_id)
        
        if step_counter == (eval_step-1):
            flag = self.__server_rref.rpc_sync().eval(self.__model.state_dict())
            if flag == True:
                step_counter = 0
                self.__next_node.rpc_async().train(self.__model.state_dict(), step_counter, eval_step)
        else:
            step_counter += 1
            self.__next_node.rpc_async().train(self.__model.state_dict(), step_counter, eval_step)


    def start_init(self) -> None:
        self.__rrefs = [rpc.RRef(param) for param in self.__model.parameters()] + self.__server_rref.rpc_sync().relay_init()
        self.__dist_optim = DistributedOptimizer(
            torch.optim.SGD,
            self.__rrefs,
            lr=self.__lr
        )
        logging.info("client initialized ...")
