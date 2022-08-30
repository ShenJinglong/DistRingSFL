
import sys
from typing import List
sys.path.append("..")
import logging
import wandb
import time
import torch

import torch.distributed as dist
from torch.distributed import rpc

from sl.Node import Node
from utils.model_utils import eval_splited_model, construct_model
from utils.data_utils import DatasetManager

class Server:
    def __init__(self,
        model_type:str,
        dataset_name:str,
        dataset_type:str,
        dataset_blocknum:int,
        lr:float,
        batch_size:int,
        local_epoch:int,
        comm_round:int,
        cut_point:int,
    ) -> None:
        self.__comm_round = comm_round
        self.__local_epoch = local_epoch
        self.__world_size = dist.get_world_size()
        self.__client_global_model, self.__server_global_model = construct_model(model_type).get_splited_module(cut_point)
        self.__loss_fn = torch.nn.CrossEntropyLoss()

        self.__nodes_rref = [
            rpc.remote(
                f"node{i}",
                Node,
                args=(rpc.RRef(self), model_type, lr, cut_point)
            ) for i in range(1, self.__world_size)
        ]

        dataset_manager = DatasetManager(dataset_name, "~/DistRingSFL/datasets", dataset_blocknum, batch_size)
        if dataset_type == "iid":
            self.__batch_num = len(dataset_manager.get_iid_loader(0))
            [node_rref.rpc_sync().set_trainloader(dataset_manager.get_iid_loader(i)) for i, node_rref in enumerate(self.__nodes_rref)]
        elif dataset_type == "noniid":
            self.__batch_num = len(dataset_manager.get_noniid_loader(0))
            [node_rref.rpc_sync().set_trainloader(dataset_manager.get_noniid_loader(i)) for i, node_rref in enumerate(self.__nodes_rref)]
        else:
            raise ValueError(f"Unrecognized dataset type: `{dataset_type}`")
        self.__testloader = dataset_manager.get_test_loader()
        
        [node_rref.rpc_sync().set_next_node(
            self.__nodes_rref[(i + 1) % len(self.__nodes_rref)]
        ) for i, node_rref in enumerate(self.__nodes_rref)]
        
        [node_rref.rpc_sync().start_init() for node_rref in self.__nodes_rref]
        logging.info("server initialized ...")

    def relay_forward(self,
        context_id:int,
        feature_map:torch.Tensor,
        labels:torch.Tensor
    ) -> None:
        outputs = self.__server_global_model(feature_map)
        loss = rpc.RRef(self.__loss_fn(outputs, labels))
        loss.backward(context_id)

    def relay_init(self) -> List[rpc.RRef]:
        return [rpc.RRef(param) for param in self.__server_global_model.parameters()]

    def eval(self,
        client_model_dict:dict
    ) -> None:
        logging.info("evaling ...")
        self.__client_global_model.load_state_dict(client_model_dict)
        acc = eval_splited_model(self.__client_global_model, self.__server_global_model, self.__testloader)
        time_cost = time.time() - self.__start_time
        wandb.log({
            'round': self.__round_counter,
            'acc': acc,
            'time': time_cost
        })
        logging.info(f"Round {self.__round_counter:3n}: acc - {acc:.4f}% | time cost - {time_cost:.4f}")
        self.__round_counter += 1
        if self.__round_counter != self.__comm_round:
            return True
        else:
            return False

    def train(self) -> None:
        logging.info("training ...")
        self.__start_time = time.time()
        self.__round_counter = 0
        self.__nodes_rref[0].rpc_async().train(self.__client_global_model.state_dict(), 0, self.__local_epoch*self.__batch_num)
        while self.__round_counter != self.__comm_round:
            pass
