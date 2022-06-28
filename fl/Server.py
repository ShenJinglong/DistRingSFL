
import sys

sys.path.append("..")
import logging
import wandb
import time

import torch.distributed as dist
from torch.distributed import rpc

from fl.Node import Node
from utils.model_utils import eval_model, aggregate_model, construct_model
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
    ) -> None:
        self.__comm_round = comm_round
        self.__world_size = dist.get_world_size()
        self.__global_model = construct_model(model_type)

        self.__nodes_rref = [
            rpc.remote(
                f"node{i}",
                Node,
                args=(model_type, lr, local_epoch)
            ) for i in range(1, self.__world_size)
        ]

        # Setting dataset
        dataset_manager = DatasetManager(dataset_name, "~/DistRingSFL/datasets", dataset_blocknum, batch_size)
        if dataset_type == "iid":
            [node_rref.rpc_sync().set_trainloader(dataset_manager.get_iid_loader(i)) for i, node_rref in enumerate(self.__nodes_rref)]
        elif dataset_type == "noniid":
            [node_rref.rpc_sync().set_trainloader(dataset_manager.get_noniid_loader(i)) for i, node_rref in enumerate(self.__nodes_rref)]
        else:
            raise ValueError(f"Unrecognized dataset type: `{dataset_type}`")
        self.__testloader = dataset_manager.get_test_loader()

    def train(self) -> None:
        start_time = time.time()
        for round in range(self.__comm_round):
            # Local training
            local_models = [node_rref.rpc_async().train(self.__global_model.state_dict()) for node_rref in self.__nodes_rref]
            local_models = [local_model.wait() for local_model in local_models]

            # Aggregation
            self.__global_model.load_state_dict(aggregate_model(local_models, [1 / len(local_models)] * len(local_models)))

            # Testing
            # acc = eval_model(self.__global_model, self.__testloader)
            time_cost = time.time() - start_time
            wandb.log({
                'round': round,
                # 'acc': acc,
                'time': time_cost
            })
            logging.info(f"Round {round:3n}: acc - {1:.4f}% | time cost - {time_cost:.4f}")

