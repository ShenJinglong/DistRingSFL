
import sys
sys.path.append("..")
import logging
import wandb
import time

import torch.distributed as dist
from torch.distributed import rpc

from ringsfl.Node import Node
from utils.model_utils import eval_model, aggregate_model, construct_model
from utils.data_utils import DatasetManager

class Server:
    def __init__(self,
        prop_ratio:str,                     # propagation ratio of each node
        model_type:str,                     # specify model structure
        dataset_name:str,                   # specify dataset used for training
        dataset_type:str,                   # iid or noniid
        dataset_blocknum:int,               # how many blocks to divide the dataset into
        lr:float,                           # learning rate
        batch_size:int,                     # batch size
        local_epoch:int,                    # local epoch num
        comm_round:int,                     # communication round
    ) -> None:
        prop_lens = [-1,] # Server (placeholder)
        prop_lens.extend([int(a) for a in prop_ratio.split(':')])
        self.__comm_round = comm_round
        self.__world_size = dist.get_world_size()
        self.__global_model = construct_model(model_type)

        # Creating training nodes ...
        self.__nodes_rref = [
            rpc.remote(
                f"node{i}",
                Node,
                args=(model_type, lr*(self.__world_size-1), local_epoch, prop_lens[i], 1/(self.__world_size-1))
            ) for i in range(1, self.__world_size)
        ]
        dist.new_group(range(1, self.__world_size))

        # Setting dataset
        dataset_manager = DatasetManager(dataset_name, "~/projects/DistRingSFL/datasets", dataset_blocknum, batch_size)
        if dataset_type == "iid":
            [node_rref.rpc_sync().set_trainloader(dataset_manager.get_iid_loader(i)) for i, node_rref in enumerate(self.__nodes_rref)]
        elif dataset_type == "noniid":
            [node_rref.rpc_sync().set_trainloader(dataset_manager.get_noniid_loader(i)) for i, node_rref in enumerate(self.__nodes_rref)]
        else:
            raise ValueError(f"Unrecognized dataset type: `{dataset_type}`")
        self.__testloader = dataset_manager.get_test_loader()

        # Setting communication topology
        [node_rref.rpc_sync().set_next_node(
            self.__nodes_rref[(i + 1) % len(self.__nodes_rref)]
        ) for i, node_rref in enumerate(self.__nodes_rref)]

        # Initialize nodes
        [node_rref.rpc_sync().start_init() for node_rref in self.__nodes_rref]

    def train(self) -> None:
        start_time = time.time()
        for round in range(self.__comm_round):
            # Local training
            local_datas = [node_rref.rpc_async().train(self.__global_model.state_dict()) for node_rref in self.__nodes_rref]
            local_datas = [local_data.wait() for local_data in local_datas]
            local_models = [local_data[0] for local_data in local_datas]

            # Aggregation
            self.__global_model.load_state_dict(aggregate_model(local_models, [1 / len(local_models)] * len(local_models)))

            # Testing
            acc = eval_model(self.__global_model, self.__testloader)

            # Logging
            time_cost = time.time() - start_time
            wandb.log({
                'round': round,
                'acc': acc,
                'time': time_cost
            })
            log_msg = f"Round {round:3n}: acc - {acc:.4f}% | time cost - {time_cost:.4f} |"
            for local_data in local_datas:
                log_msg += f" ({local_data[1]},{local_data[2]})"
            logging.info(log_msg)
