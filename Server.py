
import collections

import torchvision
import torch.distributed as dist
from torch.distributed import rpc

from Node import Node
from Model import *
from Dataset import *

class Server:
    def __init__(self,
        model_type:str,                     # specify model structure
        dataset_name:str,                   # specify dataset used for training
        dataset_type:str,                   # iid or noniid
        lr:float,                           # learning rate
        local_epoch:int,                    # local epoch num
        comm_round:int,                     # communication round
    ) -> None:
        prop_lens = [-1, # Server (placeholder)
            4, 2      # Nodes from node1 ... to ... node?
        ]
        self.__comm_round = comm_round

        if model_type == "mlp":
            self.__global_model = MLP()
        else:
            raise ValueError(f"Unrecognized model type: `{model_type}`")
        self.__world_size = dist.get_world_size()
        if dataset_name == "mnist":
            self.__testloader = MNIST(
                "~/RingSFL/datasets", torchvision.transforms.ToTensor(), 6, 64
            ).get_test_loader()
        else:
            raise ValueError(f"Unrecognized dataset name: `{dataset_name}`")

        # Creating training nodes ...
        self.__nodes_rref = [
            rpc.remote(
                f"node{i}",
                Node,
                args=(model_type, dataset_name, dataset_type, prop_lens[i], lr, 1/(self.__world_size-1), local_epoch)
            ) for i in range(1, self.__world_size)
        ]
        dist.new_group(range(1, self.__world_size))

        # Setting communication topology
        [node_rref.rpc_sync().set_next_node(
            self.__nodes_rref[(i + 1) % len(self.__nodes_rref)]
        ) for i, node_rref in enumerate(self.__nodes_rref)]

        # Initialize nodes
        [node_rref.rpc_sync().start_init() for node_rref in self.__nodes_rref]

    def train(self) -> None:
        for round in range(self.__comm_round):
            print(f"Round {round}")

            # Local training
            local_models = [node_rref.rpc_async().train(self.__global_model.state_dict()) for node_rref in self.__nodes_rref]
            local_models = [local_model.wait() for local_model in local_models]

            # Aggregation
            weight_keys = list(self.__global_model.state_dict().keys())
            global_dict = collections.OrderedDict()
            for key in weight_keys:
                key_sum = 0
                for i in range(len(local_models)):
                    key_sum += local_models[i][key]
                global_dict[key] = key_sum / len(local_models)
            self.__global_model.load_state_dict(global_dict)


