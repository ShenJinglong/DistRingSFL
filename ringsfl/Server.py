
import sys
sys.path.append("..")
import collections
import logging
import wandb

import torchvision
import torch.distributed as dist
from torch.distributed import rpc

from ringsfl.Node import Node
from model.MLP import *
from model.VGG16 import *
from model.ResNet18 import *
from data.MNIST import *
from data.Cifar10 import *

DATASET_PATH = "~/DistRingSFL/datasets"

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

        if model_type == "mlp":
            self.__global_model = MLP_Mnist()
        elif model_type == "vgg16":
            self.__global_model = VGG16_Cifar()
        elif model_type == "resnet18":
            self.__global_model = ResNet18_Cifar()
        else:
            raise ValueError(f"Unrecognized model type: `{model_type}`")
        if dataset_name == "mnist":
            self.__testloader = MNIST(
                DATASET_PATH, torchvision.transforms.ToTensor(), dataset_blocknum, batch_size
            ).get_test_loader()
        elif dataset_name == "cifar10":
            self.__testloader = Cifar10(
                DATASET_PATH, torchvision.transforms.ToTensor(), dataset_blocknum, batch_size
            ).get_test_loader()
        else:
            raise ValueError(f"Unrecognized dataset name: `{dataset_name}`")

        # Creating training nodes ...
        self.__nodes_rref = [
            rpc.remote(
                f"node{i}",
                Node,
                args=(model_type, dataset_name, dataset_type, dataset_blocknum, lr, batch_size, local_epoch, prop_lens[i], 1/(self.__world_size-1))
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

            # Testing
            correct, total = 0, 0
            self.__global_model.eval()
            with torch.no_grad():
                for inputs, labels in self.__testloader:
                    outputs = self.__global_model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            acc = 100 * correct / total
            logging.info(f"Round {round:3n}: acc - {acc:.4f}%")
