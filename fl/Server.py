
import sys
sys.path.append("..")
import collections
import logging
import wandb
import time

import torchvision
import torch.distributed as dist
from torch.distributed import rpc

from fl.Node import Node
from model.MLP import *
from model.VGG16 import *
from model.ResNet18 import *
from data.MNIST import *
from data.Cifar10 import *

DATASET_PATH = "~/DistRingSFL/datasets"

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

        self.__nodes_rref = [
            rpc.remote(
                f"node{i}",
                Node,
                args=(model_type, dataset_name, dataset_type, dataset_blocknum, lr, batch_size, local_epoch)
            ) for i in range(1, self.__world_size)
        ]

    def train(self) -> None:
        start_time = time.time()
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
            time_cost = time.time() - start_time
            wandb.log({
                'round': round,
                'acc': acc,
                'time': time_cost
            })
            logging.info(f"Round {round:3n}: acc - {acc:.4f}% | time cost - {time_cost:.4f}")

