
import sys
sys.path.append("..")
import logging
import time
import torch

from utils.model_utils import construct_model

class Node:
    def __init__(self,
        model_type:str,
        lr:float,
        local_epoch:int,
    ) -> None:
        self.__lr = lr
        self.__local_epoch = local_epoch
        self.__loss_fn = torch.nn.CrossEntropyLoss()
        self.__model = construct_model(model_type)
        self.__optim = torch.optim.SGD(self.__model.parameters(), lr=self.__lr)

    def set_trainloader(self,
        trainloader: torch.utils.data.DataLoader
    ) -> None:
        self.__trainloader = trainloader

    def train(self,
        global_model:dict
    ) -> dict:
        self.__model.load_state_dict(global_model)
        for epoch in range(self.__local_epoch):
            logging.info(f"Epoch: {epoch:3n}")
            for inputs, labels in self.__trainloader:
                start_time = time.time()
                self.__optim.zero_grad()
                outputs = self.__model(inputs)
                loss = self.__loss_fn(outputs, labels)
                logging.info(f"loss: {loss.item():.4f}")
                loss.backward()
                self.__optim.step()
                logging.info(f"time cost: {time.time() - start_time}")
        return self.__model.state_dict()
