
import string
import sys
sys.path.append("..")
import collections
from typing import List
import torch

from model.MLP import *
from model.VGG16 import *
from model.ResNet18 import *
from model.CNN import *
from model.MobileNet import *

def aggregate_model(
    models: List[dict],
    weights: List[float]
) -> dict:
    global_dict = collections.OrderedDict()
    param_keys = models[0].keys()
    for key in param_keys:
        sum = 0
        for weight, model in zip(weights, models):
            sum += weight * model[key]
        global_dict[key] = sum
    return global_dict

def eval_model(
    model: torch.nn.Module,
    testloader: torch.utils.data.DataLoader
) -> float:
    model.eval()
    device = next(model.parameters()).device
    total, correct = 0, 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            # print(torch.nn.CrossEntropyLoss()(outputs, labels))
            _, pred = outputs.max(1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
    model.train()
    return 100 * correct / total

def eval_splited_model(
    shallow_model: torch.nn.Module,
    deep_model: torch.nn.Module,
    testloader: torch.utils.data.DataLoader
) -> float:
    shallow_model.eval()
    deep_model.eval()
    device = next(shallow_model.parameters()).device
    total, correct = 0, 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            fm = shallow_model(inputs)
            outputs = deep_model(fm)
            _, pred = outputs.max(1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
    shallow_model.train()
    deep_model.train()
    return 100 * correct / total

def construct_model(
    model_type: string,
) -> torch.nn.Module:
    if model_type == "mlp":
        return MLP_Mnist()
    elif model_type == "vgg16":
        return VGG16_Cifar()
    elif model_type == "resnet18":
        return ResNet18_Cifar()
    elif model_type == "cnn":
        return CNN_Mnist()
    elif model_type == "mobilenet":
        return MobileNet_Mnist()
    elif model_type == "mobilenet_simple":
        return MobileNetSimple_Mnist()
    else:
        raise ValueError(f"Unrecognized model type: `{model_type}`")
