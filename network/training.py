import torch
from torch.utils.data import DataLoader
from typing import Callable, Tuple
import torch.nn as nn
import numpy as np
from tqdm import tqdm

def to(data, device):
    if isinstance(data, dict):
      return {key: to(data[key], device) for key in data.keys()}
    elif isinstance(data, list):
      return [to(d, device) for d in data]
    elif isinstance(data, tuple):
      return tuple([to(d, device) for d in data])
    else:
      return data.to(device)

def remove(data):
    if isinstance(data, dict):
      for key in data.keys():
        remove(data[key])
    elif isinstance(data, list) or isinstance(data, tuple):
      for d in data:
        remove(d)
    del data

def train_one_epoch(model:nn.Module, train_dataloader:DataLoader, optimizer:torch.optim.Optimizer, loss:Callable, device:torch.device):
    for _, (inputs, labels) in enumerate(train_dataloader):
        inputs = to(inputs, device)
        labels = to(labels, device)

        optimizer.zero_grad()

        outputs = model(inputs)
        l = loss(outputs, labels)
        l.backward()
        optimizer.step()

        remove(inputs)
        remove(labels)

def compute_total_loss_and_predictions(model:nn.Module, 
                                       dataloader:DataLoader, 
                                       loss:Callable, 
                                       device:torch.device):
    total_loss = 0
    total_elements = len(dataloader)
    total_predictions = []
    total_labels = []
    with torch.no_grad():
        for _, data in enumerate(dataloader):
            inputs, labels = data

            inputs = to(inputs, device)
            labels = to(labels, device)

            outputs = model(inputs)
            total_loss += loss(outputs, labels).item()
            predictions = nn.functional.sigmoid(outputs).round()
            total_predictions += predictions.tolist()
            assert isinstance(labels, torch.Tensor)
            total_labels += labels.tolist()

            remove(inputs)
            remove(labels)

    return total_loss / total_elements, (total_predictions, total_labels)

def train(model:nn.Module, 
          train_dataset:DataLoader, 
          validation_dataset:DataLoader, 
          optimizer:torch.optim.Optimizer, 
          loss:Callable,
          epochs: int,
          patience:int,
          device:torch.device,
          model_class,
          hyperparam:dict,
          shared_tqdm:tqdm,
          temporary_save:str|None=None) -> Tuple[nn.Module, dict, float]:

    model = model.to(device)
    data = {"train": {"loss":[]}, "validation": {"loss":[]}}

    best_model = model_class(**hyperparam)
    best_loss = np.inf
    current_patience = 0

    for epoch in range(epochs):
        model.train()
        train_one_epoch(model, train_dataset, optimizer, loss, device)
        model.eval()
        train_loss, (_, _) = compute_total_loss_and_predictions(model, train_dataset, loss, device)
        validation_loss, (_, _) = compute_total_loss_and_predictions(model, validation_dataset, loss, device)

        data["train"]["loss"].append(train_loss)

        data["validation"]["loss"].append(validation_loss)

        if validation_loss < best_loss:
            best_loss = validation_loss
            best_model.load_state_dict(model.state_dict())
            current_patience = 0
        else:
            current_patience += 1
        if current_patience >= patience:
            shared_tqdm.update((epochs - epoch))
            break
        shared_tqdm.update(1)
        if temporary_save is not None:
            torch.save(model.state_dict(), temporary_save)
    
    return best_model, data, best_loss
