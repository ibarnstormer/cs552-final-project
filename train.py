"""
CS 552 Team 10 Final Project

Contains methods pertaining to model training

"""
import copy
import torch.nn as nn

from torch.utils.data import DataLoader

# TODO

def train_model(model: nn.Module, dl: DataLoader, name: str, loss_fn, epochs: int, lr: float):
    print(f"[Info]: Training {name}\n")
    model.train()