"""
CS 552 Team 10 Final Project

Contains methods pertaining to model training

"""

import copy
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from tqdm.auto import tqdm

def train_model(model: nn.Module, dl: DataLoader, name: str, loss_fn, epochs: int, lr: float, device, args):
    print(f"[Info]: Training {name}\n")
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_weights = None
    lowest_loss = np.inf

    for e in range(0, epochs):
        epoch_loss = 0

        for images, _ in tqdm(dl):
            images = images.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                output = model(images)
                loss = loss_fn(output, images, args)

                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()

        epoch_loss = epoch_loss / len(dl.dataset)
        print("[Info]: Epoch {} Summary: Loss: {:.8f}\n".format(e + 1, epoch_loss))

        if epoch_loss < lowest_loss:
            best_weights = copy.deepcopy(model.state_dict())
            lowest_loss = epoch_loss

    return best_weights