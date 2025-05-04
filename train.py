"""
CS 552 Team 10 Final Project

Contains methods pertaining to model training

"""

import copy
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import matplotlib.pyplot as plt


def train_model(
    model: nn.Module,
    dl: DataLoader,
    name: str,
    loss_fn,
    epochs: int,
    lr: float,
    device,
    args,
    save_dir="loss_plots"
):
    print(f"[Info]: Training {name}\n")
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_weights = None
    lowest_loss = np.inf
    
    # Dictionary to store loss history
    loss_history = {
        'total_loss': [],
        'reconstruction_loss': [],
        'kl_divergence': [],
        'commitment_loss': [],
        'perplexity': []
    }

    for e in range(0, epochs):
        epoch_loss = 0
        # Track individual loss components for this epoch
        epoch_loss_components = defaultdict(float)
        
        for images, _ in tqdm(dl):
            images = images.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                output = model(images)
                
                # Get individual loss components
                loss, loss_components = loss_fn(output, images, args, get_components=True)

                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()
            
            # Accumulate loss components
            for key, value in loss_components.items():
                if isinstance(value, torch.Tensor):
                    epoch_loss_components[key] += value.item()
                else:
                    epoch_loss_components[key] += value

        # Calculate average loss for the epoch
        epoch_loss = epoch_loss / len(dl.dataset)
        
        # Calculate average for each loss component
        for key in epoch_loss_components:
            epoch_loss_components[key] /= len(dl)
            loss_history[key].append(epoch_loss_components[key])
        
        # Ensure total loss is recorded even if not returned as a component
        loss_history['total_loss'].append(epoch_loss)
        
        # Print epoch summary with all loss components
        print(f"[Info]: Epoch {e + 1} Summary: Total Loss: {epoch_loss:.8f}")
        for key, value in epoch_loss_components.items():
            print(f"    - {key}: {value:.8f}")
        print()

        if epoch_loss < lowest_loss:
            best_weights = copy.deepcopy(model.state_dict())
            lowest_loss = epoch_loss
    
    # Plot and save loss history
    plot_loss_history(loss_history, name, save_dir)
    
    return best_weights


def plot_loss_history(loss_history, model_name, save_dir="loss_plots"):
    """
    Plot the loss history for a model and save it to disk
    
    Args:
        loss_history (dict): Dictionary of loss histories
        model_name (str): Name of the model
        save_dir (str): Directory to save the plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 8))
    
    # Plot each component
    for loss_type, values in loss_history.items():
        if values:  # Only plot if we have values
            plt.plot(values, label=loss_type)
    
    plt.title(f'Loss History for {model_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{model_name.replace(' ', '_').lower()}_loss.png"))
    plt.close()
    
    # Save the raw data for future reference
    np.save(os.path.join(save_dir, f"{model_name.replace(' ', '_').lower()}_loss_history.npy"), loss_history)
    
    print(f"[Info]: Loss history saved for {model_name}")