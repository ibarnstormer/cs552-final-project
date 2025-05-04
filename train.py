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
    save_dir="loss_plots",
    all_model_histories=None
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
        'perplexity': [],
        'latent_loss': []
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
    
    # Plot and save individual loss history
    plot_loss_history(loss_history, name, save_dir)
    
    # Store loss history in the all_model_histories dict if provided
    if all_model_histories is not None:
        all_model_histories[name] = loss_history
    
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


def compare_model_losses(model_histories, save_dir="loss_plots"):
    """
    Create comparative plots showing loss components across different models
    
    Args:
        model_histories (dict): Dictionary mapping model names to their loss histories
        save_dir (str): Directory to save the comparative plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Get all unique loss components across all models
    all_components = set()
    for model_name, history in model_histories.items():
        all_components.update(history.keys())
    
    # Create a figure for each loss component
    for component in all_components:
        plt.figure(figsize=(14, 8))
        
        # Plot this component for each model that has it
        for model_name, history in model_histories.items():
            if component in history and len(history[component]) > 0:
                plt.plot(history[component], label=f"{model_name}")
        
        plt.title(f'Comparison of {component.replace("_", " ").title()} Across Models')
        plt.xlabel('Epoch')
        plt.ylabel(f'{component.replace("_", " ").title()}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save the comparative plot
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"comparison_{component}.png"))
        plt.close()
    
    # Create a figure for total loss comparison
    plt.figure(figsize=(14, 8))
    for model_name, history in model_histories.items():
        if 'total_loss' in history and len(history['total_loss']) > 0:
            plt.plot(history['total_loss'], label=f"{model_name}")
    
    plt.title('Comparison of Total Loss Across Models')
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "comparison_total_loss.png"))
    plt.close()
    
    print(f"[Info]: Comparative loss plots saved to {save_dir}")


def load_or_create_model_histories(trained_models, save_dir="loss_plots"):
    """
    Load existing loss histories or create an empty dictionary for tracking
    
    Args:
        trained_models (list): List of model names
        save_dir (str): Directory where loss histories are saved
        
    Returns:
        dict: Dictionary of model loss histories
    """
    model_histories = {}
    
    # Try to load existing loss histories
    for model_name in trained_models:
        safe_name = model_name.replace(' ', '_').lower()
        history_path = os.path.join(save_dir, f"{safe_name}_loss_history.npy")
        
        if os.path.exists(history_path):
            try:
                # Load with allow_pickle because we saved a dictionary
                history = np.load(history_path, allow_pickle=True).item()
                model_histories[model_name] = history
                print(f"[Info]: Loaded existing loss history for {model_name}")
            except Exception as e:
                print(f"[Warning]: Could not load history for {model_name}: {e}")
                model_histories[model_name] = {}
        else:
            model_histories[model_name] = {}
    
    return model_histories