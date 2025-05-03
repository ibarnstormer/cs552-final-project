"""
CS 552 Team 10 Final Project

Contains methods pertaining to model evaluation and results visualization

"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from typing import List, Dict, Tuple, Optional, Union

from models import vq_vae

# TODO: add more methods

def visualize_reconstructions(model, data, device):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        
        # Flattening was done before feeding data into the model,
        # so here we reshape it back into image format [batch, 3, 32, 32].

        if data.shape[-1] == 3072:
            data = data.view(data.size(0), -1).to(device)  # data is [B, 3072]

        data = data.to(device)
        
        # Pass data through the model:
        recon_batch, _, _ = model(data)
        
        # Reshape both original and reconstructed data back to images
        data = data.view(-1, 3, 32, 32).cpu()
        recon_batch = recon_batch.view(-1, 3, 32, 32).cpu()

        n = 8  # Number of images to visualize
        plt.figure(figsize=(16, 4))
        for i in range(n):
            # Plot original images
            ax = plt.subplot(2, n, i + 1)
            # Convert image from [C, H, W] to [H, W, C]
            plt.imshow(data[i].permute(1, 2, 0).numpy())
            ax.axis('off')
            if i == 0:
                ax.set_title("Original")
            
            # Plot reconstructed images
            ax = plt.subplot(2, n, i + 1 + n)
            plt.imshow(recon_batch[i].permute(1, 2, 0).numpy())
            ax.axis('off')
            if i == 0:
                ax.set_title("Reconstruction")
        plt.show()

def visualize_tensor(t: torch.Tensor, plot_title: str = "", nrow: int = None):

    grid = torchvision.utils.make_grid(t, nrow=int(np.ceil(t.shape[0] ** 0.5)) if nrow == None else nrow)

    plt.imshow(grid.permute(1, 2, 0))
    if plot_title != "":
        plt.title(plot_title)
    plt.show()



def compare_model_reconstructions(models_dict: Dict[str, torch.nn.Module], 
                                 test_data: torch.Tensor, 
                                 device: torch.device,
                                 num_samples: int = 8,
                                 figsize: Tuple[int, int] = None):
    """
    Compare reconstructions from multiple models using the same test data.
    
    Args:
        models_dict: Dictionary mapping model names to model instances
        test_data: Test data tensor [B, C, H, W] or [B, D] if flattened
        device: Device to run the models on
        num_samples: Number of samples to visualize
        figsize: Figure size for the plot
    """
    # Number of rows will be the number of models + 1 (for original)
    num_rows = len(models_dict) + 1
    
    # Set a reasonable figure size if not provided
    if figsize is None:
        figsize = (16, 3 * num_rows)
    
    # Copy and prepare test data
    if test_data.shape[-1] == 3072:  # If flattened
        data = test_data[:num_samples].view(num_samples, -1).to(device)
        original_images = test_data[:num_samples].view(num_samples, 3, 32, 32)
    else:  # If already in image format
        data = test_data[:num_samples].to(device)
        original_images = test_data[:num_samples]
    
    # Set all models to evaluation mode and collect reconstructions
    reconstructions = {}
    with torch.no_grad():
        for model_name, model in models_dict.items():
            model.eval()
            
            # Handle different model types that might have different output formats
            if isinstance(model, vq_vae.VQVAE):
                recon, _, _ = model(data)
            else:
                # Adapt this to handle other model types as needed
                try:
                    recon = model(data)
                    # Check if the output is a tuple and take the first element
                    if isinstance(recon, tuple):
                        recon = recon[0]
                except Exception as e:
                    print(f"Error running model {model_name}: {e}")
                    continue
            
            # Reshape reconstruction to image format
            if recon.shape[-1] != 32:  # If flattened
                recon = recon.view(-1, 3, 32, 32)
            
            reconstructions[model_name] = recon.cpu()
    
    # Plot the results
    plt.figure(figsize=figsize)
    
    # Plot original images in the first row
    for i in range(num_samples):
        ax = plt.subplot(num_rows, num_samples, i + 1)
        plt.imshow(original_images[i].cpu().permute(1, 2, 0).numpy())
        ax.axis('off')
        if i == 0:
            ax.set_title("Original")
    
    # Plot reconstructions for each model
    for row_idx, (model_name, recon_batch) in enumerate(reconstructions.items(), start=1):
        for i in range(num_samples):
            ax = plt.subplot(num_rows, num_samples, row_idx * num_samples + i + 1)
            plt.imshow(recon_batch[i].permute(1, 2, 0).numpy())
            ax.axis('off')
            if i == 0:
                ax.set_title(model_name)
    
    plt.tight_layout()
    plt.show()