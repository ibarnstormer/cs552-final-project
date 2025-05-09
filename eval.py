"""
CS 552 Team 10 Final Project

Contains methods pertaining to model evaluation and results visualization

"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torch.nn.functional as F
from sklearn.manifold import TSNE
from typing import List, Dict, Tuple, Optional, Union
from models import vq_vae, vq_vae_2, vq_vtae_2
from tqdm.auto import tqdm

def visualize_reconstructions(model, data, loss_fn, args, device):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        
        # Flattening was done before feeding data into the model,
        # so here we reshape it back into image format [batch, 3, 32, 32].

        if data.shape[-1] == 3072:
            data = data.view(data.size(0), -1).to(device)  # data is [B, 3072]

        data = data.to(device)
        
        # Pass data through the model:
        output = model(data)
        loss = F.mse_loss(output[0], data)
        _, loss_components = loss_fn(output, data, args, get_components=True)
        if isinstance(output, tuple):
            recon_batch = output[0]
        else:
            recon_batch = output
        
        # Reshape both original and reconstructed data back to images
        data = data.view(-1, 3, 32, 32).cpu()
        recon_batch = recon_batch.view(-1, 3, 32, 32).cpu()

        # Print losses
        print("[Info]: Reconstruction Loss (MSE): {:.16f}".format(loss.item() / data.shape[0]))
        for k, v in loss_components.items():
            print("[Info]: {}: {:.16f}".format(k, v / data.shape[0]))

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
        # if i == 0:
        #     ax.set_title("Original")
    
    # Plot reconstructions for each model
    for row_idx, (model_name, recon_batch) in enumerate(reconstructions.items(), start=1):
        for i in range(num_samples):
            ax = plt.subplot(num_rows, num_samples, row_idx * num_samples + i + 1)
            plt.imshow(recon_batch[i].permute(1, 2, 0).numpy())
            ax.axis('off')
            # if i == 0:
            #     ax.set_title(model_name)
    
    plt.tight_layout()
    plt.show()


def visualize_latent_space(model, model_name: str, latent_space_fn, test_loader, device):
    print(f"[Info]: Visualizing latent space for {model_name}:")

    model.eval()
    if isinstance(model, vq_vae_2.VQVAE2) or isinstance(model, vq_vtae_2.VQVTAE2):
        latent_vectors = ([], []) # id_b, id_t
    else:
        latent_vectors = []
    labels = []
    
    with torch.no_grad():
        for data, target in tqdm(test_loader):
            data = data.to(device)
            output = latent_space_fn(model, data)
            if isinstance(model, vq_vae_2.VQVAE2) or isinstance(model, vq_vtae_2.VQVTAE2):
                latent_vectors[0].append(output[0].cpu().numpy())
                latent_vectors[1].append(output[1].cpu().numpy())
                labels.append(target.numpy())
            elif isinstance(output, tuple):
                latent_vectors.append(output[0].cpu().numpy())
                labels.append(target.numpy())
            else:
                latent_vectors.append(output.cpu().numpy())
                labels.append(target.numpy())
    
    if isinstance(model, vq_vae_2.VQVAE2) or isinstance(model, vq_vtae_2.VQVTAE2):
        latent_b_vectors = np.concatenate(latent_vectors[0], axis=0)
        latent_t_vectors = np.concatenate(latent_vectors[1], axis=0)
        labels = np.concatenate(labels, axis=0)
        
        tsne = TSNE(n_components=2)
        latent_b_2d = tsne.fit_transform(latent_b_vectors)
        latent_t_2d = tsne.fit_transform(latent_t_vectors)
        
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(latent_b_2d[:, 0], latent_b_2d[:, 1], c=labels, cmap='viridis', alpha=0.7)
        plt.colorbar(scatter)
        plt.title(f"Latent Space Visualization of bottom codebook for {model_name} using t-SNE")
        plt.show()

        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(latent_t_2d[:, 0], latent_t_2d[:, 1], c=labels, cmap='viridis', alpha=0.7)
        plt.colorbar(scatter)
        plt.title(f"Latent Space Visualization of top codebook for {model_name} using t-SNE")
        plt.show()
    
    else:
        latent_vectors = np.concatenate(latent_vectors, axis=0)
        labels = np.concatenate(labels, axis=0)
        
        tsne = TSNE(n_components=2)
        latent_2d = tsne.fit_transform(latent_vectors)
        
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=labels, cmap='viridis', alpha=0.7)
        plt.colorbar(scatter)
        plt.title(f"Latent Space Visualization for {model_name} using t-SNE")
        plt.show()