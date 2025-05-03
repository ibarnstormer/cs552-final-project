"""
CS 552 Team 10 Final Project

Contains methods pertaining to model evaluation and results visualization

"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision

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
        output = model(data)
        if isinstance(output, tuple):
            recon_batch = output[0]
        else:
            recon_batch = output
        
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


