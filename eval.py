"""
CS 552 Team 10 Final Project

Contains methods pertaining to model evaluation and results visualization

"""

import matplotlib.pyplot as plt
import torch

# TODO: add more methods

def visualize_reconstructions(model, test_loader, device):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        # Get one batch of data from the test loader
        data, _ = next(iter(test_loader))
        
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