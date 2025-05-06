"""
CS 552 Team 10 Final Project

Main entrypoint script for project

"""

import argparse
import os
import random

import numpy as np
import scipy.stats as stats
import torch.cuda
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from tqdm.auto import tqdm

from eval import *
from models import cvae, vae, vq_vae, vq_vae_2, vq_vtae, vtae, vq_vtae_2
from train import *

""" ------ Immutable Globals ------ """

abs_path = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
seed = 42
latent_dim = 512

# Model name, Model Weights Name, Model Class
models = [
    ("Vanilla VAE", "vae", vae.VAE),
    ("Convolutional VAE", "cvae", cvae.CVAE),
    #("VTAE", "vtae", vtae.VTAE), # This cause gradient explosion
    ("VQ-VAE", "vq-vae", vq_vae.VQVAE),
    ("VQ-VAE-2", "vq-vae-2", vq_vae_2.VQVAE2),
    ("VQ-VTAE", "vq-vtae", vq_vtae.VQVTAE),
    ("VQ-VTAE-2", "vq-vtae-2", vq_vtae_2.VQVTAE2)
]

model_cstr_args = {
    "Vanilla VAE": {
        "latent_dim": latent_dim,
        "input_dim": 32 * 32 * 3,
        "hidden_dim": 1024,
        "loss_fn": vae.VAE.vae_loss,
        "latent_viz_fn": vae.VAE.encoder
    },
    "Convolutional VAE": {
        "device": device,
        "latent_dim": latent_dim,
        "input_dim": 32 * 32 * 3,
        "input_channels": 3,
        "loss_fn": cvae.CVAE.vae_loss,
        "latent_viz_fn": cvae.CVAE.encode
    },
    "VTAE": {
        "input_shape": (3, 32, 32),
        "latent_dim": latent_dim,
        "outputdensity": "gaussian",
        "ST_type": "affine",
        "loss_fn": vtae.VTAE.vtae_loss,
        "latent_viz_fn": None # VTAE loss is unstable, leave blank for now and prioritize other models
    },
    "VQ-VAE": {
        "in_channels": 3,
        "hidden_dim": 128,
        "embedding_dim": 64,
        "num_embeddings": latent_dim,
        "commitment_cost": 0.25,
        "loss_fn": vq_vae.VQVAE.vqvae_loss,
        "latent_viz_fn": vq_vae.VQVAE.encode
    },
    "VQ-VAE-2": {
        "in_channel": 3,
        "channel": 128,  # hidden channels
        "n_res_block": 2,
        "n_res_channel": 32,
        "embed_dim": 64,
        "n_embed": latent_dim,  # Number of embeddings
        "decay": 0.8,
        "loss_fn": vq_vae_2.VQVAE2.vq_vae2_loss,
        "latent_viz_fn": vq_vae_2.VQVAE2.encoding_indices
    },
    "VQ-VTAE": {
        "in_channels": 3,
        "hidden_dim": 128,
        "embedding_dim": 64,
        "num_embeddings": latent_dim,
        "commitment_cost": 0.25,
        "loss_fn": vq_vtae.VQVTAE.vqvtae_loss,
        "latent_viz_fn": vq_vtae.VQVTAE.encode
    },
    "VQ-VTAE-2": {
        "in_channel": 3,
        "channel": 128,  # hidden channels
        "n_res_block": 2,
        "n_res_channel": 32,
        "embed_dim": 64,
        "n_embed": latent_dim,  # Number of embeddings
        "decay": 0.8,
        "loss_fn": vq_vtae_2.VQVTAE2.vq_vtae2_loss,
        "latent_viz_fn": vq_vtae_2.VQVTAE2.encoding_indices
    }
}

""" ------ Argparser Arguments ------ """

argParser = argparse.ArgumentParser()

argParser.add_argument("-e", "--epochs", type=int, default=10, help="Number of epochs")
argParser.add_argument(
    "-lr", "--learning_rate", type=float, default=0.001, help="Learning Rate"
)
argParser.add_argument(
    "-b", "--batch_size", type=int, default=128, help="Batch Size"
)  # 128
argParser.add_argument(
    "-o",
    "--output",
    type=str,
    default=os.path.join(abs_path, "trained_model_weights"),
    help="Output directory for model weights",
)
argParser.add_argument(
    "-pm",
    "--use_pretrained_models",
    type=bool,
    default=True,
    help="Flag for using pre-trained models (skip training for any model that already has weights)",
)

args = argParser.parse_args()

epochs = args.epochs
lr = args.learning_rate
batch_size = args.batch_size
output_dir = args.output
use_pretrained = args.use_pretrained_models

""" ------ Datasets ------ """

# CIFAR10 (3 x 32 x 32)

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

train_ds = torchvision.datasets.CIFAR10(
    root=os.path.join(abs_path, "data"), download=True, transform=transform
)
test_ds = torchvision.datasets.CIFAR10(
    root=os.path.join(abs_path, "data"), train=False, transform=transform
)

train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=True)


""" ------ Methods ------ """


def setup():
    """
    Set up the application environment
    """
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(seed)
        cuda_info = "Cuda modules loaded."
    else:
        cuda_info = "Cuda modules not loaded."

    print("[Info]: " + cuda_info + "\n")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    # Setup Environment
    setup()

    # Initialize, train, and evaluate models
    test_img, _ = next(iter(test_dl))

    # Dictionary to store all models for comparison
    trained_models = {}
    
    # Dictionary to store all model loss histories
    all_model_histories = {}
    
    # Load any existing loss histories
    model_names = [model_name for model_name, _, _ in models]
    all_model_histories = load_or_create_model_histories(model_names, save_dir=os.path.join(output_dir, "loss_plots"))

    for model_name, model_weights_name, model_cstr in models:
        # Train model / load model weights

        model = model_cstr(**model_cstr_args[model_name])
        model.to(device)

        pretrained_available = False

        if use_pretrained:
            try:
                model_weights = torch.load(
                    os.path.join(output_dir, f"{model_weights_name}.pt"),
                    map_location=device,
                    weights_only=True,
                )
                model.load_state_dict(model_weights)
                pretrained_available = True
            except:
                print(f"[Error]: Could not load weights for model: {model_name}")

        if not pretrained_available:
            model_weights = train_model(
                model,
                train_dl,
                model_name,
                model_cstr_args[model_name]["loss_fn"],
                epochs,
                lr,
                device,
                model_cstr_args[model_name],
                save_dir=os.path.join(output_dir, "loss_plots"),
                all_model_histories=all_model_histories
            )
            os.makedirs(output_dir, exist_ok=True)  # Create output directory
            torch.save(
                model_weights, os.path.join(output_dir, f"{model_weights_name}.pt")
            )
            model.load_state_dict(model_weights)

        # Model Evaluation
        print(f"[Info]: Evaluating {model_name}")
        visualize_reconstructions(model, test_img, device)

        # Latent space visualization
        sub_dl = DataLoader(Subset(test_ds, torch.randperm(len(test_ds))[:5000]), batch_size=1, shuffle=True)

        visualize_latent_space(model, model_name, model_cstr_args[model_name]["latent_viz_fn"], sub_dl, device)
        
        # Store model for comparison
        trained_models[model_name] = model

    # After all models are trained/loaded, perform comparisons
    print("\n[Info]: Performing cross-model comparisons")
    
    # Generate comparative loss plots
    print("[Info]: Generating comparative loss plots")
    compare_model_losses(
        all_model_histories, 
        save_dir=os.path.join(output_dir, "loss_plots")
    )
    
    # Compare reconstructions from all models
    print("[Info]: Comparing image reconstructions across models")
    compare_model_reconstructions(
        models_dict=trained_models,
        test_data=test_img,
        device=device,
        num_samples=8
    )

    pass


if __name__ == "__main__":
    main()
