# Giving Attention to VQ-VAEs

## Introduction

The repository contains code for the class project "Giving Attention to VQ-VAEs" for CS 552 Generative AI. 

We propose two model architectures, VQ-VTAE and VQ-VTAE-2, which incorporate attention mechanisms (CBAM) into the encoder-decoder components of a VQ-VAE. We also conduct several experiments to evaluate its performance in image reconstruction and to compare this “attention-enhanced” VQ-VAE to other traditional VAE and VQ-VAE architectures.

## Requirements

- torch
- torchvision
- numpy
- matplotlib
- tqdm

## Run

To run the code, you can use the following command:

```bash
python main.py -e <num_epochs> -lr <learning_rate> -b <batch_size> -o <output_dir>
```

where:
- `<num_epochs>`: Number of epochs to train the model (default: 10)
- `<learning_rate>`: Learning rate for the optimizer (default: 0.001)
- `<batch_size>`: Batch size for training (default: 128)
- `<output_dir>`: Directory to save the model weights and figures (default: './trained_model_weights/')


This will train the model for the specified number of epochs with the given learning rate and batch size. The model weights and figures will be saved in the specified directory.
