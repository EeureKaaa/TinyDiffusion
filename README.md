# TinyDiffusion

A lightweight diffusion model for generating MNIST digit images.

## Overview

TinyDiffusion is a compact implementation of a diffusion model trained on the MNIST dataset. It demonstrates the core concepts of diffusion models while keeping the implementation simple and efficient.

## Features

- Simple U-Net architecture for noise prediction
- Cosine noise schedule
- MNIST digit generation
- Denoising process visualization with images and GIFs
- Latent space interpolation
- Weights & Biases integration for experiment tracking
- Efficient training on both CPU and CUDA-enabled GPUs

## Installation

### Prerequisites

- Python 3.12.9
- CUDA-compatible GPU (optional, but recommended for faster training)

### Setup

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/TinyDiffusion.git
   cd TinyDiffusion
   ```

2. Install dependencies:
   ```
   pip install -e .
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

## Usage

### Training

To train the diffusion model:

```
python -m src.train
```

The training script will automatically use CUDA if available, otherwise it will fall back to CPU.

### Image Generation

To generate images using a trained model:

```
python -m src.generate --checkpoint path/to/checkpoint.pth --num_images 16
```

### Visualizing the Denoising Process

To visualize the step-by-step denoising process:

```
python -m src.visualize_denoising --checkpoint path/to/checkpoint.pth
```

This will create:
- Individual images showing the denoising steps
- A grid visualization of key steps in the process
- An animated GIF of the entire denoising process

### Latent Space Interpolation

To visualize interpolation between two random points in latent space:

```
python -m src.visualize_denoising --checkpoint path/to/checkpoint.pth --interpolation --num_steps 10
```

### Configuration

Model and training parameters can be adjusted in `utils/config.py`.

Key parameters include:
- `timesteps`: Number of diffusion steps
- `epochs`: Number of training epochs
- `learning_rate`: Learning rate for the optimizer
- `batch_size`: Batch size for training

### Weights & Biases Integration

The training script integrates with Weights & Biases for experiment tracking. To use this feature:

1. Install wandb: `pip install wandb`
2. Log in to wandb: `wandb login`
3. Run the training script

## Project Structure

```
TinyDiffusion/
├── checkpoints/         # Saved model checkpoints
├── model/               # Model architecture
│   └── unet.py          # U-Net implementation
├── outputs/             # Generated images
├── results/             # Experimental results and metrics
├── src/                 # Source code
│   ├── train.py         # Training script
│   ├── generate.py      # Image generation script
│   └── visualize_denoising.py  # Denoising visualization script
├── utils/               # Utility functions
│   ├── config.py        # Configuration parameters
│   ├── data_utils.py    # Data loading utilities
│   ├── diffusion.py     # Diffusion process utilities
│   ├── ddim.py          # DDIM sampling implementation
│   ├── image_utils.py   # Image processing utilities
│   ├── fid_utils.py     # FID score calculation utilities
│   └── visualization.py # Visualization utilities
└── README.md            # This file
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This implementation is inspired by various diffusion model papers and implementations.
- The MNIST dataset is used for training and evaluation.
