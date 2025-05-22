# TinyDiffusion

A lightweight diffusion model for generating MNIST digit images.

## Overview

TinyDiffusion is a compact implementation of a diffusion model trained on the MNIST dataset. It demonstrates the core concepts of diffusion models while keeping the implementation simple and efficient.

## Features

- Simple U-Net architecture for noise prediction
- Cosine noise schedule
- MNIST digit generation
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
├── src/                 # Source code
│   └── train.py         # Training script
├── utils/               # Utility functions
│   ├── config.py        # Configuration parameters
│   ├── data_utils.py    # Data loading utilities
│   └── diffusion.py     # Diffusion process utilities
└── README.md            # This file
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This implementation is inspired by various diffusion model papers and implementations.
- The MNIST dataset is used for training and evaluation.
