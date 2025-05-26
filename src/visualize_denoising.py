import torch
import argparse
import os
from pathlib import Path
from datetime import datetime

from model.unet import SimpleUnet
from utils.config import config
from utils.visualization import visualize_denoising_process, visualize_interpolation


def visualize_generation(checkpoint_path, batch_size=1, timesteps=None, 
                         save_dir=None, show=True, create_gif=True, create_grid=True, seed=None):
    """
    Visualize the denoising process of image generation
    
    Args:
        checkpoint_path: Path to the model checkpoint
        batch_size: Number of images to generate
        timesteps: Number of diffusion timesteps (default: from config)
        save_dir: Directory to save visualizations
        show: Whether to display the visualizations
        create_gif: Whether to create a GIF of the denoising process
        create_grid: Whether to create a grid visualization
        seed: Random seed for reproducibility
    """
    # Set random seed if provided
    if seed is not None:
        torch.manual_seed(seed)
        
    # Create timestamp for save directory
    if save_dir is None:
        now = datetime.now()
        timestamp_str = now.strftime("%Y%m%d_%H%M%S")
        save_dir = f"{config['generated_images_path']}/denoising_viz_{timestamp_str}"
    
    # Get device
    device = config['device']
    print(f"Using device: {device}")
    
    # Initialize model
    model = SimpleUnet()
    
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Set shape for generation
    sample_shape = (batch_size, config['channels'], config['image_size'], config['image_size'])
    
    # Visualize the denoising process
    print(f"Visualizing denoising process for {batch_size} images...")
    final_img, all_images = visualize_denoising_process(
        model=model,
        shape=sample_shape,
        timesteps=timesteps,
        device=device,
        save_path=save_dir,
        show_plot=show,
        create_gif=create_gif,
        create_grid=create_grid,
        seed=seed
    )
    
    print(f"Visualization complete. Results saved to: {save_dir}")
    return final_img, all_images


def visualize_latent_interpolation(checkpoint_path, num_steps=10, batch_size=1, 
                                  timesteps=None, save_dir=None, show=True, seed=None):
    """
    Visualize interpolation between two random latent vectors
    
    Args:
        checkpoint_path: Path to the model checkpoint
        num_steps: Number of interpolation steps
        batch_size: Number of images to generate
        timesteps: Number of diffusion timesteps (default: from config)
        save_dir: Directory to save visualizations
        show: Whether to display the visualizations
        seed: Random seed for reproducibility
    """
    # Set random seed if provided
    if seed is not None:
        torch.manual_seed(seed)
        
    # Create timestamp for save directory
    if save_dir is None:
        now = datetime.now()
        timestamp_str = now.strftime("%Y%m%d_%H%M%S")
        save_dir = f"{config['generated_images_path']}/interpolation_viz_{timestamp_str}"
    
    # Get device
    device = config['device']
    print(f"Using device: {device}")
    
    # Initialize model
    model = SimpleUnet()
    
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Set shape for generation
    sample_shape = (batch_size, config['channels'], config['image_size'], config['image_size'])
    
    # Visualize the interpolation
    print(f"Visualizing latent interpolation with {num_steps} steps...")
    interpolated_images = visualize_interpolation(
        model=model,
        shape=sample_shape,
        num_steps=num_steps,
        timesteps=timesteps,
        device=device,
        save_path=save_dir,
        show_plot=show,
        seed=seed
    )
    
    print(f"Interpolation visualization complete. Results saved to: {save_dir}")
    return interpolated_images


def main():
    parser = argparse.ArgumentParser(description="Visualize the denoising process of diffusion models")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--batch_size", type=int, default=1, help="Number of images to generate")
    parser.add_argument("--timesteps", type=int, default=None, help="Number of diffusion timesteps (default: from config)")
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to save visualizations")
    parser.add_argument("--no_show", action="store_true", help="Don't display the visualizations")
    parser.add_argument("--no_gif", action="store_true", help="Don't create GIFs")
    parser.add_argument("--no_grid", action="store_true", help="Don't create grid visualizations")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--interpolation", action="store_true", help="Visualize latent interpolation")
    parser.add_argument("--num_steps", type=int, default=10, help="Number of interpolation steps")
    
    args = parser.parse_args()
    
    # Create save directory if provided
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
    
    if args.interpolation:
        visualize_latent_interpolation(
            checkpoint_path=args.checkpoint,
            num_steps=args.num_steps,
            batch_size=args.batch_size,
            timesteps=args.timesteps,
            save_dir=args.save_dir,
            show=not args.no_show,
            seed=args.seed
        )
    else:
        visualize_generation(
            checkpoint_path=args.checkpoint,
            batch_size=args.batch_size,
            timesteps=args.timesteps,
            save_dir=args.save_dir,
            show=not args.no_show,
            create_gif=not args.no_gif,
            create_grid=not args.no_grid,
            seed=args.seed
        )


if __name__ == "__main__":
    main()
