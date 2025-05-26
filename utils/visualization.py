import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import imageio
from pathlib import Path
import os

from utils.config import config
from utils.diffusion import reverse_diffusion_sample
from utils.image_utils import normalize_image


def visualize_denoising_process(model, shape, timesteps=None, device=None, save_path=None, 
                               show_plot=True, create_gif=True, create_grid=True, seed=None):
    """
    Visualize the denoising process during inference.
    
    Args:
        model: The trained diffusion model
        shape: Shape of the image to generate (batch_size, channels, height, width)
        timesteps: Number of timesteps in the diffusion process (default: from config)
        device: Device to run the model on (default: from config)
        save_path: Path to save the visualization (default: None)
        show_plot: Whether to display the visualization (default: True)
        create_gif: Whether to create a GIF of the denoising process (default: True)
        create_grid: Whether to create a grid visualization of selected timesteps (default: True)
        seed: Random seed for reproducibility (default: None)
        
    Returns:
        final_img: The final denoised image
        all_images: List of all intermediate images during denoising
    """
    # Set defaults from config if not provided
    if timesteps is None:
        timesteps = config['timesteps']
    if device is None:
        device = config['device']
        
    # Set random seed if provided
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    # Create save directory if needed
    if save_path:
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
    
    # Start with random noise
    img = torch.randn(shape, device=device)
    all_images = []
    
    # Set model to evaluation mode
    model.eval()
    
    print(f"Starting denoising process with {timesteps} steps...")
    
    # Perform denoising steps
    for i in reversed(range(0, timesteps)):
        t = torch.full((shape[0],), i, device=device, dtype=torch.long)
        
        with torch.no_grad():
            # Get the denoised image at this timestep
            img = reverse_diffusion_sample(model, img, t, device)
            
            # Check for NaN values and replace them
            if torch.isnan(img).any():
                print(f"Warning: NaN values detected at timestep {i}. Replacing with zeros.")
                img = torch.where(torch.isnan(img), torch.zeros_like(img), img)
            
            # Check for infinity values and replace them
            if torch.isinf(img).any():
                print(f"Warning: Infinity values detected at timestep {i}. Replacing with ones.")
                img = torch.where(torch.isinf(img), torch.ones_like(img), img)
            
            # Clip values to a reasonable range to prevent numerical instability
            img = torch.clamp(img, -10, 10)
            
            # Save a copy of the current state
            img_np = img.cpu().detach().numpy()
            all_images.append(img_np)
            
            # Optionally save individual frames
            if save_path:
                # Only save selected frames to avoid too many files
                if i % max(1, timesteps // 100) == 0 or i == timesteps - 1:
                    for b in range(shape[0]):
                        # Normalize the image for visualization
                        norm_img = normalize_image(img_np[b])
                        
                        plt.figure(figsize=(5, 5))
                        plt.imshow(norm_img.squeeze(), cmap='gray')
                        plt.axis('off')
                        plt.title(f'Step {timesteps - i}/{timesteps}')
                        plt.savefig(f"{save_path}/sample_{b}_step_{timesteps - i:03d}.png")
                        plt.close()
    
    # Create visualization based on user preferences
    if create_grid and (show_plot or save_path):
        _create_denoising_grid(all_images, timesteps, shape[0], save_path)
    
    if create_gif and save_path:
        _create_denoising_gif(all_images, timesteps, shape[0], save_path)
    
    # Show the final result
    if show_plot:
        for b in range(shape[0]):
            plt.figure(figsize=(5, 5))
            plt.imshow(normalize_image(all_images[-1][b]).squeeze(), cmap='gray')
            plt.axis('off')
            plt.title('Final Denoised Image')
            plt.show()
    
    return img, all_images


def _create_denoising_grid(all_images, timesteps, batch_size, save_path=None):
    """
    Create a grid visualization of the denoising process.
    
    Args:
        all_images: List of all intermediate images
        timesteps: Total number of timesteps
        batch_size: Number of images in the batch
        save_path: Path to save the visualization
    """
    # Select a subset of timesteps to display
    num_steps_to_show = min(10, timesteps)
    step_indices = np.linspace(0, len(all_images) - 1, num_steps_to_show, dtype=int)
    
    for b in range(batch_size):
        fig, axes = plt.subplots(1, num_steps_to_show, figsize=(20, 3))
        
        for i, idx in enumerate(step_indices):
            if num_steps_to_show == 1:
                ax = axes
            else:
                ax = axes[i]
            
            # Get the image at this step
            img = all_images[idx][b]
            
            # Normalize for visualization
            norm_img = normalize_image(img)
            
            # Display
            ax.imshow(norm_img.squeeze(), cmap='gray')
            ax.set_title(f'Step {timesteps - (timesteps - idx - 1)}/{timesteps}')
            ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f"{save_path}/denoising_grid_sample_{b}.png", dpi=150)
            plt.close()
        else:
            plt.show()


def _create_denoising_gif(all_images, timesteps, batch_size, save_path):
    """
    Create a GIF of the denoising process.
    
    Args:
        all_images: List of all intermediate images
        timesteps: Total number of timesteps
        batch_size: Number of images in the batch
        save_path: Path to save the GIF
    """
    # For large timesteps, we may want to skip some frames to keep the GIF size reasonable
    frame_skip = max(1, len(all_images) // 50)
    
    for b in range(batch_size):
        frames = []
        
        for i in range(0, len(all_images), frame_skip):
            # Create a figure for this frame
            fig, ax = plt.subplots(figsize=(5, 5))
            
            # Get the image at this step
            img = all_images[i][b]
            
            # Normalize for visualization
            norm_img = normalize_image(img)
            
            # Display
            ax.imshow(norm_img.squeeze(), cmap='gray')
            ax.set_title(f'Step {timesteps - (timesteps - i - 1)}/{timesteps}')
            ax.axis('off')
            
            # Convert the plot to an image
            fig.canvas.draw()
            frame = np.array(fig.canvas.renderer.buffer_rgba())
            frames.append(frame)
            
            plt.close()
        
        # Save as GIF
        imageio.mimsave(f"{save_path}/denoising_process_sample_{b}.gif", frames, fps=10)
        print(f"GIF saved to {save_path}/denoising_process_sample_{b}.gif")


def visualize_interpolation(model, shape, num_steps=10, timesteps=None, device=None, 
                           save_path=None, show_plot=True, seed=None):
    """
    Visualize interpolation between two random latent vectors during the denoising process.
    
    Args:
        model: The trained diffusion model
        shape: Shape of the image to generate (batch_size, channels, height, width)
        num_steps: Number of interpolation steps
        timesteps: Number of timesteps in the diffusion process (default: from config)
        device: Device to run the model on (default: from config)
        save_path: Path to save the visualization (default: None)
        show_plot: Whether to display the visualization (default: True)
        seed: Random seed for reproducibility (default: None)
        
    Returns:
        interpolated_images: List of interpolated images
    """
    # Set defaults from config if not provided
    if timesteps is None:
        timesteps = config['timesteps']
    if device is None:
        device = config['device']
        
    # Set random seed if provided
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    # Create save directory if needed
    if save_path:
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate two random latent vectors
    z1 = torch.randn(shape, device=device)
    z2 = torch.randn(shape, device=device)
    
    # Set model to evaluation mode
    model.eval()
    
    # Store interpolated images
    interpolated_images = []
    
    # Perform interpolation
    for alpha in np.linspace(0, 1, num_steps):
        # Interpolate between the two latent vectors
        z_t = (1 - alpha) * z1 + alpha * z2
        
        # Denoise the interpolated latent
        img = z_t.clone()
        
        for i in reversed(range(0, timesteps)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            
            with torch.no_grad():
                # Get the denoised image at this timestep
                img = reverse_diffusion_sample(model, img, t, device)
                
                # Handle numerical issues
                img = torch.where(torch.isnan(img), torch.zeros_like(img), img)
                img = torch.where(torch.isinf(img), torch.ones_like(img), img)
                img = torch.clamp(img, -10, 10)
        
        # Save the final denoised image for this interpolation step
        img_np = img.cpu().detach().numpy()
        interpolated_images.append(img_np)
        
        # Optionally save individual images
        if save_path:
            for b in range(shape[0]):
                # Normalize the image for visualization
                norm_img = normalize_image(img_np[b])
                
                plt.figure(figsize=(5, 5))
                plt.imshow(norm_img.squeeze(), cmap='gray')
                plt.axis('off')
                plt.title(f'Interpolation: {alpha:.2f}')
                plt.savefig(f"{save_path}/interp_{b}_alpha_{alpha:.2f}.png")
                plt.close()
    
    # Create a grid visualization
    if show_plot or save_path:
        fig, axes = plt.subplots(1, num_steps, figsize=(20, 4))
        
        for i, img_np in enumerate(interpolated_images):
            if num_steps == 1:
                ax = axes
            else:
                ax = axes[i]
            
            # Normalize for visualization
            norm_img = normalize_image(img_np[0])
            
            # Display
            ax.imshow(norm_img.squeeze(), cmap='gray')
            ax.set_title(f'α={i/(num_steps-1):.2f}')
            ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f"{save_path}/interpolation_grid.png", dpi=150)
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    return interpolated_images
