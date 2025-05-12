import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import os
from datetime import datetime

from model.unet import SimpleUnet
from utils.config import config
from utils.diffusion import p_sample, forward_diffusion_sample, reverse_diffusion_sample
from utils.dataload import get_data_loaders
from utils.image_utils import normalize_image
from utils.ddim import ddim_sample


def generate_images(checkpoint_path, num_images=4, batch_size=2, show=True, seed=None, use_ddim=False, ddim_steps=50, ddim_eta=0.0):
    """
    Generate images using a trained diffusion model
    
    Args:
        checkpoint_path: Path to the model checkpoint
        num_images: Number of images to generate
        batch_size: Batch size for generation
        save_dir: Directory to save generated images
        show: Whether to display the generated images
        seed: Random seed for reproducibility
    """
    # Set random seed if provided
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    now = datetime.now()
    timestamp_str = now.strftime("%Y%m%d_%H%M%S")
    save_dir = f"./outputs/generated/{timestamp_str}"
    
    device = config['device']
    print(f"Using device: {device}")
    
    # Initialize model
    model = SimpleUnet()
    
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Create save directory if needed
    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
    
    # Generate images in batches
    remaining = num_images
    generated_images = []
    
    while remaining > 0:
        current_batch = min(batch_size, remaining)
        print(f"Generating batch of {current_batch} images...")
        
        # Generate images
        sample_shape = (current_batch, config['channels'], config['image_size'], config['image_size'])
        
        with torch.no_grad():
            if use_ddim:
                print(f"Using DDIM sampling with {ddim_steps} steps and eta={ddim_eta}")
                samples, _ = ddim_sample(model, sample_shape, timesteps=config['timesteps'], 
                                       device=device, eta=ddim_eta, sampling_steps=ddim_steps)
            else:
                samples, _ = p_sample(model, sample_shape, timesteps=config['timesteps'], device=device)
        
        # Print statistics about the generated samples
        print(f"Sample stats - min: {samples.min().item()}, max: {samples.max().item()}, mean: {samples.mean().item()}, std: {samples.std().item()}")
        
        # Convert to numpy for plotting
        samples = samples.cpu().numpy()
        generated_images.append(samples)
        
        # # Save raw samples for debugging
        # if save_dir:
        #     raw_dir = Path(save_dir) / "raw"
        #     raw_dir.mkdir(parents=True, exist_ok=True)
        #     for i in range(current_batch):
        #         raw_img = samples[i].transpose(1, 2, 0).squeeze()
        #         plt.figure(figsize=(5, 5))
        #         plt.imshow(raw_img, cmap='gray')
        #         plt.title(f"Raw Output\nMin: {raw_img.min():.4f}, Max: {raw_img.max():.4f}")
        #         plt.colorbar()
        #         plt.savefig(raw_dir / f"raw_sample_{num_images - remaining + i}.png")
        #         plt.close()
        
        # Save individual images if requested
        if save_dir:
            for i in range(current_batch):
                img_idx = num_images - remaining + i
                img = samples[i].transpose(1, 2, 0).squeeze()
                
                # Normalize to [0, 1] range with safeguards
                # img = normalize_image(img)
                
                # Save the image
                plt.figure(figsize=(5, 5))
                plt.imshow(img, cmap='gray')
                plt.axis('off')
                plt.tight_layout(pad=0)
                plt.savefig(save_path / f"sample_{img_idx}.png", bbox_inches='tight', pad_inches=0)
                plt.close()
        
        remaining -= current_batch
    
    # Concatenate all batches
    all_samples = np.concatenate(generated_images, axis=0)
    
    # Show grid of images if requested
    if show:
        # Create a grid of images
        grid_size = int(np.ceil(np.sqrt(num_images)))
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
        axes = axes.flatten()
        
        for i in range(grid_size * grid_size):
            ax = axes[i]
            ax.axis('off')
            
            if i < num_images:
                img = all_samples[i].transpose(1, 2, 0).squeeze()
                # Normalize to [0, 1] range with safeguards
                # img = normalize_image(img)
                ax.imshow(img, cmap='gray')
        
        plt.tight_layout()
        
        # Save grid if requested
        if save_dir:
            grid_path = save_path / "sample_grid.png"
            plt.savefig(grid_path)
            print(f"Saved grid to {grid_path}")
        
        plt.show()
    
    print(f"Generated {num_images} images successfully!")
    return all_samples


def generate_interpolation(checkpoint_path, steps=10, show=True, seed=None):
    """
    Generate an interpolation between two random latent vectors
    
    Args:
        checkpoint_path: Path to the model checkpoint
        steps: Number of interpolation steps
        save_dir: Directory to save generated images
        show: Whether to display the generated images
        seed: Random seed for reproducibility
    """
    # Set random seed if provided
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    now = datetime.now()
    timestamp_str = now.strftime("%Y%m%d_%H%M%S")
    save_dir = f'./outputs/interpolation/{timestamp_str}'
    
    device = config['device']
    print(f"Using device: {device}")
    
    # Initialize model
    model = SimpleUnet()
    
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Create save directory if needed
    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
    
    # Create two random latent vectors
    shape = (1, config['channels'], config['image_size'], config['image_size'])
    z1 = torch.randn(shape, device=device)
    z2 = torch.randn(shape, device=device)
    
    # Generate interpolation
    samples = []
    alphas = np.linspace(0, 1, steps)
    
    for alpha in alphas:
        # Interpolate between the two latent vectors
        z = (1 - alpha) * z1 + alpha * z2
        
        # Generate image from the interpolated latent
        with torch.no_grad():
            t = torch.full((1,), 999, device=device, dtype=torch.long)  # Start from t=999
            img = z.clone()
            
            # Perform the reverse diffusion process
            for i in reversed(range(0, config['timesteps'])):
                t.fill_(i)
                # img = model(img, t)
                img = reverse_diffusion_sample(model, img, t, device)
            
            samples.append(img.cpu().numpy())
    
    # Convert to numpy array
    samples = np.concatenate(samples, axis=0)
    
    # Show interpolation if requested
    if show:
        fig, axes = plt.subplots(1, steps, figsize=(2*steps, 2))
        
        for i, ax in enumerate(axes):
            img = samples[i].transpose(1, 2, 0).squeeze()
            # Normalize to [0, 1] range
            img = normalize_image(img)
            ax.imshow(img, cmap='gray')
            ax.axis('off')
        
        plt.tight_layout()
        
        # Save interpolation if requested
        if save_dir:
            interp_path = save_path / "interpolation.png"
            plt.savefig(interp_path)
            print(f"Saved interpolation to {interp_path}")
        
        plt.show()
    
    print(f"Generated interpolation with {steps} steps successfully!")
    return samples

def reconstruction(checkpoint_path, show=True, seed=None):
    """    
    Reconstruct original images by running the diffusion process forward and then backward
    
    Args:
        checkpoint_path: Path to the model checkpoint
        save_dir: Directory to save reconstructed images
        show: Whether to display the reconstructed images
        seed: Random seed for reproducibility
    """
    # Set random seed if provided
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    now = datetime.now()
    timestamp_str = now.strftime("%Y%m%d_%H%M%S")
    save_dir = f'./outputs/reconstruction/{timestamp_str}'
    
    device = config['device']
    print(f"Using device: {device}")
    
    # Initialize model
    model = SimpleUnet()
    
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    # Create save directory if needed
    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

    # Get test data
    _, test_loader = get_data_loaders(batch_size=1000)

    with torch.no_grad():
        data, labels = next(iter(test_loader))
        data = data.to(device)
        
        # Create a list to store one example of each digit
        selected_digits = []
        selected_data_list = []
        selected_indices = []

        # Find one example of each digit (0-9)
        for digit in range(10):
            indices = (labels == digit).nonzero(as_tuple=True)[0]
            if len(indices) == 0:
                print(f"No example of digit {digit} found in current batch")
                continue

            # Take the first example of this digit
            idx = indices[0].item()
            selected_digits.append(digit)
            selected_data_list.append(data[idx].unsqueeze(0))  # Add batch dimension back
            selected_indices.append(idx)
        
        # # Concatenate all selected digits
        # selected_data = torch.cat(selected_data_list, dim=0)
        
        # Forward diffusion process (add noise)
        noisy_images = []
        original_images = []
        reconstructed_images = []
        
        print("Running forward diffusion process...")
        for img_idx, img in enumerate(selected_data_list):
            # img = img.unsqueeze(0)  # Add batch dimension
            original_images.append(img.cpu().numpy())
            
            # Apply forward diffusion with maximum noise (t=999)
            t = torch.tensor([999], device=device)
            noisy_img, _ = forward_diffusion_sample(img, t, device)
            noisy_images.append(noisy_img.cpu().numpy())
            
            # Now run the reverse diffusion process to reconstruct
            print(f"Reconstructing digit {selected_digits[img_idx]}...")
            current_img = noisy_img.clone()

            # Perform the reverse diffusion process
            for i in reversed(range(0, config['timesteps'])):
                t = torch.tensor([i], device=device)
                current_img = reverse_diffusion_sample(model, current_img, t)
                # current_img = model(current_img, t)
            
            reconstructed_images.append(current_img.cpu().numpy())
        
        # Convert to numpy arrays
        original_images = np.concatenate(original_images, axis=0)
        noisy_images = np.concatenate(noisy_images, axis=0)
        reconstructed_images = np.concatenate(reconstructed_images, axis=0)
        
        # Show results if requested
        if show:
            num_digits = len(selected_digits)
            fig, axes = plt.subplots(3, num_digits, figsize=(2*num_digits, 6))
            
            # Plot original images
            for i in range(num_digits):
                img = original_images[i].transpose(1, 2, 0).squeeze()
                img = normalize_image(img)
                axes[0, i].imshow(img, cmap='gray')
                axes[0, i].set_title(f"Original {selected_digits[i]}")
                axes[0, i].axis('off')
            
            # Plot noisy images
            for i in range(num_digits):
                img = noisy_images[i].transpose(1, 2, 0).squeeze()
                img = normalize_image(img)
                axes[1, i].imshow(img, cmap='gray')
                axes[1, i].set_title(f"Noisy")
                axes[1, i].axis('off')
            
            # Plot reconstructed images
            for i in range(num_digits):
                img = reconstructed_images[i].transpose(1, 2, 0).squeeze()
                img = normalize_image(img)
                axes[2, i].imshow(img, cmap='gray')
                axes[2, i].set_title(f"Reconstructed")
                axes[2, i].axis('off')
            
            plt.tight_layout()
            
            # Save figure if requested
            if save_dir:
                recon_path = save_path / "reconstruction.png"
                plt.savefig(recon_path)
                print(f"Saved reconstruction visualization to {recon_path}")
            
            plt.show()
    
    print(f"Reconstructed {len(selected_digits)} digit images successfully!")
    return original_images, noisy_images, reconstructed_images


def main():
    parser = argparse.ArgumentParser(description="Generate images using a trained diffusion model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--num_images", type=int, default=4, help="Number of images to generate")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for generation")
    parser.add_argument("--save_dir", type=str, default="./outputs/generated", help="Directory to save generated images")
    parser.add_argument("--no_show", action="store_true", help="Don't display the generated images")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--interpolation", action="store_true", help="Generate interpolation between two latent vectors")
    parser.add_argument("--reconstruction", action="store_true", help="Reconstruct digit images from the test set")
    parser.add_argument("--steps", type=int, default=50, help="Number of interpolation steps")
    parser.add_argument("--use_ddim", action="store_true", help="Use DDIM sampling for higher quality generation")
    parser.add_argument("--ddim_steps", type=int, default=10, help="Number of DDIM sampling steps")
    parser.add_argument("--ddim_eta", type=float, default=0.0, help="DDIM eta parameter (0.0 = deterministic, 1.0 = DDPM)")
    
    args = parser.parse_args()
    
    if args.interpolation:
        generate_interpolation(
            checkpoint_path=args.checkpoint,
            steps=args.steps,
            show=not args.no_show,
            seed=args.seed
        )
    elif args.reconstruction:
        reconstruction(
            checkpoint_path=args.checkpoint,
            show=not args.no_show,
            seed=args.seed
        )
    else:
        generate_images(
            checkpoint_path=args.checkpoint,
            num_images=args.num_images,
            batch_size=args.batch_size,
            show=not args.no_show,
            seed=args.seed,
            use_ddim=args.use_ddim,
            ddim_steps=args.ddim_steps,
            ddim_eta=args.ddim_eta
        )


if __name__ == "__main__":
    main()
