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
from utils.dataload import get_data_loaders, ensure_directory_exists
from utils.image_utils import normalize_image
from utils.ddim import ddim_sample
from torch_fidelity import calculate_metrics



def generate_images(checkpoint_path, num_images=4, batch_size=2, show=True, seed=None, use_ddim=False, ddim_steps=50, ddim_eta=0.0):
    """
    Generate images using a trained diffusion model
    
    Args:
        checkpoint_path: Path to the model checkpoint
        num_images: Number of images to generate
        batch_size: Batch size for generation
        show: Whether to display the generated images
        seed: Random seed for reproducibility
        use_ddim: Whether to use DDIM sampling
        ddim_steps: Number of DDIM sampling steps
        ddim_eta: DDIM eta parameter
    """
    # Set random seed if provided
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    now = datetime.now()
    timestamp_str = now.strftime("%Y%m%d_%H%M%S")
    save_dir = f"{config['generated_images_path']}/{timestamp_str}"
    
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
        
        # Process samples like in save_real_images
        import torch.nn.functional as F
        
        # Convert to CPU tensor first
        samples = samples.cpu()
        
        # Convert to 3 channels and resize to 299x299 (FID standard size)
        samples = samples.repeat(1, 3, 1, 1)  # [N, 1, H, W] -> [N, 3, H, W]
        samples = F.interpolate(samples, size=(299, 299), mode='bilinear')  # Resize to 299x299
        
        # Normalize to [0, 1]
        samples = (samples - samples.min()) / (samples.max() - samples.min())
        
        # Convert to numpy and store for grid display
        samples_np = samples.numpy()
        generated_images.append(samples_np)
        
        # Save individual images if requested
        if save_dir:
            for i in range(current_batch):
                img_idx = num_images - remaining + i
                
                # Convert to numpy for saving (CHW -> HWC)
                img_np = samples_np[i].transpose(1, 2, 0)
                
                # Save the image
                plt.figure(figsize=(5, 5))
                plt.imshow(img_np)  # No cmap for RGB images
                plt.axis('off')
                plt.savefig(save_path / f"sample_{img_idx:04d}.png", bbox_inches='tight', pad_inches=0)
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
                # Use the processed RGB images
                img = all_samples[i].transpose(1, 2, 0)  # CHW -> HWC
                ax.imshow(img)  # No cmap for RGB images
        
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


def save_real_images(output_dir, num_images=64, batch_size=32, seed=None):
    """
    Save real images from the test dataloader for FID evaluation.
    
    Args:
        output_dir: Directory to save the real images
        num_images: Number of images to save
        batch_size: Batch size for loading data
        seed: Random seed for reproducibility
    """
    # Set random seed if provided
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    # Create output directory
    save_path = Path(output_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Get test dataloader
    _, test_loader = get_data_loaders(batch_size=batch_size)
    
    # Save images
    print(f"Saving {num_images} real images to {output_dir}...")
    
    import torch.nn.functional as F
    
    img_count = 0
    for batch, _ in test_loader:
        # Process entire batch at once for efficiency
        # Convert to 3 channels and resize
        images = batch.repeat(1, 3, 1, 1)  # [N, 1, H, W] -> [N, 3, H, W]
        images = F.interpolate(images, size=(299, 299), mode='bilinear')  # Resize to 512x512
        
        # Normalize to [0, 1]
        images = (images - images.min()) / (images.max() - images.min())
        
        for img in images:
            if img_count >= num_images:
                break
                
            # Convert to numpy for saving
            img_np = img.numpy().transpose(1, 2, 0)
            
            # Save the image
            plt.figure(figsize=(5, 5))
            plt.imshow(img_np)
            plt.axis('off')
            plt.savefig(save_path / f"real_{img_count:04d}.png", bbox_inches='tight', pad_inches=0)
            plt.close()
            
            img_count += 1
            
            if img_count >= num_images:
                break
    
    print(f"Saved {img_count} real images to {output_dir}")

def evaluate_fid(real_dir, generated_dir, checkpoint_path=None, num_images=64, batch_size=4, seed=None, use_ddim=False, ddim_steps=50, ddim_eta=0.0):
    # Calculate FID
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print("Calculating FID score using GPU...")
    else:
        print("Calculating FID score using CPU...")
        
    metrics = calculate_metrics(
        input1=real_dir,
        input2=generated_dir,
        cuda=use_cuda,
        fid=True,
        verbose=True
    )
    
    # The key is 'frechet_inception_distance' not 'fid'
    fid_score = metrics['frechet_inception_distance']
    print(f"FID Score: {fid_score:.4f}")

    # # store json for fid score
    # if fid_score:
    #     timestamp = time.strftime("%Y%m%d-%H%M%S")
    #     fid_score = {
    #         'model_path': args.model_path,
    #         'num_images': num_images,
    #         'fid_score': fid_score,
    #         'timestamp': timestamp
    #     }
    #     with open(config['fid_score_path'] + f'/fid_score_{timestamp}.json', 'w') as f:
    #         json.dump(fid_score, f, indent=4)
    
    return fid_score


def main():
    parser = argparse.ArgumentParser(description="Generate images using a trained diffusion model")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to the model checkpoint")
    parser.add_argument("--num_images", type=int, default=500, help="Number of images to generate")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for generation")
    parser.add_argument("--save_dir", type=str, default="./outputs/generated", help="Directory to save generated images")
    parser.add_argument("--no_show", action="store_true", help="Don't display the generated images")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--interpolation", action="store_true", help="Generate interpolation between two latent vectors")
    parser.add_argument("--steps", type=int, default=50, help="Number of interpolation steps")
    parser.add_argument("--use_ddim", action="store_true", help="Use DDIM sampling for higher quality generation")
    parser.add_argument("--ddim_steps", type=int, default=10, help="Number of DDIM sampling steps")
    parser.add_argument("--ddim_eta", type=float, default=0.0, help="DDIM eta parameter (0.0 = deterministic, 1.0 = DDPM)")
    parser.add_argument("--evaluate_fid", action="store_true", help="Evaluate generated images using FID")
    parser.add_argument("--real_dir", type=str, help="Directory containing real images for FID evaluation")
    parser.add_argument("--generated_dir", type=str, help="Directory containing generated images for FID evaluation")
    parser.add_argument("--save_real", action="store_true", help="Save real images from test dataset")
    parser.add_argument("--real_images", type=int, default=500, help="Number of real images to save")
    
    args = parser.parse_args()
    
    if args.save_real:
        # Create a timestamped directory for real images
        real_dir = f"./outputs/real_images"
        
        # Save real images
        save_real_images(
            output_dir=real_dir,
            num_images=args.real_images,
            batch_size=args.batch_size,
            seed=args.seed
        )
        
        print(f"Real images saved to: {real_dir}")
        print(f"You can use this directory for FID evaluation with: --real_dir {real_dir}")
        
    elif args.evaluate_fid:
        if args.real_dir is None:
            parser.error("--real_dir is required when using --evaluate_fid")
        
        evaluate_fid(
            real_dir=args.real_dir,
            generated_dir=args.generated_dir,
            checkpoint_path=args.checkpoint if args.generated_dir is None else None,
            num_images=args.real_images,
            batch_size=args.batch_size,
            seed=args.seed,
            use_ddim=args.use_ddim,
            ddim_steps=args.ddim_steps,
            ddim_eta=args.ddim_eta
        )
    elif args.interpolation:
        generate_interpolation(
            checkpoint_path=args.checkpoint,
            steps=args.steps,
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
