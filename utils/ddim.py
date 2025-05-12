import torch
import numpy as np
from utils.config import config
from utils.diffusion import get_diffusion_parameters

def ddim_sample(model, shape, timesteps=config['timesteps'], device=config['device'], eta=0.0, sampling_steps=50):
    """
    DDIM sampling for faster and higher quality generation
    
    Args:
        model: The diffusion model
        shape: Shape of the samples to generate
        timesteps: Total number of diffusion timesteps
        device: Device to use
        eta: Controls the stochasticity (0 = deterministic, 1 = DDPM)
        sampling_steps: Number of sampling steps (fewer = faster)
    """
    # Get diffusion parameters
    diffusion_params = get_diffusion_parameters(device=device)
    
    # Start with random noise
    x = torch.randn(shape, device=device)
    imgs = []
    
    # Create sampling timestep sequence
    skip = timesteps // sampling_steps
    seq = list(range(0, timesteps, skip))
    if seq[-1] != timesteps - 1:
        seq.append(timesteps - 1)
    
    # Reverse sequence for sampling
    seq = seq[::-1]
    
    # DDIM sampling loop
    for i in range(len(seq) - 1):
        t_current = torch.full((shape[0],), seq[i], device=device, dtype=torch.long)
        t_next = torch.full((shape[0],), seq[i+1], device=device, dtype=torch.long)
        
        # Get predicted noise
        with torch.no_grad():
            predicted_noise = model(x, t_current)
            
            # Check for NaN values and replace them
            if torch.isnan(predicted_noise).any():
                print(f"Warning: NaN values detected at timestep {seq[i]}. Replacing with zeros.")
                predicted_noise = torch.where(torch.isnan(predicted_noise), torch.zeros_like(predicted_noise), predicted_noise)
            
            # Extract alphas for current and next timestep
            alpha_cumprod_current = diffusion_params['alphas_cumprod'][seq[i]]
            alpha_cumprod_next = diffusion_params['alphas_cumprod'][seq[i+1]] if i < len(seq) - 2 else torch.tensor(1.0, device=device)
            
            # Predict x0 (clean image)
            sqrt_alpha_cumprod = torch.sqrt(alpha_cumprod_current)
            sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - alpha_cumprod_current)
            x0_pred = (x - sqrt_one_minus_alpha_cumprod * predicted_noise) / sqrt_alpha_cumprod
            
            # Clip predicted x0 for stability
            x0_pred = torch.clamp(x0_pred, -1, 1)
            
            # Direction pointing to xt
            dir_xt = torch.sqrt(1 - alpha_cumprod_current) * predicted_noise
            
            # Direction pointing to x0
            dir_x0 = torch.sqrt(alpha_cumprod_current) * x0_pred
            
            # Compute coefficient for next timestep
            c1 = torch.sqrt(alpha_cumprod_next)
            c2 = torch.sqrt(1 - alpha_cumprod_next)
            
            # Add noise if eta > 0 (stochastic sampling)
            if eta > 0:
                noise = torch.randn_like(x)
                sigma = eta * torch.sqrt((1 - alpha_cumprod_next) / (1 - alpha_cumprod_current)) * torch.sqrt(1 - alpha_cumprod_current / alpha_cumprod_next)
                x = c1 * x0_pred + c2 * predicted_noise + sigma * noise
            else:
                # Deterministic sampling
                x = c1 * x0_pred + c2 * predicted_noise
            
            imgs.append(x.cpu().numpy())
    
    return x, imgs
