import torch
import torch.nn.functional as F
import numpy as np
from utils.config import config

def linear_beta_schedule(timesteps, beta_start, beta_end):
    """
    Linear schedule for beta values from beta_start to beta_end
    """
    return torch.linspace(beta_start, beta_end, timesteps)

def cosine_beta_schedule(timesteps, s=0.008):
    """
    Cosine schedule for beta values from beta_start to beta_end
    """
    
    def f(t, s, timesteps):
        return (torch.cos((t + s) / (timesteps + s) * np.pi / 2)) ** 2 
    
    t = torch.linspace(0, timesteps, timesteps+1)
    alphas_bar = f(t, s, timesteps) / f(torch.tensor(0.0), s, timesteps)
    alphas_bar_prev = torch.cat([torch.tensor([1.0]), alphas_bar[:-1]], dim=0)

    return (1 - alphas_bar / alphas_bar_prev)  

def get_index_from_list(vals, t, x_shape):
    batch_size = t.shape[0]
    out = vals.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

def forward_diffusion_sample(x_0, t, device=config['device']):
    """
    Forward diffusion process
    x_t = sqrt(alphas_bar[t]) * x_0 + sqrt(1 - alphas_bar[t]) * noise
    """
    # Get noise schedule parameters
    timesteps = config['timesteps']
    beta_schedule = config['beta_schedule']
    
    if beta_schedule == 'linear':
        betas = linear_beta_schedule(timesteps, config['beta_start'], config['beta_end']).to(device)
    elif beta_schedule == 'cosine':
        betas = cosine_beta_schedule(timesteps, config['s']).to(device)
    
    alphas = (1. - betas).to(device)
    alphas_bar = torch.cumprod(alphas, dim=0).to(device)
    sqrt_alphas_bar = torch.sqrt(alphas_bar).to(device)
    sqrt_one_minus_alphas_bar = torch.sqrt(1. - alphas_bar).to(device)

    # Get the values at timestep t
    sqrt_alphas_bar_t = get_index_from_list(sqrt_alphas_bar, t, x_0.shape)
    sqrt_one_minus_alphas_bar_t = get_index_from_list(sqrt_one_minus_alphas_bar, t, x_0.shape)
    
    # Generate random noise
    noise = torch.randn_like(x_0)
    
    # Return noisy image using the closed form solution
    return sqrt_alphas_bar_t * x_0 + sqrt_one_minus_alphas_bar_t * noise, noise

def get_diffusion_parameters(device=config['device']):
    """
    Get diffusion params for a given beta schedule
    """
    
    # Get noise schedule parameters
    timesteps=config['timesteps']
    beta_schedule=config['beta_schedule']
    beta_start=config['beta_start']
    beta_end=config['beta_end']

    # Get diffusion parameters
    betas = cosine_beta_schedule(timesteps, s=0.008).to(device)

    # Pre-calculate different terms for closed form
    alphas = (1. - betas).to(device)
    alphas_cumprod = torch.cumprod(alphas, dim=0).to(device)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0).to(device)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas).to(device) # for calculation of model mean
    
    # Calculations for diffusion q(x_t | x_{t-1})
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).to(device)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod).to(device)
    
    # Calculation for posterior q(x_{t-1} | x_t, x_0)
    posterior_variance = (betas * (1-alphas_cumprod_prev) / (1-alphas_cumprod)).to(device)
    
    return {
        'betas': betas,
        'alphas': alphas,
        'alphas_cumprod': alphas_cumprod,
        'alphas_cumprod_prev': alphas_cumprod_prev,
        'sqrt_recip_alphas': sqrt_recip_alphas,
        'sqrt_alphas_cumprod': sqrt_alphas_cumprod,
        'sqrt_one_minus_alphas_cumprod': sqrt_one_minus_alphas_cumprod,
        'posterior_variance': posterior_variance,
    }

def sample_timesteps(batch_size, timesteps=config['timesteps'], device=config['device']):
    return torch.randint(0, timesteps, (batch_size,), device=device).long()

def reverse_diffusion_sample(model, x, t, device=config['device']):
    """
    Reverse diffusion process
    """
    diffusion_params=get_diffusion_parameters(device=device)
    
    betas = diffusion_params['betas']
    
    # Calculate the mean for the posterior distribution
    sqrt_recip_alphas = diffusion_params['sqrt_recip_alphas']
    sqrt_one_minus_alphas_cumprod = diffusion_params['sqrt_one_minus_alphas_cumprod']
    posterior_variance = diffusion_params['posterior_variance']

    # Extract values at timestep t
    betas_t = get_index_from_list(betas, t, x.shape)
    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(sqrt_one_minus_alphas_cumprod, t, x.shape)
    posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)
    
    # Get the predict noise from the model
    predicted_noise = model(x, t)
    
    # Check for NaN values in predicted noise and replace them
    if torch.isnan(predicted_noise).any():
        print(f"Warning: NaN values detected in predicted noise at timestep {t[0]}. Replacing with zeros.")
        predicted_noise = torch.where(torch.isnan(predicted_noise), torch.zeros_like(predicted_noise), predicted_noise)
    
    # Avoid division by zero or very small numbers
    epsilon = 1e-8
    safe_denominator = sqrt_one_minus_alphas_cumprod_t.clone()
    safe_denominator[safe_denominator < epsilon] = epsilon
    
    # Calculate the model mean with numerical stability safeguards
    noise_term = betas_t * predicted_noise / safe_denominator
    model_mean = sqrt_recip_alphas_t * (x - noise_term)
    
    # Check for NaN values in model mean and replace them
    if torch.isnan(model_mean).any():
        print(f"Warning: NaN values detected in model mean at timestep {t[0]}. Replacing with input x.")
        model_mean = torch.where(torch.isnan(model_mean), x, model_mean)
    
    # No noise when t == 0
    if t[0] == 0:
        return model_mean
    else:
        noise = torch.randn_like(x)
        # Ensure posterior_variance_t is positive
        safe_variance = torch.clamp(posterior_variance_t, min=0.0)
        return model_mean + torch.sqrt(safe_variance) * noise

def p_sample(model, shape, timesteps=config['timesteps'], device=config['device']):
   
    # Start with random noise
    img = torch.randn(shape, device=device)
    imgs = []
    
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
            
            imgs.append(img.cpu().numpy())
            
    return img, imgs

if __name__ == "__main__":
    print(linear_beta_schedule(100, 1e-4, 0.02))
    print(cosine_beta_schedule(timesteps=100, s=0.008))  