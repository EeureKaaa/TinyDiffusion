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
        betas = linear_beta_schedule(timesteps, config['beta_start'], config['beta_end'])
    elif beta_schedule == 'cosine':
        betas = cosine_beta_schedule(timesteps, config['s'])
    
    alphas=1. - betas
    alphas_bar=torch.cumprod(alphas, dim=0)
    sqrt_alphas_bar = torch.sqrt(alphas_bar)
    sqrt_one_minus_alphas_bar = torch.sqrt(1. - alphas_bar)

    # Get the values at timestep t
    sqrt_alphas_bar_t = get_index_from_list(sqrt_alphas_bar, t, x_0.shape)
    sqrt_one_minus_alphas_bar_t = get_index_from_list(sqrt_one_minus_alphas_bar, t, x_0.shape)
    
    # Generate random noise
    noise = torch.randn_like(x_0)
    
    # Return noisy image using the closed form solution
    return sqrt_alphas_bar_t * x_0 + sqrt_one_minus_alphas_bar_t * noise, noise

def get_diffusion_parameters():
    """
    Get diffusion params for a given beta schedule
    """
    
    # Get noise schedule parameters
    timesteps=config['timesteps']
    beta_schedule=config['beta_schedule']
    beta_start=config['beta_start']
    beta_end=config['beta_end']

    # Get diffusion parameters
    betas = cosine_beta_schedule(timesteps, s=0.008)

    # Pre-calculate different terms for closed form
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas) # for calculation of model mean
    
    # Calculations for diffusion q(x_t | x_{t-1})
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    
    # Calculation for posterior q(x_{t-1} | x_t, x_0)
    posterior_variance = betas * (1-alphas_cumprod_prev) / (1-alphas_cumprod)
    
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

def reverse_diffusion_sample(model, x, t, diffusion_params, device=config['device']):
    """
    Reverse diffusion process
    """
    
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
    predicted_noise=model(x,t)

    model_mean=sqrt_recip_alphas_t*(x-betas_t*predicted_noise/sqrt_one_minus_alphas_cumprod_t)
    
    # No noise when t == 0
    if t[0] == 0:
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise

def p_sample(model, shape, timesteps=config['timesteps'], device=config['device']):
    diffusion_params=get_diffusion_parameters()
    
    img = torch.randn(shape, device=device)
    imgs = []
    
    for i in reversed(range(0, timesteps)):
        t = torch.full((shape[0],), i, device=device, dtype=torch.long)
        
        with torch.no_grad():
            img = reverse_diffusion_sample(model, img, t, diffusion_params, device)
            imgs.append(img.cpu().numpy())
            
    return img, imgs

if __name__ == "__main__":
    print(linear_beta_schedule(100, 1e-4, 0.02))
    print(cosine_beta_schedule(timesteps=100, s=0.008))  
    