import torch
from utils.config import config
from model.unet import SimpleUnet
from utils.data_utils import get_data_loaders
from utils.diffusion import sample_timesteps, forward_diffusion_sample
from utils.config import is_wandb_ready
import wandb
import matplotlib.pyplot as plt
import numpy as np

# Initialize model
model = SimpleUnet()
model.to(config['device'])

def train():
    device=config['device']
    print(f"Using device: {device}")

    use_wandb = is_wandb_ready()
    if use_wandb:
        wandb.init(project="mnist-diffusion", config=config)

    # Get data loaders
    train_loader, test_loader = get_data_loaders()

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    # MSE loss
    mse = nn.MSELoss()
    
    # Training loop
    for epoch in range(config['epochs']):
        
        for i, (images, _) in enumerate(train_loader):
            images = images.to(device)
            batch_size = images.shape[0]
            
            t = sample_timesteps(batch_size, device=device)

            # Forward pass
            x_noisy, noise = forward_diffusion_sample(images, t, device=device)
            
            # Predict noise
            noise_pred = model(x_noisy, t)

            # Calculate loss
            loss = mse(noise, noise_pred)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update progress bar
            running_loss += loss.item()
            pbar.set_description(f"Epoch {epoch+1}/{config['epochs']}, Loss: {running_loss/(i+1):.6f}")
        
        # Save model checkpoint
        if (epoch + 1) % 10 == 0 or epoch == config['epochs'] - 1:
            torch.save(model.state_dict(), config['model_save_path'])
            print(f"Model saved to {config['model_save_path']}")
            
            # Generate and save sample images
            generate_samples(model, epoch, device=device)
    
    print("Training completed!")
    
    # Final sample generation
    generate_samples(model, config['epochs'], device=device)
    
    # Close wandb
    if use_wandb:
        wandb.finish()

            
            
            
            
            
            
        