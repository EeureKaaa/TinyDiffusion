import torch
from utils.config import config
from model.unet import SimpleUnet
from utils.data_utils import get_data_loaders
from utils.diffusion import sample_timesteps, forward_diffusion_sample
from utils.config import is_wandb_ready
import wandb
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from tqdm import tqdm

# Initialize model
model = SimpleUnet()
model.to(config['device'])

def train():
    device=config['device']
    print(f"Using device: {device}")

    use_wandb = is_wandb_ready()
    if use_wandb:
        # Initialize wandb with configuration from config.py
        wandb.init(
            project=config['wandb']['project'],
            name=config['wandb']['name'],
            config=config,
            save_code=config['wandb']['save_code']
        )

    # Get data loaders
    train_loader, test_loader = get_data_loaders()

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    # MSE loss
    mse = nn.MSELoss()
    
    # Training loop
    for epoch in range(config['epochs']):
        running_loss = 0.0
        # Create progress bar for batches in this epoch
        pbar = tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{config['epochs']}", leave=True)
        
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

            # Update running loss and progress bar
            loss_value = loss.item()
            running_loss += loss_value
            avg_loss = running_loss / (i + 1)
            pbar.set_postfix({"loss": f"{avg_loss:.6f}"})
            pbar.update(1)
            
            # Log batch loss to wandb
            if use_wandb:
                wandb.log({"batch_loss": loss_value, "step": epoch * len(train_loader) + i})
        
        # Close progress bar for this epoch
        pbar.close()
        
        # Log to wandb if enabled
        if use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "epoch_loss": avg_loss,
                "learning_rate": optimizer.param_groups[0]['lr']
            })
            
        # Print epoch summary
        print(f"Epoch {epoch+1}/{config['epochs']} completed. Avg Loss: {avg_loss:.6f}")
        
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

if __name__ == "__main__":
    train()
            
            
            
            
            
            
        