import torch
import wandb

def is_wandb_ready():
    try:
        import wandb
        return True
    except ImportError:
        print("Wandb is not installed")
        return False
        

config = {
    # Dataset parameters
    'dataset': 'MNIST',
    'batch_size': 128,
    'image_size': 28,
    'channels': 1,  # MNIST is grayscale

    # Noise schedule parameters
    'beta_schedule': 'cosine',
    'beta_start': 1e-4,
    'beta_end': 0.02,
    's': 0.008,

    # Diffusion parameters
    'timesteps': 500,
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),

    # Training parameters
    'epochs': 50,
    'learning_rate': 2e-4,
    'save_interval': 5
}

# Paths for saving models and outputs
config_update = {
    'model_save_path': f'./checkpoints/{config["dataset"]}_{config["beta_schedule"]}_{config["timesteps"]}/diffusion_model.pt',
    'generated_images_path': f'./outputs/{config["dataset"]}_{config["beta_schedule"]}_{config["timesteps"]}/generated/',
    'sample_grid_path': f'./outputs/{config["dataset"]}_{config["beta_schedule"]}_{config["timesteps"]}/sample_grid/',
}
config.update(config_update)

# Wandb configuration
config['wandb'] = {
    'project': 'mnist-diffusion',
    'name': f'{config["dataset"]}_diffusion',
    'save_code': True,
}

