config = {
    # Dataset parameters
    'dataset': 'MNIST',
    'batch_size': 128,
    'image_size': 28,
    'channels': 1,  # MNIST is grayscale

    # Noise schedule parameters
    'beta_schedule': 'linear',
    'beta_start': 1e-4,
    'beta_end': 0.02,
    's': 0.008,

    # Diffusion parameters
    'timesteps': 1000,
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),

    # Training parameters
    'epochs': 100,
    'learning_rate': 2e-4,
}

# Paths for saving models and outputs
config_update = {
    'model_save_path': f'./checkpoints/{config["dataset"]}_diffusion_model.pt',
    'generated_images_path': f'./outputs/{config["dataset"]}/generated/',
    'sample_grid_path': f'./outputs/{config["dataset"]}/sample_grid/',
}
config.update(config_update)
