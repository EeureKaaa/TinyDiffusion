import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms
from scipy import linalg
from tqdm import tqdm
import os
from PIL import Image

class InceptionV3Feature(nn.Module):
    """
    InceptionV3 network for feature extraction.
    Uses the model up to a specific layer to extract features.
    """
    def __init__(self, device='cuda'):
        super(InceptionV3Feature, self).__init__()
        # Load pretrained InceptionV3 model
        inception = models.inception_v3(pretrained=True)
        # We don't need the classification part, only the features
        self.model = nn.Sequential(
            # Take everything up to the final classification layer
            *list(inception.children())[:-1]
        )
        self.model.eval()
        self.device = device
        self.model = self.model.to(device)
        
        # Define the transformation for input images
        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225]),
        ])
    
    def forward(self, x):
        with torch.no_grad():
            features = self.model(x)
            # Reshape to get the feature vector
            features = features.squeeze(2).squeeze(2)
        return features


def calculate_activation_statistics(images, model, batch_size=32, device='cuda'):
    """
    Calculate the mean and covariance of the features extracted from the images.
    
    Args:
        images: List of PIL images or numpy arrays
        model: The feature extraction model
        batch_size: Batch size for processing
        device: Device to use for computation
        
    Returns:
        mu: Mean of features
        sigma: Covariance matrix of features
    """
    n_images = len(images)
    n_batches = (n_images + batch_size - 1) // batch_size
    
    # Initialize feature array
    features = []
    
    for i in tqdm(range(n_batches), desc="Extracting features"):
        start = i * batch_size
        end = min((i + 1) * batch_size, n_images)
        batch = images[start:end]
        
        # Process batch
        if isinstance(batch[0], np.ndarray):
            # Convert numpy arrays to PIL images
            batch = [Image.fromarray((img * 255).astype(np.uint8)) for img in batch]
        
        # Apply transformations
        batch_tensor = torch.stack([model.transform(img) for img in batch])
        batch_tensor = batch_tensor.to(device)
        
        # Extract features
        batch_features = model(batch_tensor).cpu().numpy()
        features.append(batch_features)
    
    # Concatenate all features
    features = np.concatenate(features, axis=0)
    
    # Calculate statistics
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    
    return mu, sigma


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    Calculate the Fréchet distance between two multivariate Gaussians.
    
    Args:
        mu1: Mean of the first distribution
        sigma1: Covariance of the first distribution
        mu2: Mean of the second distribution
        sigma2: Covariance of the second distribution
        eps: Small epsilon for numerical stability
        
    Returns:
        fid: Fréchet distance
    """
    # Calculate the squared difference between means
    diff = mu1 - mu2
    
    # Calculate the product of covariances
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    
    # Numerical stability
    if not np.isfinite(covmean).all():
        msg = f"FID calculation produced singular product; adding {eps} to diagonal of cov estimates"
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    
    # Check and correct imaginary components
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f"Imaginary component {m}")
        covmean = covmean.real
    
    # Calculate the trace term
    tr_covmean = np.trace(covmean)
    
    # Calculate FID
    fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
    
    return fid


def load_images_from_dir(directory, max_images=None):
    """
    Load images from a directory.
    
    Args:
        directory: Path to the directory containing images
        max_images: Maximum number of images to load (None for all)
        
    Returns:
        List of PIL images
    """
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    images = []
    
    files = [f for f in os.listdir(directory) 
             if os.path.isfile(os.path.join(directory, f)) and 
             os.path.splitext(f)[1].lower() in valid_extensions]
    
    if max_images is not None:
        files = files[:max_images]
    
    for file in tqdm(files, desc="Loading images"):
        try:
            img = Image.open(os.path.join(directory, file)).convert('RGB')
            images.append(img)
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    return images


def calculate_fid(real_images, generated_images, batch_size=32, device='cuda'):
    """
    Calculate the Fréchet Inception Distance between two sets of images.
    
    Args:
        real_images: List of real images (PIL images or numpy arrays)
        generated_images: List of generated images (PIL images or numpy arrays)
        batch_size: Batch size for processing
        device: Device to use for computation
        
    Returns:
        fid: Fréchet Inception Distance score
    """
    # Initialize the feature extractor
    inception = InceptionV3Feature(device=device)
    
    # Calculate statistics for real images
    mu_real, sigma_real = calculate_activation_statistics(
        real_images, inception, batch_size, device)
    
    # Calculate statistics for generated images
    mu_gen, sigma_gen = calculate_activation_statistics(
        generated_images, inception, batch_size, device)
    
    # Calculate FID
    fid = calculate_frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)
    
    return fid


def calculate_fid_from_dirs(real_dir, generated_dir, max_images=None, batch_size=32, device='cuda'):
    """
    Calculate the Fréchet Inception Distance between images in two directories.
    
    Args:
        real_dir: Directory containing real images
        generated_dir: Directory containing generated images
        max_images: Maximum number of images to use from each directory
        batch_size: Batch size for processing
        device: Device to use for computation
        
    Returns:
        fid: Fréchet Inception Distance score
    """
    # Load images
    print(f"Loading real images from {real_dir}")
    real_images = load_images_from_dir(real_dir, max_images)
    
    print(f"Loading generated images from {generated_dir}")
    generated_images = load_images_from_dir(generated_dir, max_images)
    
    # Ensure we have the same number of images
    min_count = min(len(real_images), len(generated_images))
    if min_count < len(real_images):
        print(f"Using only {min_count} real images to match generated count")
        real_images = real_images[:min_count]
    if min_count < len(generated_images):
        print(f"Using only {min_count} generated images to match real count")
        generated_images = generated_images[:min_count]
    
    # Calculate FID
    return calculate_fid(real_images, generated_images, batch_size, device)
