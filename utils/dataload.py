from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from utils.config import config

def get_data_loaders(batch_size=None):
    """
    Create and return train and test data loaders for MNIST dataset
    """
    if batch_size is None:
        batch_size = config['batch_size']
        
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Load MNIST dataset
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
    