import os 
from PIL import Image
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt
from pathlib import Path

def load_image(path):
    return Image.open(path).convert('L')  # Convert to grayscale

def process_image(img, target_size=(299, 299)):
    # Convert PIL Image to tensor
    if isinstance(img, Image.Image):
        # Convert to numpy array and normalize to [0, 1]
        img_array = np.array(img).astype(np.float32) / 255.0
        # Add channel dimension if grayscale
        if len(img_array.shape) == 2:
            img_array = img_array[..., np.newaxis]
        # Convert to torch tensor [C, H, W]
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
    else:
        # Assume it's already a numpy array
        img_array = img.astype(np.float32) / 255.0
        if len(img_array.shape) == 2:
            img_array = img_array[..., np.newaxis]
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
    
    # Repeat to 3 channels if grayscale
    if img_tensor.shape[0] == 1:
        img_tensor = img_tensor.repeat(3, 1, 1)  # [1, H, W] -> [3, H, W]
    
    # Resize to target size
    img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
    img_tensor = F.interpolate(img_tensor, size=target_size, mode='bilinear')
    img_tensor = img_tensor.squeeze(0)  # Remove batch dimension
    
    # Normalize to [0, 1]
    img_tensor = (img_tensor - img_tensor.min()) / (img_tensor.max() - img_tensor.min())
    
    # Convert back to numpy
    img_processed = img_tensor.permute(1, 2, 0).numpy()
    
    return img_processed

def process_images(image_dir, output_dir, target_size=(299, 299)):
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = [f for f in os.listdir(image_dir) 
                  if os.path.isfile(os.path.join(image_dir, f)) and 
                  os.path.splitext(f)[1].lower() in valid_extensions]
    
    print(f"Processing {len(image_files)} images...")
    
    for img_file in tqdm(image_files, desc="Processing images"):
        # Load image
        img_path = os.path.join(image_dir, img_file)
        img = load_image(img_path)
        
        # Process image
        processed_img = process_image(img, target_size)
        
        # Save processed image
        plt.figure(figsize=(5, 5))
        plt.imshow(processed_img)
        plt.axis('off')
        plt.savefig(output_path / img_file, bbox_inches='tight', pad_inches=0)
        plt.close()
    
    print(f"Processed images saved to {output_dir}")
    return output_dir

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Process images for FID calculation')
    parser.add_argument('--image_dir', type=str, required=True, help='Directory containing images')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save processed images')
    parser.add_argument('--target_size', type=int, nargs=2, default=[28, 28], help='Target size for resizing images')
    
    args = parser.parse_args()
    
    process_images(args.image_dir, args.output_dir, tuple(args.target_size))

if __name__ == "__main__":
    main()