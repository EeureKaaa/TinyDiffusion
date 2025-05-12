import numpy as np

def normalize_image(img, epsilon=1e-6, default_pattern=True):
    """
    Normalize an image to [0, 1] range with safeguards against division by zero
    or very small numbers.
    
    Args:
        img: Input image array
        epsilon: Small value to add to denominator to prevent division by zero
        default_pattern: Whether to use a gradient pattern (True) or flat gray (False) when range is too small
        
    Returns:
        Normalized image in [0, 1] range
    """
    img_min, img_max = img.min(), img.max()
    img_range = img_max - img_min
    
    # Check if the image has a meaningful range
    if img_range < epsilon:
        print(f"Warning: Image has very small range: {img_range}. Using default values.")
        if default_pattern:
            # Create a gradient pattern based on image dimensions
            h, w = img.shape if len(img.shape) == 2 else img.shape[:2]
            y = np.linspace(0, 1, h)[:, np.newaxis]
            x = np.linspace(0, 1, w)[np.newaxis, :]
            pattern = (y + x) / 2
            
            # Reshape pattern to match img dimensions
            if len(img.shape) > 2:
                pattern = pattern[:, :, np.newaxis]
                if img.shape[2] > 1:
                    pattern = np.repeat(pattern, img.shape[2], axis=2)
            
            return pattern
        else:
            # Use a flat gray image
            return 0.5 * np.ones_like(img)
    else:
        # Normal normalization with epsilon for numerical stability
        return (img - img_min) / (img_range + epsilon)
