from datasets import load_dataset, Dataset, Features, Image, Value, Sequence
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from PIL import Image as pilimage
import numpy as np
from huggingface_hub import login, create_branch, delete_branch
import cv2
import logging
logging.basicConfig(level=logging.INFO)

def gaussian_kernel(kernel_size, sigma):
    """Create a 2D Gaussian kernel using PyTorch"""
    x = torch.linspace(-kernel_size // 2 + 0.5, kernel_size // 2 - 0.5, kernel_size)
    kernel_1d = torch.exp(-0.5 * (x / sigma).pow(2))
    kernel_2d = kernel_1d.unsqueeze(0) * kernel_1d.unsqueeze(1)
    kernel_2d = kernel_2d / kernel_2d.sum()
    return kernel_2d

def otsu_pytorch(image):
    """PyTorch implementation of Otsu thresholding"""
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Convert to PyTorch tensor and add batch & channel dimensions
    img_tensor = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)
    
    # Apply Gaussian blur
    sigma = 10
    kernel_size = sigma * 4
    kernel = gaussian_kernel(kernel_size, sigma)
    kernel = kernel.unsqueeze(0).unsqueeze(0)
    img_blurred = F.conv2d(img_tensor, kernel, padding=kernel_size//2)
    
    # Calculate histogram
    hist = torch.histc(img_blurred, bins=256, min=0, max=255)
    hist = hist / hist.sum()
    
    # Calculate cumulative sums
    cumsum = torch.cumsum(hist, dim=0)
    cumsum_vals = torch.cumsum(hist * torch.arange(256), dim=0)
    
    # Find threshold that maximizes between-class variance
    global_mean = cumsum_vals[-1]
    max_variance = 0
    threshold = 0
    
    for t in range(256):
        w0 = cumsum[t]
        w1 = 1 - w0
        if w0 == 0 or w1 == 0:
            continue
            
        mu0 = cumsum_vals[t] / w0
        mu1 = (global_mean - cumsum_vals[t]) / w1
        
        variance = w0 * w1 * (mu0 - mu1) ** 2
        if variance > max_variance:
            max_variance = variance
            threshold = t
    
    # Apply threshold
    mask = (img_blurred > threshold).float()
    return mask.squeeze().numpy()

def process_image(example):
    """Process a single image example"""
    try:
        image = np.array(example['image'])
    except Exception as e:
        print(f"Error loading image for dr8_id {example['dr8_id']}: {str(e)}")
        image = np.zeros((512, 512, 3), dtype=np.uint8)
        image[:, :, 0] = 255

    # Get galaxy mask using Otsu
    mask = otsu_pytorch(image)
    
    # Count non-zero pixels to get galaxy size
    gsize = int(np.count_nonzero(mask))
    
    # Define cropping parameters based on size
    cropping_threshold = 17500
    cropping_x, cropping_y = 128, 384
    new_size = (256, 256)
    center_x, center_y = 255, 255  # Center of 512x512 image
    
    if gsize > cropping_threshold:
        # For large galaxies, resize the whole image
        pil_image = pilimage.fromarray(image)
        resized_image = pil_image.resize(new_size)
    else:
        # Get center region and its mask
        center_crop = image[cropping_x:cropping_y, cropping_x:cropping_y]
        center_mask = mask[cropping_x:cropping_y, cropping_x:cropping_y]
        # Find connected components
        num_features, labeled_mask = cv2.connectedComponents(center_mask.astype(np.uint8))

        # Find which component contains the center
        rel_center_y = center_y - cropping_x
        rel_center_x = center_x - cropping_x
        center_label = labeled_mask[rel_center_y, rel_center_x]

        if center_label == 0:  # No galaxy at center
            cropped = center_crop
        else:
            # Get just the central galaxy mask
            center_galaxy_mask = (labeled_mask == center_label)
            y_indices, x_indices = np.nonzero(center_galaxy_mask)
            
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)
            
            # Add padding (20% of galaxy size)
            padding = int(max(x_max - x_min, y_max - y_min) * 0.4)
            x_min = max(0, x_min - padding)
            x_max = min(center_crop.shape[1], x_max + padding)
            y_min = max(0, y_min - padding)
            y_max = min(center_crop.shape[0], y_max + padding)
            
            cropped = center_crop[y_min:y_max, x_min:x_max]

        # For smaller galaxies, crop then resize
        resized_image = pilimage.fromarray(cropped).resize(new_size)
    
    # Convert back to numpy array
    processed_image = np.array(resized_image, dtype=np.uint8)
    
    # Add galaxy size to example
    example['galaxy_size'] = gsize
    example['image_crop'] = pilimage.fromarray(processed_image)

    return example

def main():
    # Load dataset
    dataset = load_dataset("Smith42/galaxies", cache_dir="/raid/huggingface_cache/galaxies")
    
    repo_id = "Smith42/galaxies"
    new_branch = "refs/pr/2"
    
    new_features = Features({
        'image': Image(),
        'image_crop': Image(),
        'dr8_id': Value('string'),
        'galaxy_size': Value('int64'),
    })

    for split_name in ["train"]:#["test", "validation", "train"]:
        print(f"Processing split: {split_name}")
        processed_dataset = dataset[split_name].map(
            process_image,
            features = new_features,
            new_fingerprint="cached_processing",
            load_from_cache_file=True,
        )
        processed_dataset.push_to_hub(
            repo_id,
            max_shard_size="5GB",
            config_name = "with_crops",
            split=split_name,
            revision=new_branch
        )
        
if __name__ == "__main__":
    main()
