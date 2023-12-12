import os
import hashlib
import random
import string
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torchvision import transforms
import torchvision
import random
from PIL import Image

def generate_noise(tensor_like, device, repeat=False):
    """
    Generates random noise tensor with the same size as the given tensor or with repeated values across channels.

    This method creates a tensor of random noise that either matches the size of the input tensor (`tensor_like`)
    or has the noise repeated across its channels if `repeat` is True. The generated noise tensor is then transferred 
    to the specified device (CPU or GPU).

    Parameters:
    tensor_like (torch.Tensor): A PyTorch tensor whose size is used as a reference for generating the noise tensor.
    device (torch.device): The device (CPU or GPU) to which the noise tensor will be transferred.
    repeat (bool, optional): If True, the noise is generated for one channel and then repeated across all channels.
                            Defaults to False, where the noise is independently generated for each channel.

    Returns:
    torch.Tensor: A tensor of random noise, either matching the size of `tensor_like` or with repeated values across 
                its channels, transferred to the specified device.
    """
    if not repeat:
        noise = torch.randn(tensor_like.size())
    else:
        noise = torch.randn((tensor_like.size(0), 1, tensor_like.size(2), tensor_like.size(3)))
        noise = noise.repeat((1, 3, 1, 1))
    return noise.to(device)

def generate_fixed_noise(channels, height, width, device, number_of_noises=1, repeat=False, seed=42):
    # Set the random seed for reproducibility
    torch.manual_seed(seed)

    # Generate the fixed noise
    if not repeat:
        noise = torch.randn((number_of_noises, channels, height, width), device=device)
    else:
        # Generate noise for one channel and repeat
        noise = torch.randn((number_of_noises, 1, height, width), device=device).repeat((1, channels, 1, 1))

    return noise



def prepare_image(image_path):
    # Load image using OpenCV
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert to PyTorch tensor
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
    return image_tensor

def scale_image(image_tensor, new_height, new_width):
    # Resize the image to the new dimensions
    scaled_image = F.interpolate(image_tensor, size=(new_height, new_width), mode='bilinear', align_corners=False)
    return scaled_image

def device_option():
    # Check for CUDA
    if torch.cuda.is_available():
        default_device = 'cuda'
    # Check for MPS (Apple Silicon)
    elif torch.backends.mps.is_available():
        default_device = 'mps'
    else:
        default_device = 'cpu'
    return default_device

def generate_random_hash(length=5):
    """
    Generate a random hash of the specified length.
    """
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))

def extract_patches(image, patch_size):
    if not isinstance(image, torch.Tensor) or image.ndim != 4:
        raise ValueError("Input image must be a 4D PyTorch tensor")

    _, channels, height, width = image.shape

    if patch_size > height or patch_size > width:
        raise ValueError("Patch size cannot be larger than image dimensions")
    
    patches = []
    for i in range(0, height - patch_size + 1, patch_size):
        for j in range(0, width - patch_size + 1, patch_size):
            # Extract the patch using all dimensions
            patch = image[:, :, i:i + patch_size, j:j + patch_size]
            patches.append(patch)

    # Convert list of patches into a 4D tensor
    patches = torch.stack(patches, dim=0)
    # Reshape to (N, channels, patch_size, patch_size)
    N = len(patches)
    patches = patches.view(N, channels, patch_size, patch_size)
    
    return patches

def calculate_gradient_penalty(discriminator, real_images, fake_images, device, lambda_gp=10):
    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand(real_images.size(0), 1, 1, 1, device=device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_images + ((1 - alpha) * fake_images)).requires_grad_(True)
    d_interpolates = discriminator(interpolates)
    fake_output = torch.ones(d_interpolates.size(), device=device)

    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates, inputs=interpolates,
        grad_outputs=fake_output, create_graph=True, retain_graph=True, only_inputs=True
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = lambda_gp * ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def save_model(model, path):
    """Save the model to the specified path."""
    torch.save(model.state_dict(), path)

def save_image(image, path):
    """Save the image to the specified path."""
    # Convert the image tensor to a suitable format for saving
    # This conversion depends on how your images are represented
    # For example, if the image is in the range [0, 1], you might want to convert it to [0, 255]
    torchvision.utils.save_image(image, path)

def load_and_scale_image(image_path, target_size):
    """Load and scale an image to the target size."""
    image = Image.open(image_path).convert('RGB')  # Assuming RGB images, modify if different
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor()
    ])
    return transform(image)

