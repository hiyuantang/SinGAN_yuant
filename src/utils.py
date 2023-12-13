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

def generate_noise_single(tensor_like, device, repeat=False):
    if not repeat:
        noise = torch.randn(tensor_like.size())
    else:
        noise = torch.randn((tensor_like.size(0), 1, tensor_like.size(2), tensor_like.size(3)))
        noise = noise.repeat((1, 3, 1, 1))
    return noise.to(device)

def generate_noise(tensor_like, device, number_of_noises, repeat=False):
    c, h, w = tensor_like.size(1), tensor_like.size(2), tensor_like.size(3)
    
    if not repeat:
        noise = torch.randn((number_of_noises, c, h, w))
    else:
        single_channel_noise = torch.randn((number_of_noises, 1, h, w))
        noise = single_channel_noise.repeat(1, c, 1, 1)
    
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

def extract_patches(images, patch_size, stride=3):
    if not isinstance(images, torch.Tensor) or images.ndim != 4:
        raise ValueError("Input images must be a 4D PyTorch tensor")

    num_images, channels, height, width = images.shape

    if patch_size > height or patch_size > width:
        raise ValueError("Patch size cannot be larger than image dimensions")
    
    all_patches = []
    for n in range(num_images):
        patches = []
        for i in range(0, height - patch_size + 1, stride):
            for j in range(0, width - patch_size + 1, stride):
                # Extract the patch for the current image
                patch = images[n:n+1, :, i:i + patch_size, j:j + patch_size]
                patches.append(patch)
        
        # Convert list of patches for the current image into a 4D tensor
        patches = torch.cat(patches, dim=0)
        all_patches.append(patches)

    # Combine all patches from all images into a single tensor
    all_patches = torch.cat(all_patches, dim=0)
    
    return all_patches

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

from torch.autograd import grad as torch_grad, Variable
def gradient_penalty(device, real_data, d, generated_data):
        # calculate interpolation
        alpha = torch.rand(real_data.size(0), 1, 1, 1)
        alpha = alpha.expand_as(real_data)
        alpha = alpha.to(device)
        interpolated = alpha * real_data + (1 - alpha) * generated_data
        interpolated = Variable(interpolated, requires_grad=True)
        interpolated = interpolated.to(device)

        # calculate probability of interpolated examples
        prob_interpolated = d(interpolated)

        # calculate gradients of probabilities with respect to examples
        gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(prob_interpolated.size()).to(device),
                               create_graph=True, retain_graph=True, only_inputs=True)[0]

        # return gradient penalty
        return ((gradients.norm(2, dim=1) - 1) ** 2).mean()

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

