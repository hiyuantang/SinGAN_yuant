import argparse
from math import ceil
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
from generator import Generator
from discriminator import Discriminator
from utils import *

def main():
    parser = argparse.ArgumentParser(description='SinGAN Training')
    parser.add_argument('--device', default=device_option(),
                        help='Device to use for computation (default: CUDA if available, then MPS, else CPU)')
    parser.add_argument('--input_path', type=str, default='../data/images/',
                        help='Path to the directory containing input images.')
    parser.add_argument('--input_image', type=str, required=True,
                        help='Name of the input image file (e.g., "stone.png").')
    parser.add_argument('--result_path', type=str, default='../results/',
                        help='Path for saving the results.')

    args = parser.parse_args()

    # Ensure the result path ends with '/'
    result_path = os.path.join(args.result_path, '')

    # Create the results directory if it does not exist
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    # Construct full path for input
    input_full_path = os.path.join(args.input_path, args.input_image)

    # Generate a unique hash for the output file
    unique_hash = generate_random_hash()
    result_file_name = args.input_image.split('.')[0] + '_' + unique_hash
    result_full_path = os.path.join(result_path, result_file_name)

    # Check if the result_full_path exists, if so, generate a new hash
    while os.path.exists(result_full_path):
        unique_hash = generate_random_hash()
        result_file_name = args.input_image.split('.')[0] + '_' + unique_hash
        result_full_path = os.path.join(result_path, result_file_name)

    # Create the full results directory
    os.makedirs(result_full_path)

    train_pyramid(result_full_path, num_scale=5, number_of_noises=100, real_image_path=args.input_path+args.input_image, num_epochs=3000, 
                lr_g=0.001, lr_d=0.001, alpha=1.0, device=args.device, patch_size=11, in_channels=3, g_features=32, d_max_features=32, 
                d_min_features=32, g_num_blocks=3, d_num_blocks=3, g_kernel_size=3, d_kernel_size=3, normalization=True)
    
def train_single_scale(result_full_path, scale_dir, real_image_scaled, noise_fixed, num_epochs, lr_g, lr_d, 
                       alpha, device, patch_size, scale_index, in_channels, g_features, d_max_features, 
                       d_min_features, g_num_blocks, d_num_blocks, g_kernel_size, d_kernel_size, normalization):
    # Initialize Generator and Discriminator
    g = Generator(in_channels=in_channels, features=g_features, num_blocks=g_num_blocks, kernel_size=g_kernel_size).to(device)
    d = Discriminator(in_channels=in_channels, max_features=d_max_features, min_features=d_min_features, 
                      num_blocks=d_num_blocks, kernel_size=d_kernel_size, padding=0, normalization=normalization).to(device)

    # Optimizers
    optimizer_g = optim.Adam(g.parameters(), lr=lr_g)
    optimizer_d = optim.Adam(d.parameters(), lr=lr_d)

    train_dir = os.path.join(scale_dir, 'train')
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    
    real_image_scaled = real_image_scaled.to(device)
    if scale_index > 0:
        rec_fake_reserve = torch.load(os.path.join(result_full_path, f'scale_{scale_index-1}', 'rec_fake_reserve', 'fake_rec.pt'))
        rec_fake_reserve = scale_image(rec_fake_reserve, real_image_scaled.size()[2], real_image_scaled.size()[3])

    for epoch in range(num_epochs):
        # Generate noise and move data to device
        noise = generate_noise(real_image_scaled, device, repeat=True)
        
        # Create the zero tensor for the generator input at scale 0
        if scale_index == 0:
            fake_image_scaled = torch.zeros_like(real_image_scaled).to(device)
        else:
            previous_scale_output_dir = os.path.join(result_full_path, f'scale_{scale_index-1}', 'outputs')
            fake_images = os.listdir(previous_scale_output_dir)
            selected_fake_image = random.choice(fake_images)
            fake_image_path = os.path.join(previous_scale_output_dir, selected_fake_image)
            fake_image_scaled = load_and_scale_image(fake_image_path, real_image_scaled.size()[2:]).to(device)

        # Discriminator training
        optimizer_d.zero_grad()
        fake_image_generated = g(noise, fake_image_scaled).detach()
        real_patches = extract_patches(real_image_scaled, patch_size)
        fake_patches = extract_patches(fake_image_generated, patch_size)
        d_loss_real = d(real_patches).mean()
        d_loss_fake = d(fake_patches).mean()
        gradient_penalty = calculate_gradient_penalty(d, real_patches, fake_patches, device)
        d_loss = d_loss_fake - d_loss_real + gradient_penalty
        d_loss.backward()
        optimizer_d.step()

        # Generator training
        optimizer_g.zero_grad()
        # Adversarial loss
        fake_image_generated = g(noise, fake_image_scaled)
        fake_patches = extract_patches(fake_image_generated, patch_size)
        g_loss_fake_adv = d(fake_patches).mean()
        g_loss_adv = -g_loss_fake_adv
        # Reconstruction loss
        if scale_index > 0:  
            fake_image_generated = g(torch.zeros_like(real_image_scaled), rec_fake_reserve)
        else:
            fake_image_generated = g(noise_fixed, fake_image_scaled)
        if epoch % 100 == 0:
            save_image(fake_image_generated[0], os.path.join(train_dir, f'train_epoch{epoch}.png'))
        real_image_expanded = real_image_scaled.expand_as(fake_image_generated)
        g_loss_rec = torch.norm(fake_image_generated - real_image_expanded, p=2, dim=[1, 2, 3]) ** 2
        g_loss_rec = g_loss_rec.mean()
        # Generator total loss
        g_loss_total = g_loss_adv + alpha * g_loss_rec
        g_loss_total.backward()
        optimizer_g.step()

        # Logging the training process
        if epoch % 100 == 0 or epoch == num_epochs - 1:
            print(f"Epoch [{epoch}/{num_epochs}] - D loss: {round(d_loss.item(), 5)} - G loss: {round(g_loss_total.item(), 5)} - Rec loss: {round(g_loss_rec.item(), 5)} - Adv loss: {round(g_loss_adv.item(), 5)}")

    # Return the trained generator and discriminator
    return g, d
    
def train_pyramid(result_full_path, num_scale, number_of_noises, real_image_path, num_epochs, lr_g, lr_d, alpha, device, 
                  patch_size, in_channels, g_features, d_max_features, d_min_features, g_num_blocks, 
                  d_num_blocks, g_kernel_size, d_kernel_size, normalization):
    # Prepare real image
    real_image = prepare_image(real_image_path)
    real_image = real_image.to(device)
    original_height, original_width = real_image.size(2), real_image.size(3)

    # Generate list of scales
    scale_factors = [i / num_scale for i in range(1, num_scale+1)]
    h_list = [int(original_height * scale) for scale in scale_factors]
    w_list = [int(original_width * scale) for scale in scale_factors]

    # Fixed noise for reconstruction loss
    noise_fixed = generate_fixed_noise(3, h_list[0], w_list[0], device, number_of_noises=number_of_noises, repeat=True, seed=42)

    # Training loop for each scale
    for i in range(num_scale):
        # Generate the directory name for the current scale
        scale_dir = os.path.join(result_full_path, f'scale_{i}')
        
        # Create the directory if it doesn't exist
        if not os.path.exists(scale_dir):
            os.makedirs(scale_dir)
        
        # Save the scaled real image
        real_dir = os.path.join(scale_dir, 'real_image')
        if not os.path.exists(real_dir):
            os.makedirs(real_dir)
        real_image_scaled = scale_image(real_image, h_list[i], w_list[i])
        save_image(real_image_scaled, os.path.join(real_dir, f'real_image_{i}.png'))

        output_dir = os.path.join(scale_dir, 'outputs')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        rec_fake_reserve_dir = os.path.join(scale_dir, 'rec_fake_reserve')
        if not os.path.exists(rec_fake_reserve_dir):
            os.makedirs(rec_fake_reserve_dir)

        g, d = train_single_scale(result_full_path, scale_dir, real_image_scaled, noise_fixed, num_epochs, lr_g, lr_d, 
                       alpha, device, patch_size, i, in_channels, g_features, d_max_features, 
                       d_min_features, g_num_blocks, d_num_blocks, g_kernel_size, d_kernel_size, normalization)
        
        # Save the trained models
        save_model(g, os.path.join(scale_dir, 'g.pth'))
        save_model(d, os.path.join(scale_dir, 'd.pth'))

        # Generate and save fake images outputs
        num_images = ceil(num_epochs / 100)
        for j in range(num_images):
            noise = generate_noise(real_image_scaled, device, repeat=True)
            fake_image = g(noise, torch.zeros_like(real_image_scaled)).detach()
            save_image(fake_image, os.path.join(output_dir, f'fake_image_{j}.png'))
    
        try:
            loaded_tensor = torch.load(os.path.join(result_full_path, f'scale_{i-1}', 'rec_fake_reserve', 'fake_rec.pt'))
            # Scale the loaded tensor to match the real_image_scaled dimensions
            scaled_tensor = scale_image(loaded_tensor, real_image_scaled.size()[2], real_image_scaled.size()[3])
        except:
            pass
        
        # Generate and save fake images reserved for calculating reconstruction loss
        if i > 0:  
            fake_images_rec = g(torch.zeros_like(real_image_scaled), scaled_tensor)
        else:
            fake_images_rec = g(noise_fixed, torch.zeros_like(real_image_scaled))
        torch.save(fake_images_rec, os.path.join(rec_fake_reserve_dir, 'fake_rec.pt'))
        for j in range(number_of_noises):
            save_image(fake_images_rec[j], os.path.join(rec_fake_reserve_dir, f'fake_rec_{j}.png'))



if __name__ == '__main__':
    main()
