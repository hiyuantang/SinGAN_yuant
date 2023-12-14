import json
import argparse
from discriminator import Discriminator
from generator import Generator
import torch
import os
from utils import *

# Use this .py file for paint-to-image application after you have trained at least one model
# Modify 3 necessary parts to make this script work

# Path to the JSON file containing the arguments
result_full_path = '../results/balloons_downsampled_1' # Part 1: define your model path

with open(result_full_path+'/args.json', 'r') as file:
    args_dict = json.load(file)

args = argparse.Namespace(**args_dict)


def load_models_for_scale(scale_path, args):
    # Initialize the models with the parameters from args
    g = Generator(
        in_channels=args.in_channels,
        features=args.g_features,
        num_blocks=args.g_num_blocks,
        kernel_size=args.g_kernel_size
    ).to(args.device)
    d = Discriminator(
        in_channels=args.in_channels,
        max_features=args.d_max_features,
        min_features=args.d_min_features,
        num_blocks=args.d_num_blocks,
        kernel_size=args.d_kernel_size,
        padding=0,
        normalization=args.normalization
    ).to(args.device)

    # Load the state dictionaries from the .pth files
    g.load_state_dict(torch.load(os.path.join(scale_path, 'g.pth'), map_location=args.device))
    d.load_state_dict(torch.load(os.path.join(scale_path, 'd.pth'), map_location=args.device))
    
    return g, d

models = {}
for i in range(5): 
    scale_dir = os.path.join(result_full_path, f'scale_{i}')
    models[f'scale_{i}'] = load_models_for_scale(scale_dir, args)

test_results_dir = 'test_results'
current_dir = os.getcwd()
test_results_path = os.path.join(current_dir, test_results_dir)
if not os.path.exists(test_results_path):
    os.makedirs(test_results_path)

editing_0 = prepare_image(os.path.join(args.input_path, 'edit_0.png')) # Part 2: define paint input name. 
                                                                        # You can find available inputs by browsing data/images/
                                                                        # Paint input names start with 'edit'
editing_0_scale_0 = (scale_image(editing_0, 25, 33)).to(args.device)

# test_noise = generate_fixed_noise(3, 44, 59, 'cpu', number_of_noises=1, repeat=False, seed=42)
# save_image(test_noise, os.path.join(test_results_path, 'noise_2.png'))

real_image = prepare_image(args.input_path+args.input_image)
real_image = real_image.to(args.device)
real_image = scale_image(real_image, 25, 33)
save_image(models['scale_0'][0](editing_0_scale_0, real_image), os.path.join(test_results_path, 'result_0.png')) # Part 3: define output image name.
                                                                                                                # Better to be different each time of running
                                                                                                                # Results will be recorded at src/test_results/
