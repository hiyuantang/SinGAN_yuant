import json
import argparse

# Path to the JSON file containing the arguments
result_full_path = './results/balloons_downsampled_gixpq/args.json'

# Read and deserialize args from the file
with open(result_full_path, 'r') as file:
    args_dict = json.load(file)

# Convert to argparse.Namespace if needed
args = argparse.Namespace(**args_dict)

