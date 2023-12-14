import torch
import torch.nn as nn
import torch.nn.functional as F
from discriminator import Discriminator

class Generator(nn.Module):
    def __init__(self, in_channels, features, num_blocks, kernel_size):
        super().__init__()

        if not (kernel_size % 2 == 1 and kernel_size >= 3):
            raise ValueError("kernel_size must be an odd number and >= 3")

        padding = kernel_size // 2
        # Constructing the sequential blocks
        blocks = [Discriminator.create_block(in_channels=in_channels, out_channels=features, kernel_size=kernel_size, padding=padding, normalization=False)]

        for _ in range(num_blocks - 1):
            blocks.append(Discriminator.create_block(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, normalization=False))

        self.features = nn.Sequential(*blocks)

        # Classifier layer
        self.last_conv = nn.Conv2d(in_channels=features, out_channels=in_channels, kernel_size=kernel_size, padding=padding)


    def forward(self, noise, x):
        # Combining the noise and the real image
        combined_input = x + noise
        combined_input = self.features(combined_input)
        combined_input = self.last_conv(combined_input)
        return torch.sigmoid(combined_input + x)
    
    def get_config(self):
        config_str = 'Generator Configuration:\n'
        for i, layer in enumerate(self.features):
            config_str += f'Feature Block {i}: {layer}\n'
        config_str += f'Classifier: {self.classifier}\n{self.activation}\n'
        return config_str
    
if __name__ == '__main__':
    generator = Generator(in_channels=3, features=32, num_blocks=5, kernel_size=3)
    print(generator.get_config())
