import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm as SpectralNorm

class Discriminator(nn.Module):
    def __init__(self, in_channels, max_features, min_features, num_blocks, kernel_size, padding, normalization=False):
        super().__init__()

        if not (kernel_size % 2 == 1 and kernel_size >= 3):
            raise ValueError("kernel_size must be an odd number and >= 3")

        # Constructing the sequential blocks
        blocks = [self.create_block(in_channels=in_channels, out_channels=max_features, kernel_size=kernel_size, padding=padding, normalization=normalization)]

        f = max_features
        for _ in range(num_blocks - 2):
            f_next = max(min_features, f // 2)
            blocks.append(self.create_block(in_channels=f, out_channels=f_next, kernel_size=kernel_size, padding=padding, normalization=normalization))
            f = f_next

        self.features = nn.Sequential(*blocks)

        # Classifier layer
        self.classifier = nn.Conv2d(in_channels=f, out_channels=1, kernel_size=kernel_size, padding=padding)



    @staticmethod
    def create_block(in_channels, out_channels, kernel_size, padding, normalization):
        padding = kernel_size // 2
        if normalization:
            layers = [SpectralNorm(nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding))]
        else:
            layers = [nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)]
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def get_config(self):
        config_str = 'Discriminator Configuration:\n'
        for i, layer in enumerate(self.features):
            config_str += f'Feature Block {i}: {layer}\n'
        config_str += f'Classifier: {self.classifier}\n'
        return config_str

# Example usage
if __name__ == '__main__':
    discriminator = Discriminator(in_channels=3, max_features=32, min_features=32, num_blocks=5, kernel_size=3, padding=0, normalization=False)
    print(discriminator.get_config())
