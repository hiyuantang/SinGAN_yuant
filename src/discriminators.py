import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm as SpectralNorm

class Discriminator(nn.Module):
    def __init__(self, in_channels, max_features, min_features, num_blocks, kernel_size, padding, normalization=True):
        """
        Constructs the Discriminator model with flexible configurations.

        Parameters:
        in_channels (int): Number of input channels.
        max_features (int): Maximum number of features for convolutional layers.
        min_features (int): Minimum number of features for convolutional layers.
        num_blocks (int): Number of blocks in the model.
        kernel_size (int): Kernel size for convolutional layers.
        padding (int): Padding for convolutional layers.
        normalization (bool): Flag to apply spectral normalization.
        """
        super(Discriminator, self).__init__()

        # Constructing the sequential blocks
        blocks = [self._create_block(in_channels=in_channels, out_channels=max_features, kernel_size=kernel_size, padding=padding, normalization=normalization)]

        f = max_features
        for i in range(num_blocks - 2):
            f_next = max(min_features, f // 2)
            blocks.append(self._create_block(in_channels=f, out_channels=f_next, kernel_size=kernel_size, padding=padding, normalization=normalization))
            f = f_next

        self.features = nn.Sequential(*blocks)

        # Classifier layer
        self.classifier = nn.Conv2d(in_channels=f, out_channels=1, kernel_size=kernel_size, padding=padding)

    @staticmethod
    def _create_block(in_channels, out_channels, kernel_size, padding, normalization):
        """
        Create a convolutional block with optional spectral normalization and LeakyReLU activation.

        Parameters:
        in_channels (int): Number of input channels for the block.
        out_channels (int): Number of output channels for the block.
        kernel_size (int): Kernel size for the convolutional layer.
        padding (int): Padding for the convolutional layer.
        normalization (bool): Apply spectral normalization if True.

        Returns:
        nn.Sequential: The constructed convolutional block.
        """
        if normalization:
            layers = [SpectralNorm(nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding))]
        else:
            layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding)]
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass of the Discriminator.

        Parameters:
        x (Tensor): The input tensor to the model.

        Returns:
        Tensor: The output tensor after processing by the model.
        """
        x = self.features(x)
        x = self.classifier(x)
        return x

    def get_config(self):
        """
        Retrieves the configuration of the Discriminator.

        Returns:
        str: A string representation of the model's configuration.
        """
        config_str = "Discriminator Configuration:\n"
        for i, layer in enumerate(self.features):
            config_str += f"Feature Block {i}: {layer}\n"
        config_str += f"Classifier: {self.classifier}\n"
        return config_str

# Example usage
if __name__ == '__main__':
    discriminator = Discriminator(in_channels=3, max_features=32, min_features=32, num_blocks=5, kernel_size=3, padding=1, normalization=False)
    print(discriminator.get_config())
