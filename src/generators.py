import torch
import torch.nn as nn
import torch.nn.functional as F
from discriminators import Discriminator

class Generator(nn.Module):
    def __init__(self, in_channels, max_features, min_features, num_blocks, kernel_size, padding):
        super().__init__()
        layers = []
        features = in_channels

        # Building the downsampling layers
        for _ in range(num_blocks):
            layers += [
                nn.Conv2d(features, min(max_features, features * 2), kernel_size, stride=2, padding=padding, bias=False),
                nn.BatchNorm2d(min(max_features, features * 2)),
                nn.ReLU(inplace=True)
            ]
            features = min(max_features, features * 2)

        # Building the upsampling layers
        for _ in range(num_blocks):
            layers += [
                nn.ConvTranspose2d(features, features // 2, kernel_size, stride=2, padding=padding, output_padding=1, bias=False),
                nn.BatchNorm2d(features // 2),
                nn.ReLU(inplace=True)
            ]
            features = features // 2

        # Output layer
        layers += [nn.Conv2d(features, in_channels, kernel_size, stride=1, padding=padding), nn.Tanh()]

        self.model = nn.Sequential(*layers)

    def forward(self, x, noise):
        # Combining the noise and the real image
        combined_input = torch.cat([x, noise], dim=1)
        return self.model(combined_input)

    @staticmethod
    def _generate_noise(tensor_like, device, repeat=False):
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
    
    @staticmethod
    def _upscale(image_tensor, new_height, new_width):
        """
        Upscale an image tensor to a new height and width.

        Parameters:
        image_tensor (torch.Tensor): A tensor representing the image with shape [1, 3, height, width].
        new_height (int): The desired height of the upscaled image.
        new_width (int): The desired width of the upscaled image.

        Returns:
        torch.Tensor: An upscaled image tensor with shape [1, 3, new_height, new_width].
        """
        # Resize the image to the new dimensions
        upscaled_image = F.interpolate(image_tensor, size=(new_height, new_width), mode='bilinear', align_corners=False)

        return upscaled_image

    def get_config(self):
        pass
    
if __name__ == '__main__':
    pass
