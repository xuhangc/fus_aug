import torch
from torch import nn
from typing import List, Iterator


def initialize_weights(modules: Iterator[nn.Module]) -> None:
    for module in modules:
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight)
            module.weight.data *= 0.1
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.normal_(module.weight, 1.0, 0.02)
            module.weight.data *= 0.1
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight)
            module.weight.data *= 0.1
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)


class Generator(nn.Module):
    def __init__(self, image_size: int = 128, channels: int = 1, num_classes: int = 2, latent_dim: int = 100) -> None:
        """Implementation of the Vanilla GAN model.

        Args:
            image_size (int, optional): Size of the generated square image (height = width). Default is 28 (e.g., for MNIST).
            channels (int, optional): Number of channels in the generated image. Default is 1 (grayscale image).
            num_classes (int, optional): Number of classes for conditional generation. Default is 10.
            latent_dim (int, optional): Dimension of the latent noise vector. Default is 100.
        """
        super().__init__()
        self.image_size = image_size
        self.channels = channels

        # Generate a random matrix of size (num_classes, num_classes)
        self.label_embedding = nn.Embedding(num_classes, num_classes)

        self.backbone = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, True),

            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, True),

            nn.Linear(1024, int(channels * image_size * image_size)),
            nn.Tanh()
        )

        # Initializing all neural network weights.
        initialize_weights(self.modules())

    def forward(self, x: torch.Tensor, labels: List) -> torch.Tensor:
        """Forward pass of the Vanilla GAN model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, latent).
            labels (List): List of labels for conditional generation.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, channels, image_size, image_size).
        """
        conditional_inputs = torch.cat([x, self.label_embedding(labels.flatten())], -1)
        x = self.backbone(conditional_inputs)
        return x.reshape(x.size(0), self.channels, self.image_size, self.image_size)
    

class Discriminator(nn.Module):
    def __init__(self, image_size: int = 128, channels: int = 1, dropout: float = 0.5, num_classes: int = 2) -> None:
        """Discriminator model architecture.

        Args:
            image_size (int, optional): Size of the generated square image (height = width). Default is 28 (e.g., for MNIST).
            channels (int, optional): Number of channels in the generated image. Default is 1 (grayscale image).
            dropout (float, optional): Dropout rate. Default is 0.5.
            num_classes (int, optional): Number of classes for conditional generation. Default is 10.
        """
        super().__init__()
        # Embedding layer for the labels.
        self.label_embedding = nn.Embedding(num_classes, num_classes)

        self.backbone = nn.Sequential(
            nn.Linear(channels * image_size * image_size + num_classes, 512),
            nn.LeakyReLU(0.2, True),

            nn.Linear(512, 512),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(dropout),

            nn.Linear(512, 512),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(dropout),

            nn.Linear(512, 1),
            nn.Sigmoid()
        )

        # Initializing all neural network weights.
        initialize_weights(self.modules())

    def forward(self, x: torch.Tensor, labels: List) -> torch.Tensor:
        """Forward pass of the Vanilla GAN model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, latent).
            labels (List): List of labels for conditional generation.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, channels, image_size, image_size).
        """
        x = torch.flatten(x, 1)
        label_embedding = self.label_embedding(labels.flatten())
        x = torch.cat([x, label_embedding], dim=-1)
        return self.backbone(x)
    

if __name__ == '__main__':
    # Define the generator model
    gen = Generator(image_size=128, channels=1, num_classes=2, latent_dim=100)
    print(gen)

    # Define the discriminator model
    disc = Discriminator(image_size=128, channels=1, dropout=0.5, num_classes=2)
    print(disc)

    # Generate random noise
    noise = torch.randn(2, 100)
    labels = torch.randint(0, 2, (2, 1))
    print(noise.shape, labels.shape)

    # Generate a fake image
    fake_image = gen(noise, labels)
    print(fake_image.shape, labels.shape)

    # Pass the fake image to the discriminator
    validity = disc(fake_image, labels)
    print(validity.shape)