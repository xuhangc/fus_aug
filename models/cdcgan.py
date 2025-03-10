import torch
from torch import nn
import torch.nn.functional as F
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
        """Implementation of the Conditional GAN model using Convolutional Neural Networks.

        Args:
            image_size (int, optional): Size of the generated square image (height = width). Default is 28.
            channels (int, optional): Number of channels in the generated image. Default is 1 (grayscale image).
            num_classes (int, optional): Number of classes for conditional generation. Default is 10.
            latent_dim (int, optional): Dimension of the latent noise vector. Default is 100.
        """
        super().__init__()
        self.image_size = image_size
        self.channels = channels
        self.num_classes = num_classes
        self.latent_dim = latent_dim

        # Embedding layer for the labels.
        self.label_embedding = nn.Sequential(
            nn.Linear(self.latent_dim + self.num_classes, 4 * 4 * 512),
            nn.LeakyReLU(0.2, True),
        )
        # self.label_embedding = nn.Sequential(
        #     nn.Linear(self.latent_dim + self.num_classes,
        #               8 * 8 * self.latent_dim),
        #     nn.LeakyReLU(0.2, True),
        # )

        # self.backbone = nn.Sequential(
        #     nn.ConvTranspose2d(self.latent_dim, 512, 5, bias=False),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(True),

        #     nn.ConvTranspose2d(512, 512, 4, bias=False),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(True),

        #     nn.ConvTranspose2d(512, 256, 4, bias=False),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(True),

        #     nn.ConvTranspose2d(256, 256, 4, bias=False),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(True),

        #     nn.ConvTranspose2d(256, 128, 4, bias=False),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(True),

        #     nn.ConvTranspose2d(128, 64, 3, bias=False),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(True),

        #     nn.ConvTranspose2d(64, self.channels, 4),

        #     nn.Tanh()
        # )
        self.backbone = nn.Sequential(
            # Starting with 4x4
            nn.ConvTranspose2d(512, 256, kernel_size=4,
                               stride=2, padding=1, bias=False),  # 8x8
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, kernel_size=4,
                               stride=2, padding=1, bias=False),  # 16x16
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2,
                               padding=1, bias=False),   # 32x32
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2,
                               padding=1, bias=False),    # 64x64
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.ConvTranspose2d(32, self.channels, kernel_size=4,
                               stride=2, padding=1),    # 128x128
            nn.Tanh()
        )

        # Initializing all neural network weights.
        initialize_weights(self.modules())

    def forward(self, x: torch.Tensor, labels: List = None) -> torch.Tensor:
        """Forward pass of the Deep Conditional GAN model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, latent).
            labels (List, optional): List of labels for conditional generation. Default is None.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, channels, image_size, image_size).
        """
        if x.dim() != 2:
            raise ValueError(
                f"Expected input tensor 'x' to have 2 dimensions, but got {x.dim()}.")

        if labels is None:
            raise ValueError(
                "Labels must be provided for conditional generation.")
        labels = F.one_hot(labels.flatten(), num_classes=self.num_classes).float()
        x = torch.cat([x, labels], 1)
        x = self.label_embedding(x).reshape(-1, 512, 4, 4)
        return self.backbone(x)


class Discriminator(nn.Module):
    def __init__(self, image_size: int = 28, channels: int = 1, num_classes: int = 2) -> None:
        """Discriminator model architecture.

        Args:
            image_size (int, optional): Size of the generated square image (height = width). Default is 28
            channels (int, optional): Number of channels in the generated image. Default is 1 (grayscale image).
            num_classes (int, optional): Number of classes for conditional generation. Default is 10.
        """
        super().__init__()
        self.image_size = image_size
        self.channels = channels
        self.num_classes = num_classes

        # Embedding layer for the labels.
        self.label_embedding = nn.Sequential(
            nn.Linear(self.num_classes, int(
                self.channels * self.image_size * self.image_size)),
            nn.LeakyReLU(0.2, True),
        )

        self.backbone = nn.Sequential(
            # Input: (channels + 1, 128, 128)
            nn.Conv2d(self.channels + 1, 64, 3, stride=2, padding=1),  # 64x64
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 32x32
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d(2),  # 16x16

            nn.Conv2d(128, 256, 3, stride=2, padding=1),  # 8x8
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d(2),  # 4x4

            nn.Conv2d(256, 512, 3, stride=1, padding=1),  # 4x4
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(512, self.channels, 4, stride=1, padding=0),  # 1x1
        )

        # Initializing all neural network weights.
        initialize_weights(self.modules())


    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(
                f"Expected input tensor 'x' to have 4 dimensions, but got {x.dim()}.")
        if labels is None:
            raise ValueError("Labels must be provided for conditional generation.")

        # One-hot encode labels and pass through embedding
        # [[Fix: Add one-hot encoding]]
        labels = F.one_hot(labels, num_classes=self.num_classes).float()
        label_embedding = self.label_embedding(
            labels).reshape(-1, self.channels, self.image_size, self.image_size)

        # Concatenate image and label embedding along channel dimension
        x = torch.cat([x, label_embedding], 1)
        x = self.backbone(x)
        return torch.flatten(x, 1)


if __name__ == "__main__":
    # Define the generator model
    gen = Generator(image_size=128, channels=1, num_classes=2, latent_dim=100)
    print(gen)

    # Define the discriminator model
    disc = Discriminator(image_size=128, channels=1, num_classes=2)
    print(disc)

    # Generate random noise
    noise = torch.randn(2, 100)
    labels = torch.randint(0, 2, (2, 1))
    print(noise.shape, labels.shape)

    # Generate a fake image
    fake_image = gen(noise, labels)
    print(fake_image.shape)

    # Pass the fake image to the discriminator
    validity = disc(fake_image, labels)
    print(validity.shape)