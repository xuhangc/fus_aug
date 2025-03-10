import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Spectral Normalization for stabilization


def spectral_norm(module):
    return nn.utils.spectral_norm(module)

# Self-Attention Block


class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query_conv = spectral_norm(nn.Conv2d(in_dim, in_dim // 8, 1))
        self.key_conv = spectral_norm(nn.Conv2d(in_dim, in_dim // 8, 1))
        self.value_conv = spectral_norm(nn.Conv2d(in_dim, in_dim, 1))
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, width, height = x.size()

        proj_query = self.query_conv(x).view(
            batch_size, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch_size, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)

        proj_value = self.value_conv(x).view(batch_size, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)

        out = self.gamma * out + x
        return out

# Conditional Batch Normalization


class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_features, num_classes):
        super(ConditionalBatchNorm2d, self).__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.embed = spectral_norm(nn.Embedding(num_classes, num_features * 2))
        self.embed.weight.data[:, :num_features].normal_(
            1, 0.02)  # Initialize scale to 1
        # Initialize bias to 0
        self.embed.weight.data[:, num_features:].zero_()

    def forward(self, x, y):
        out = self.bn(x)
        gamma, beta = self.embed(y.flatten()).chunk(2, dim=1)
        out = gamma.view(-1, self.num_features, 1, 1) * out + \
            beta.view(-1, self.num_features, 1, 1)
        return out

# Residual Block for Generator


class ResBlockGenerator(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes):
        super(ResBlockGenerator, self).__init__()

        self.cbn1 = ConditionalBatchNorm2d(in_channels, num_classes)
        self.conv1 = spectral_norm(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1))
        self.cbn2 = ConditionalBatchNorm2d(out_channels, num_classes)
        self.conv2 = spectral_norm(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1))

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.shortcut = nn.Sequential(
            self.upsample,
            spectral_norm(nn.Conv2d(in_channels, out_channels, 1, 1, 0))
        )

    def forward(self, x, class_label):
        h = F.relu(self.cbn1(x, class_label))
        h = self.upsample(h)
        h = self.conv1(h)
        h = F.relu(self.cbn2(h, class_label))
        h = self.conv2(h)

        return h + self.shortcut(x)

# Residual Block for Discriminator


class ResBlockDiscriminator(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=True):
        super(ResBlockDiscriminator, self).__init__()

        self.conv1 = spectral_norm(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1))
        self.conv2 = spectral_norm(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1))

        self.skip_connection = spectral_norm(
            nn.Conv2d(in_channels, out_channels, 1, 1, 0))
        self.downsample = downsample

    def forward(self, x):
        h = F.relu(x)
        h = self.conv1(h)
        h = F.relu(h)
        h = self.conv2(h)

        if self.downsample:
            h = F.avg_pool2d(h, 2)
            x = F.avg_pool2d(x, 2)

        return h + self.skip_connection(x)

# BigGAN Generator


class Generator(nn.Module):
    def __init__(self, z_dim=128, num_classes=1000, channel_width=128, num_attention_heads=1):
        super(Generator, self).__init__()

        self.z_dim = z_dim
        self.num_classes = num_classes

        # Initial dense layer
        self.linear = spectral_norm(
            nn.Linear(z_dim, 4 * 4 * 16 * channel_width))

        # Residual blocks with upsampling
        self.res_blocks = nn.ModuleList([
            ResBlockGenerator(16 * channel_width, 16 *
                              channel_width, num_classes),  # 4x4 -> 8x8
            ResBlockGenerator(16 * channel_width, 8 * \
                              channel_width, num_classes),   # 8x8 -> 16x16
            ResBlockGenerator(8 * channel_width, 4 * \
                              channel_width, num_classes),    # 16x16 -> 32x32
            ResBlockGenerator(4 * channel_width, 2 * \
                              channel_width, num_classes),    # 32x32 -> 64x64
            ResBlockGenerator(2 * channel_width, channel_width,
                              num_classes)         # 64x64 -> 128x128
        ])

        # Self-attention after reaching a certain resolution (e.g., 64x64)
        self.attention = SelfAttention(2 * channel_width)

        # Final layers
        self.bn = nn.BatchNorm2d(channel_width)
        self.conv_out = spectral_norm(nn.Conv2d(channel_width, 1, 3, 1, 1))

    def forward(self, z, class_labels):
        # Convert input class_labels from [B, 1] to [B]
        class_labels = class_labels.squeeze()

        # Initial projection and reshaping
        h = self.linear(z)
        h = h.view(h.size(0), -1, 4, 4)

        # Residual blocks with attention
        for i, res_block in enumerate(self.res_blocks):
            h = res_block(h, class_labels)
            # Apply self-attention at 64x64 resolution
            if i == 3:  # After the 4th block, resolution is 64x64
                h = self.attention(h)

        # Final layers
        h = F.relu(self.bn(h))
        h = self.conv_out(h)
        h = torch.tanh(h)

        return h

# BigGAN Discriminator


class Discriminator(nn.Module):
    def __init__(self, num_classes=1000, channel_width=128):
        super(Discriminator, self).__init__()

        # Initial conv layer
        self.conv_in = spectral_norm(nn.Conv2d(1, channel_width, 3, 1, 1))

        # Residual blocks with downsampling
        self.res_blocks = nn.ModuleList([
            # 128x128 -> 64x64
            ResBlockDiscriminator(channel_width, 2 * channel_width),
            ResBlockDiscriminator(2 * channel_width, 4 * \
                                  channel_width),         # 64x64 -> 32x32
            ResBlockDiscriminator(4 * channel_width, 8 * \
                                  channel_width),         # 32x32 -> 16x16
            ResBlockDiscriminator(8 * channel_width, 16 * \
                                  channel_width),        # 16x16 -> 8x8
            ResBlockDiscriminator(16 * channel_width,
                                  16 * channel_width)        # 8x8 -> 4x4
        ])

        # Self-attention at 64x64 resolution
        self.attention = SelfAttention(2 * channel_width)

        # Final layers
        self.fc = spectral_norm(nn.Linear(16 * channel_width * 4 * 4, 1))

        # Class embedding
        self.embed = spectral_norm(nn.Embedding(
            num_classes, 16 * channel_width * 4 * 4))

    def forward(self, x, class_labels):
        # Convert input class_labels from [B, 1] to [B]
        class_labels = class_labels.squeeze()

        h = self.conv_in(x)

        # Apply residual blocks with attention
        for i, res_block in enumerate(self.res_blocks):
            h = res_block(h)
            # Apply self-attention at 64x64 resolution
            if i == 0:  # After the 1st block, resolution is 64x64
                h = self.attention(h)

        h = F.relu(h)
        h = h.view(h.size(0), -1)

        # Unconditional output
        output = self.fc(h)

        # Conditional output with class embedding
        class_embedding = self.embed(class_labels)
        output += torch.sum(class_embedding * h, dim=1, keepdim=True)

        return output

# Complete BigGAN model


class BigGAN(nn.Module):
    def __init__(self, z_dim=128, num_classes=1000, channel_width=128):
        super(BigGAN, self).__init__()

        self.z_dim = z_dim
        self.generator = Generator(z_dim, num_classes, channel_width)
        self.discriminator = Discriminator(num_classes, channel_width)

    def generate(self, z, class_labels):
        return self.generator(z, class_labels)

    def discriminate(self, x, class_labels):
        return self.discriminator(x, class_labels)

    # Truncation trick for improved sample quality
    def generate_truncated(self, z, class_labels, truncation=0.7):
        z = torch.clamp(z, -truncation, truncation)
        return self.generator(z, class_labels)


def initialize_biggan(num_classes=1000, z_dim=128, channel_width=128):
    model = BigGAN(z_dim=z_dim, num_classes=num_classes,
                   channel_width=channel_width)
    return model


if __name__ == '__main__':
    # Use actual number of classes
    biggan = initialize_biggan(num_classes=2)
    batch_size = 2
    z = torch.randn(batch_size, 128)  # Latent vectors
    class_labels = torch.randint(0, 2, (batch_size, 1))  # Class labels

    generated_images = biggan.generate(
        z, class_labels)  # Shape: [B, 1, 128, 128]
    
    disc_output = biggan.discriminate(
        generated_images, class_labels)  # Shape: [B, 1]

    print(generated_images.shape, disc_output.shape)