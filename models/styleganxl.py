import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# StyleGAN-XL Architecture

# Equalized Linear Layer for StyleGAN
class EqualizedLinear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1.0, activation=None):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.lr_mul = lr_mul
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))
        else:
            self.bias = None

        self.activation = activation
        self.scale = (1 / math.sqrt(in_dim)) * lr_mul

    def forward(self, x):
        out = F.linear(x, self.weight * self.scale, bias=self.bias *
                       self.lr_mul if self.bias is not None else None)

        if self.activation:
            out = self.activation(out)

        return out

# Equalized Convolutional Layer for StyleGAN


class EqualizedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.weight = nn.Parameter(torch.randn(
            out_channels, in_channels, kernel_size, kernel_size))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None

        self.scale = 1 / math.sqrt(in_channels * kernel_size ** 2)

    def forward(self, x):
        weight = self.weight * self.scale
        return F.conv2d(x, weight, self.bias, self.stride, self.padding)

# Mapping Network for StyleGAN-XL
# Mapping Network for StyleGAN-XL


class MappingNetwork(nn.Module):
    def __init__(self, z_dim=512, w_dim=512, num_layers=8, label_dim=1):
        super().__init__()
        layers = []
        # Embedding for binary labels
        self.label_emb = nn.Embedding(2, label_dim)  # Binary labels (0 or 1)

        # Combine noise and label
        input_dim = z_dim + label_dim

        for i in range(num_layers):
            layers.append(EqualizedLinear(
                input_dim if i == 0 else w_dim,
                w_dim,
                lr_mul=0.01,
                activation=nn.LeakyReLU(0.2)
            ))

        self.mapping = nn.Sequential(*layers)

    def forward(self, z, label):
        # Ensure label is the right shape and type for embedding
        # Ensure it's within range [0,1] for binary labels
        label = label.long().clamp(0, 1)

        # Handle different label shapes
        if label.dim() > 1:
            label = label.squeeze()

        # Handle single-sample case
        if label.dim() == 0:
            label = label.unsqueeze(0)

        # Process the label
        label_embed = self.label_emb(label)

        # Concatenate noise and label embedding
        z_with_label = torch.cat([z, label_embed], dim=1)

        # Map to W space
        w = self.mapping(z_with_label)
        return w

# Modulation and Demodulation for StyleGAN


class ModulatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, style_dim, demodulate=True, upsample=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.demodulate = demodulate
        self.upsample = upsample

        self.scale = 1 / math.sqrt(in_channels * kernel_size ** 2)
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(torch.randn(
            1, out_channels, in_channels, kernel_size, kernel_size))
        self.modulation = EqualizedLinear(style_dim, in_channels, bias_init=1)

    def forward(self, x, style):
        batch, in_channel, height, width = x.shape

        # Style modulation
        style = self.modulation(style).view(batch, 1, in_channel, 1, 1)
        weight = self.weight * style * self.scale

        # Demodulation
        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channels, 1, 1, 1)

        weight = weight.view(
            batch * self.out_channels, in_channel, self.kernel_size, self.kernel_size
        )

        if self.upsample:
            x = F.interpolate(x, scale_factor=2,
                              mode="bilinear", align_corners=False)

        x = x.reshape(1, batch * in_channel, height *
                      (2 if self.upsample else 1), width * (2 if self.upsample else 1))

        # Group convolution operation
        out = F.conv2d(x, weight, padding=self.padding, groups=batch)
        _, _, height, width = out.shape
        out = out.view(batch, self.out_channels, height, width)

        return out

# Noise injection module


class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, image, noise=None):
        if noise is None:
            batch, _, height, width = image.shape
            noise = torch.randn(batch, 1, height, width, device=image.device)

        return image + self.weight * noise

# StyleGAN Generator Block


class StyleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, style_dim, upsample=True):
        super().__init__()
        self.upsample = upsample
        self.conv1 = ModulatedConv2d(
            in_channels, out_channels, 3, style_dim, upsample=upsample)
        self.noise1 = NoiseInjection()
        self.act1 = nn.LeakyReLU(0.2)
        self.conv2 = ModulatedConv2d(out_channels, out_channels, 3, style_dim)
        self.noise2 = NoiseInjection()
        self.act2 = nn.LeakyReLU(0.2)

    def forward(self, x, style):
        out = self.conv1(x, style)
        out = self.noise1(out)
        out = self.act1(out)
        out = self.conv2(out, style)
        out = self.noise2(out)
        out = self.act2(out)
        return out

# ToRGB layer


class ToRGB(nn.Module):
    def __init__(self, in_channels, style_dim, upsample=True):
        super().__init__()
        self.upsample = upsample
        # Output is 1 channel (grayscale)
        self.conv = ModulatedConv2d(in_channels, 1, 1, style_dim)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x, style, skip=None):
        out = self.conv(x, style)
        out = out + self.bias.view(1, 1, 1, 1)

        if skip is not None:
            if self.upsample:
                skip = F.interpolate(skip, scale_factor=2,
                                     mode="bilinear", align_corners=False)

            out = out + skip

        return out

# Complete StyleGAN-XL Generator
# Complete StyleGAN-XL Generator


class StyleGANXLGenerator(nn.Module):
    def __init__(
        self,
        style_dim=512,
        num_layers=8,
        starting_size=4,
        img_size=128,
        label_dim=1
    ):
        super().__init__()

        self.style_dim = style_dim
        self.starting_size = starting_size

        self.mapping = MappingNetwork(
            style_dim, style_dim, num_layers, label_dim)

        # Number of style blocks needed
        log_size = int(math.log(img_size, 2))
        log_starting = int(math.log(starting_size, 2))
        self.num_blocks = log_size - log_starting + 1

        # Initial constant input
        self.input = nn.Parameter(torch.randn(
            1, 512, starting_size, starting_size))

        # Style blocks and ToRGB layers
        self.blocks = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()

        # Channel definitions for each resolution
        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256,
            128: 128,
        }

        in_channels = channels[starting_size]

        for i in range(self.num_blocks):
            res = starting_size * (2 ** i)
            next_res = res * 2

            # Make sure we don't exceed the maximum resolution
            if next_res <= img_size:
                out_channels = channels[next_res]
            else:
                out_channels = in_channels

            self.blocks.append(StyleBlock(
                in_channels,
                out_channels,
                style_dim,
                upsample=(i != 0)  # No upsampling for first block
            ))

            self.to_rgbs.append(ToRGB(
                out_channels,
                style_dim,
                upsample=(i != 0)  # No upsampling for first block
            ))

            in_channels = out_channels

    def forward(self, z, label, truncation=1.0, truncation_latent=None):
        # Map noise and label to W space
        styles = self.mapping(z, label)

        # Apply truncation trick
        if truncation < 1.0:
            if truncation_latent is None:
                truncation_latent = self.mean_latent
            styles = truncation_latent + truncation * \
                (styles - truncation_latent)

        # Start with constant input
        x = self.input.repeat(styles.shape[0], 1, 1, 1)

        # Apply style blocks and generate RGB output
        skip = None

        for i, (block, to_rgb) in enumerate(zip(self.blocks, self.to_rgbs)):
            x = block(x, styles)
            skip = to_rgb(x, styles, skip)

        return skip

    def mean_latent(self, n_latent=10000):
        latent_in = torch.randn(
            n_latent, self.style_dim, device=self.input.device
        )
        labels = torch.randint(0, 2, (n_latent, 1), device=self.input.device)
        latent = self.mapping(latent_in, labels).mean(0, keepdim=True)
        return latent

# Discriminator Block


class DiscriminatorBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=True):
        super().__init__()

        self.conv1 = EqualizedConv2d(in_channels, in_channels, 3, padding=1)
        self.act1 = nn.LeakyReLU(0.2)
        self.conv2 = EqualizedConv2d(in_channels, out_channels, 3, padding=1)
        self.act2 = nn.LeakyReLU(0.2)

        self.downsample = downsample

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.act2(out)

        if self.downsample:
            out = F.avg_pool2d(out, 2)

        return out

# From RGB Layer for Discriminator


class FromRGB(nn.Module):
    def __init__(self, out_channels):
        super().__init__()

        # Input is 1 channel (grayscale)
        self.conv = EqualizedConv2d(1, out_channels, 1)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.act(self.conv(x))

# StyleGAN-XL Discriminator with Label Conditioning
# StyleGAN-XL Discriminator with Label Conditioning


class StyleGANXLDiscriminator(nn.Module):
    def __init__(self, img_size=128):
        super().__init__()

        # Channel definitions for each resolution
        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256,
            128: 128,
        }

        log_size = int(math.log(img_size, 2))

        self.from_rgbs = nn.ModuleList()
        self.blocks = nn.ModuleList()

        for i in range(log_size, 2, -1):
            res = 2 ** i
            self.from_rgbs.append(FromRGB(channels[res]))

            self.blocks.append(
                DiscriminatorBlock(
                    channels[res],
                    channels[res // 2],
                    downsample=(i != 3)
                )
            )

        # Label embedding for conditioning
        self.label_embedding = nn.Embedding(2, 512)  # Binary labels (0 or 1)

        # Calculate the final feature size
        self.final_res = 4
        self.final_channels = channels[4]

        # Final layers
        self.final_conv = EqualizedConv2d(
            channels[4] + 1, channels[4], 3, padding=1)  # +1 for minibatch std
        self.final_act = nn.LeakyReLU(0.2)
        self.final_linear = EqualizedLinear(
            channels[4] * self.final_res * self.final_res, 1)
        self.label_proj = EqualizedLinear(512, 1)

    def forward(self, x, label):
        batch_size = x.shape[0]
        
        # Ensure label is properly formatted
        label = label.long().clamp(0, 1)
        
        # Handle different label shapes
        if label.dim() > 1:
            label = label.squeeze()
            
        # Handle single-sample case
        if label.dim() == 0:
            label = label.unsqueeze(0)
        
        out = None
        
        for from_rgb, block in zip(self.from_rgbs, self.blocks):
            if out is None:
                out = from_rgb(x)
            out = block(out)
        
        # Use the fixed minibatch_std function
        out = minibatch_std(out)
        
        out = self.final_conv(out)
        out = self.final_act(out)
        
        # Ensure the feature map size is as expected
        expected_size = self.final_channels * self.final_res * self.final_res
        out = F.adaptive_avg_pool2d(out, (self.final_res, self.final_res))
        out = out.view(batch_size, -1)
        
        out_img = self.final_linear(out)
        
        # Label conditioning
        label_embed = self.label_embedding(label)
        out_label = self.label_proj(label_embed)
        
        return out_img + out_label


# Fixed minibatch standard deviation implementation
def minibatch_std(x, eps=1e-8):
    batch_size, _, height, width = x.shape
    
    # [B, C, H, W] -> [B, H, W, C]
    y = x.permute(0, 2, 3, 1)
    
    # [B, H, W, C] -> [B, H*W*C]
    y = y.reshape(batch_size, -1)
    
    # Calculate standard deviation across batch dimension
    std = torch.sqrt(y.var(dim=0, unbiased=False) + eps)
    
    # Get mean standard deviation
    std = std.mean().view(1, 1, 1, 1)
    
    # Expand to same shape as input
    std = std.repeat(batch_size, 1, height, width)
    
    # Concatenate to input
    return torch.cat([x, std], dim=1)


if __name__ == "__main__":
    # Test the generator
    gen = StyleGANXLGenerator()
    z = torch.randn(2, 512)
    label = torch.randint(0, 10, (2, 1))
    img = gen(z, label)
    print(img.shape)

    # Test the discriminator
    disc = StyleGANXLDiscriminator()
    out = disc(img, label)
    print(out.shape)