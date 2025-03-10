import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Basic convolutional block with optional batch normalization and dropout."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 use_bn=True, dropout_rate=0.0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, stride, padding, bias=not use_bn)
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout2d(
            dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class DownBlock(nn.Module):
    """Downsampling block for UNet-like architecture."""

    def __init__(self, in_channels, out_channels, dropout_rate=0.0):
        super(DownBlock, self).__init__()
        self.conv1 = ConvBlock(in_channels, out_channels,
                               dropout_rate=dropout_rate)
        self.conv2 = ConvBlock(out_channels, out_channels,
                               dropout_rate=dropout_rate)
        self.downsample = nn.MaxPool2d(2)

    def forward(self, x):
        skip = self.conv1(x)
        skip = self.conv2(skip)
        down = self.downsample(skip)
        return down, skip


class UpBlock(nn.Module):
    """Upsampling block for UNet-like architecture."""

    def __init__(self, in_channels, out_channels, dropout_rate=0.0):
        super(UpBlock, self).__init__()
        self.upsample = nn.ConvTranspose2d(
            in_channels, in_channels//2, kernel_size=2, stride=2)
        self.conv1 = ConvBlock(in_channels, out_channels,
                               dropout_rate=dropout_rate)
        self.conv2 = ConvBlock(out_channels, out_channels,
                               dropout_rate=dropout_rate)

    def forward(self, x, skip_connection):
        x = self.upsample(x)
        # Ensure spatial dimensions match
        diffY = skip_connection.size()[2] - x.size()[2]
        diffX = skip_connection.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX //
                  2, diffY // 2, diffY - diffY // 2])

        x = torch.cat([skip_connection, x], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class ClassEmbedding(nn.Module):
    """Embeds class labels for conditioning."""

    def __init__(self, num_classes, embedding_dim):
        super(ClassEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_classes, embedding_dim)

    def forward(self, label):
        # Ensure label is the right shape
        label = label.view(-1)  # Flatten from [B, 1] to [B]
        return self.embedding(label)


class UNetGenerator(nn.Module):
    """UNet-based generator architecture with class conditioning."""

    def __init__(self, in_channels=1, initial_filters=64, num_classes=10, embedding_dim=128,
                 dropout_rate=0.0, noise_dim=100):
        super(UNetGenerator, self).__init__()

        self.noise_dim = noise_dim

        # Class embedding
        self.class_embedding = ClassEmbedding(num_classes, embedding_dim)

        # Initial projection of noise + class embedding
        self.noise_projection = nn.Linear(
            noise_dim + embedding_dim, 128 * 8 * 8)

        # Encoder
        self.down1 = DownBlock(in_channels, initial_filters, dropout_rate)
        self.down2 = DownBlock(
            initial_filters, initial_filters*2, dropout_rate)
        self.down3 = DownBlock(
            initial_filters*2, initial_filters*4, dropout_rate)
        self.down4 = DownBlock(
            initial_filters*4, initial_filters*8, dropout_rate)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            ConvBlock(initial_filters*8 + 128, initial_filters *
                      16, dropout_rate=dropout_rate),
            ConvBlock(initial_filters*16, initial_filters *
                      16, dropout_rate=dropout_rate)
        )

        # Decoder
        self.up1 = UpBlock(initial_filters*16, initial_filters*8, dropout_rate)
        self.up2 = UpBlock(initial_filters*8, initial_filters*4, dropout_rate)
        self.up3 = UpBlock(initial_filters*4, initial_filters*2, dropout_rate)
        self.up4 = UpBlock(initial_filters*2, initial_filters, dropout_rate)

        # Output
        self.final_conv = nn.Conv2d(
            initial_filters, in_channels, kernel_size=1)
        self.final_activation = nn.Tanh()

    def forward(self, x, label, noise=None):
        # Get class embedding
        class_embed = self.class_embedding(label)  # [B, embedding_dim]

        # Generate noise if not provided
        if noise is None:
            noise = torch.randn(x.size(0), self.noise_dim, device=x.device)

        # Combine noise and class embedding
        noise_class = torch.cat([noise, class_embed], dim=1)

        # Project and reshape to spatial feature map
        noise_map = self.noise_projection(noise_class)
        noise_map = noise_map.view(-1, 128, 8, 8)

        # Encoder
        d1, skip1 = self.down1(x)
        d2, skip2 = self.down2(d1)
        d3, skip3 = self.down3(d2)
        d4, skip4 = self.down4(d3)

        # Upsample noise map to match d4's spatial dimensions
        noise_map = F.interpolate(
            noise_map, size=d4.shape[2:], mode='bilinear', align_corners=False)

        # Bottleneck with class conditioning
        bottleneck_input = torch.cat([d4, noise_map], dim=1)
        bottleneck = self.bottleneck(bottleneck_input)

        # Decoder
        u1 = self.up1(bottleneck, skip4)
        u2 = self.up2(u1, skip3)
        u3 = self.up3(u2, skip2)
        u4 = self.up4(u3, skip1)

        # Output
        out = self.final_conv(u4)
        out = self.final_activation(out)

        return out


class ConvDiscBlock(nn.Module):
    """Convolutional block for the discriminator."""

    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1,
                 use_bn=True, use_activation=True):
        super(ConvDiscBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, stride, padding, bias=not use_bn)
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.activation = nn.LeakyReLU(
            0.2, inplace=True) if use_activation else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class Discriminator(nn.Module):
    """Discriminator with class conditioning that outputs a single scalar per image."""

    def __init__(self, in_channels=1, initial_filters=64, num_classes=10, embedding_dim=128):
        super(Discriminator, self).__init__()

        # Class embedding
        self.class_embedding = ClassEmbedding(num_classes, embedding_dim)

        # Main convolutional blocks
        self.conv1 = ConvDiscBlock(in_channels, initial_filters, use_bn=False)
        self.conv2 = ConvDiscBlock(initial_filters, initial_filters*2)
        self.conv3 = ConvDiscBlock(initial_filters*2, initial_filters*4)
        self.conv4 = ConvDiscBlock(initial_filters*4, initial_filters*8)

        # Calculate the size of the feature map after convolutions
        # For 128x128 input, after 4 stride-2 convs: 128 -> 64 -> 32 -> 16 -> 8
        feature_size = 8

        # Class conditioning
        self.class_projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(embedding_dim, initial_filters*8)
        )

        # Flatten and linear layers
        self.flatten = nn.Flatten()

        # Calculate input features for the linear layer
        # The feature map has dimensions [initial_filters*8, feature_size, feature_size]
        # Plus we add the projected class embedding (initial_filters*8)
        linear_input_size = initial_filters*8 * \
            feature_size * feature_size + initial_filters*8

        self.linear1 = nn.Linear(linear_input_size, 512)
        self.linear2 = nn.Linear(512, 1)

    def forward(self, x, label):
        # Get class embedding
        class_embed = self.class_embedding(label)  # [B, embedding_dim]

        # Convolutional features
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)  # [B, initial_filters*8, H/16, W/16]

        # Flatten spatial dimensions
        x_flat = self.flatten(x)  # [B, initial_filters*8 * H/16 * W/16]

        # Project class embedding
        class_projection = self.class_projection(
            class_embed)  # [B, initial_filters*8]

        # Concatenate flattened features with class projection
        combined = torch.cat([x_flat, class_projection], dim=1)

        # Final dense layers
        x = F.leaky_relu(self.linear1(combined), 0.2)
        x = self.linear2(x)  # [B, 1]

        return x


class UNetGAN(nn.Module):
    """Complete UNet GAN model with generator and discriminator."""

    def __init__(self, in_channels=1, initial_filters=64, num_classes=10,
                 noise_dim=100, embedding_dim=128, dropout_rate=0.0):
        super(UNetGAN, self).__init__()

        self.generator = UNetGenerator(
            in_channels=in_channels,
            initial_filters=initial_filters,
            num_classes=num_classes,
            embedding_dim=embedding_dim,
            dropout_rate=dropout_rate,
            noise_dim=noise_dim
        )

        self.discriminator = Discriminator(
            in_channels=in_channels,
            initial_filters=initial_filters,
            num_classes=num_classes,
            embedding_dim=embedding_dim
        )

    def generate(self, x, label, noise=None):
        return self.generator(x, label, noise)

    def discriminate(self, x, label):
        return self.discriminator(x, label)


# Example usage
def test_unetgan():
    # Create model instance
    model = UNetGAN(in_channels=1, num_classes=10)

    # Create sample input
    batch_size = 4
    input_shape = (batch_size, 1, 128, 128)
    label_shape = (batch_size, 1)

    x = torch.randn(input_shape)
    labels = torch.randint(0, 10, label_shape)
    noise = torch.randn(batch_size, 100)

    # Test generator
    generated = model.generate(x, labels, noise)
    print(f"Generator output shape: {generated.shape}")

    # Test discriminator
    disc_output = model.discriminate(x, labels)
    print(f"Discriminator output shape: {disc_output.shape}")

    return model


# Training functions
# def train_unetgan(model, train_loader, num_epochs=100, lr=0.0002, beta1=0.5, beta2=0.999):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = model.to(device)

#     # Define optimizers
#     optimizer_G = torch.optim.Adam(
#         model.generator.parameters(), lr=lr, betas=(beta1, beta2))
#     optimizer_D = torch.optim.Adam(
#         model.discriminator.parameters(), lr=lr, betas=(beta1, beta2))

#     # Loss functions
#     adversarial_loss = nn.BCEWithLogitsLoss()
#     l1_loss = nn.L1Loss()

#     # Training loop
#     for epoch in range(num_epochs):
#         for i, (real_images, labels) in enumerate(train_loader):
#             batch_size = real_images.size(0)
#             real_images = real_images.to(device)
#             labels = labels.to(device)

#             # Real and fake labels
#             real_labels = torch.ones(batch_size, 1).to(device)  # [B, 1]
#             fake_labels = torch.zeros(batch_size, 1).to(device)  # [B, 1]

#             # ---------------------
#             #  Train Discriminator
#             # ---------------------
#             optimizer_D.zero_grad()

#             # Generate fake images
#             noise = torch.randn(batch_size, 100).to(device)
#             fake_images = model.generate(real_images, labels, noise)

#             # Discriminate real images
#             real_outputs = model.discriminate(real_images, labels)
#             real_loss = adversarial_loss(real_outputs, real_labels)

#             # Discriminate fake images
#             fake_outputs = model.discriminate(fake_images.detach(), labels)
#             fake_loss = adversarial_loss(fake_outputs, fake_labels)

#             # Total discriminator loss
#             d_loss = real_loss + fake_loss
#             d_loss.backward()
#             optimizer_D.step()

#             # -----------------
#             #  Train Generator
#             # -----------------
#             optimizer_G.zero_grad()

#             # Generate fake images again (since they might have changed)
#             fake_images = model.generate(real_images, labels, noise)

#             # Try to fool the discriminator
#             fake_outputs = model.discriminate(fake_images, labels)
#             g_adversarial_loss = adversarial_loss(fake_outputs, real_labels)

#             # L1 loss for reconstruction
#             g_l1_loss = l1_loss(fake_images, real_images) * \
#                 100.0  # Weight for reconstruction

#             # Total generator loss
#             g_loss = g_adversarial_loss + g_l1_loss
#             g_loss.backward()
#             optimizer_G.step()

#             # Print progress
#             if i % 50 == 0:
#                 print(f"[Epoch {epoch}/{num_epochs}] [Batch {i}] "
#                       f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")

if __name__ == '__main__':
    test_unetgan()
    # train_unetgan(model, train_loader)