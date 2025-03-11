import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_channels=1, label_embedding_dim=64):
        super().__init__()
        self.label_embedding = nn.Embedding(
            2, label_embedding_dim)  # Binary labels (0/1)
        self.conv1 = nn.Conv2d(
            input_channels, 64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64 + label_embedding_dim,
                               128, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
        self.norm = nn.BatchNorm2d(512)

    def forward(self, x, label):
        # Embed label and tile to match spatial dimensions
        B, _, H, W = x.shape
        label_embed = self.label_embedding(label).view(
            B, -1, 1, 1).repeat(1, 1, H//2, W//2)

        x = F.relu(self.conv1(x))
        x = torch.cat([x, label_embed], dim=1)  # Inject label after first conv
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.norm(self.conv4(x)))
        return x


class Decoder(nn.Module):
    def __init__(self, latent_dim=512, label_embedding_dim=64):
        super().__init__()
        self.label_embedding = nn.Embedding(2, label_embedding_dim)
        self.conv1 = nn.ConvTranspose2d(
            latent_dim + label_embedding_dim, 256, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.ConvTranspose2d(
            256, 128, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.ConvTranspose2d(
            128, 64, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.ConvTranspose2d(
            64, 1, kernel_size=4, stride=2, padding=1)
        self.norm = nn.BatchNorm2d(256)

    def forward(self, z, label):
        # Embed label and tile to match spatial dimensions
        B, C, H, W = z.shape
        label_embed = self.label_embedding(
            label).view(B, -1, 1, 1).repeat(1, 1, H, W)

        x = torch.cat([z, label_embed], dim=1)  # Inject label at decoder input
        x = F.relu(self.norm(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.sigmoid(self.conv4(x))  # Output in [0, 1]
        return x


class Codebook(nn.Module):
    def __init__(self, codebook_size=1024, latent_dim=512):
        super().__init__()
        self.embedding = nn.Embedding(codebook_size, latent_dim)
        self.embedding.weight.data.uniform_(-1.0 /
                                            codebook_size, 1.0 / codebook_size)

    def forward(self, z):
        z_flattened = z.permute(0, 2, 3, 1).contiguous().view(-1, z.shape[1])
        distances = (z_flattened.pow(2).sum(1, keepdim=True)
                     - 2 * torch.matmul(z_flattened, self.embedding.weight.T)
                     + self.embedding.weight.pow(2).sum(1, keepdim=True).T)
        min_indices = torch.argmin(distances, dim=1)
        quantized = self.embedding(min_indices).view(z.shape)
        return quantized, min_indices.view(z.shape[0], -1)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, 1, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1, 4, 1, 1)
        )

    def forward(self, x):
        return self.net(x)


class VQGAN(nn.Module):
    def __init__(self, codebook_size=1024):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.codebook = Codebook(codebook_size)
        self.discriminator = Discriminator()
        self.quant_conv = nn.Conv2d(512, 512, 1)  # Post-encoder 1x1 conv

    def forward(self, x, label):
        z = self.encoder(x, label)
        z = self.quant_conv(z)
        quantized, indices = self.codebook(z)
        recon = self.decoder(quantized, label)
        return recon, indices, z, quantized  # Return z and quantized

    def compute_losses(self, x, label):
        recon, indices, z, quantized = self(x, label)  # Capture z and quantized

        # Reconstruction loss
        recon_loss = F.l1_loss(recon, x)
        
        # Adversarial loss (hinge loss)
        real_logits = self.discriminator(x)
        fake_logits = self.discriminator(recon.detach())
        d_loss = torch.mean(F.relu(1. - real_logits) + F.relu(1. + fake_logits))
        g_loss = -torch.mean(fake_logits)
        
        # Codebook loss (commitment loss)
        codebook_loss = F.mse_loss(z, quantized.detach())  # Correct order
        return recon_loss, d_loss, g_loss, codebook_loss
    

if __name__ == "__main__":
    # Example usage
    model = VQGAN()
    x = torch.randn(2, 1, 128, 128)
    labels = torch.randint(0, 2, (2, 1))
    recon, _, _, _ = model(x, labels)
    print(recon.shape)
    print(model.discriminator(recon).shape)
    model.compute_losses(x, labels)
