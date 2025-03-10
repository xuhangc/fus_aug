import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Skip connection if in_channels != out_channels
        self.skip = nn.Identity()
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        residual = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out


class Encoder(nn.Module):
    def __init__(self, in_channels=1, hidden_channels=128, num_residual_blocks=2, embedding_dim=64):
        super(Encoder, self).__init__()

        # Initial convolution
        self.conv1 = nn.Conv2d(in_channels, hidden_channels //
                               2, kernel_size=4, stride=2, padding=1)

        # Residual blocks and downsampling
        self.resblock1 = ResidualBlock(hidden_channels//2, hidden_channels//2)
        self.conv2 = nn.Conv2d(
            hidden_channels//2, hidden_channels, kernel_size=4, stride=2, padding=1)

        self.resblock2 = ResidualBlock(hidden_channels, hidden_channels)
        self.conv3 = nn.Conv2d(
            hidden_channels, hidden_channels, kernel_size=4, stride=2, padding=1)

        self.resblock3 = ResidualBlock(hidden_channels, hidden_channels)
        self.conv4 = nn.Conv2d(
            hidden_channels, hidden_channels, kernel_size=4, stride=2, padding=1)

        # Additional residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_channels, hidden_channels) for _ in range(num_residual_blocks)
        ])

        # Final convolution to get embedding_dim channels
        self.conv_out = nn.Conv2d(
            hidden_channels, embedding_dim, kernel_size=3, padding=1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.resblock1(x)

        x = self.relu(self.conv2(x))
        x = self.resblock2(x)

        x = self.relu(self.conv3(x))
        x = self.resblock3(x)

        x = self.relu(self.conv4(x))

        for block in self.residual_blocks:
            x = block(x)

        x = self.conv_out(x)

        return x


class ConditionalDecoder(nn.Module):
    def __init__(self, out_channels=1, hidden_channels=128, num_residual_blocks=2, embedding_dim=64, num_classes=10):
        super(ConditionalDecoder, self).__init__()

        # Class embedding
        self.class_embedding = nn.Embedding(num_classes, hidden_channels)

        # Initial convolution from embedding_dim to hidden_channels
        self.conv_in = nn.Conv2d(
            embedding_dim, hidden_channels, kernel_size=3, padding=1)

        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_channels, hidden_channels) for _ in range(num_residual_blocks)
        ])

        # Upsampling layers
        self.upsample1 = nn.ConvTranspose2d(
            hidden_channels, hidden_channels, kernel_size=4, stride=2, padding=1)
        self.resblock1 = ResidualBlock(hidden_channels, hidden_channels)

        self.upsample2 = nn.ConvTranspose2d(
            hidden_channels, hidden_channels, kernel_size=4, stride=2, padding=1)
        self.resblock2 = ResidualBlock(hidden_channels, hidden_channels)

        self.upsample3 = nn.ConvTranspose2d(
            hidden_channels, hidden_channels//2, kernel_size=4, stride=2, padding=1)
        self.resblock3 = ResidualBlock(hidden_channels//2, hidden_channels//2)

        self.upsample4 = nn.ConvTranspose2d(
            hidden_channels//2, hidden_channels//4, kernel_size=4, stride=2, padding=1)
        self.resblock4 = ResidualBlock(hidden_channels//4, hidden_channels//4)

        # Final convolution
        self.conv_out = nn.Conv2d(
            hidden_channels//4, out_channels, kernel_size=3, padding=1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, class_labels):
        batch_size = x.shape[0]

        # Process class condition
        class_emb = self.class_embedding(
            class_labels.squeeze(1))  # B, hidden_channels
        # B, hidden_channels, 1, 1
        class_emb = class_emb.view(batch_size, -1, 1, 1)

        # Initial processing
        x = self.relu(self.conv_in(x))  # B, hidden_channels, 8, 8

        # Incorporate class information through addition
        x = x + class_emb.expand(-1, -1, x.size(2), x.size(3))

        # Apply residual blocks
        for block in self.residual_blocks:
            x = block(x)

        # Upsampling path
        x = self.relu(self.upsample1(x))
        x = self.resblock1(x)

        x = self.relu(self.upsample2(x))
        x = self.resblock2(x)

        x = self.relu(self.upsample3(x))
        x = self.resblock3(x)

        x = self.relu(self.upsample4(x))
        x = self.resblock4(x)

        # Final convolution
        x = self.conv_out(x)

        return x


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super(VectorQuantizer, self).__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        # Initialize the embeddings
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 /
                                            num_embeddings, 1.0 / num_embeddings)

    def forward(self, inputs):
        # Convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self.embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True)
                     + torch.sum(self.embedding.weight**2, dim=1)
                     - 2 * torch.matmul(flat_input, self.embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(
            encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(
            encodings, self.embedding.weight).view(input_shape)

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()

        # Convert quantized from BHWC -> BCHW
        quantized = quantized.permute(0, 3, 1, 2).contiguous()

        return quantized, loss, encoding_indices.view(input_shape[0], input_shape[1], input_shape[2])


class ClassConditionalVQVAE(nn.Module):
    def __init__(self, in_channels=1, hidden_channels=128, num_residual_blocks=2,
                 embedding_dim=64, num_embeddings=512, commitment_cost=0.25, num_classes=2):
        super(ClassConditionalVQVAE, self).__init__()

        self.encoder = Encoder(in_channels, hidden_channels,
                               num_residual_blocks, embedding_dim)
        self.vq = VectorQuantizer(
            num_embeddings, embedding_dim, commitment_cost)
        self.decoder = ConditionalDecoder(
            in_channels, hidden_channels, num_residual_blocks, embedding_dim, num_classes)

    def forward(self, x, class_labels):
        z = self.encoder(x)
        z_q, vq_loss, indices = self.vq(z)
        x_recon = self.decoder(z_q, class_labels)

        return x_recon, vq_loss, indices

    def encode(self, x):
        z = self.encoder(x)
        z_q, _, indices = self.vq(z)
        return z_q, indices

    def decode(self, indices, class_labels):
        batch_size = indices.shape[0]
        h_dim = w_dim = indices.shape[1]  # Assuming square latent space

        # Convert indices to one-hot
        one_hot = torch.zeros(batch_size, self.vq.num_embeddings,
                              h_dim, w_dim, device=indices.device)
        one_hot = one_hot.scatter_(1, indices.unsqueeze(1), 1)

        # Multiply one-hot by embedding weights
        z_q = torch.matmul(one_hot.permute(0, 2, 3, 1),
                           self.vq.embedding.weight)
        z_q = z_q.permute(0, 3, 1, 2)

        # Decode
        x_recon = self.decoder(z_q, class_labels)
        return x_recon

# Example usage


# def main():
#     # Create a sample input
#     batch_size = 1
#     input_shape = (batch_size, 1, 128, 128)
#     class_labels = torch.randint(0, 2, (batch_size, 1))

#     # Create the model
#     model = ClassConditionalVQVAE(in_channels=1, hidden_channels=128,
#                                   num_residual_blocks=2, embedding_dim=64,
#                                   num_embeddings=512, num_classes=2)

#     # Forward pass
#     x = torch.randn(input_shape)
#     x_recon, vq_loss, indices = model(x, class_labels)

#     # Print shapes
#     print(f"Input shape: {x.shape}")
#     print(f"Reconstruction shape: {x_recon.shape}")
#     print(f"VQ loss: {vq_loss.item()}")
#     print(f"Indices shape: {indices.shape}")

#     # Test encoding and decoding
#     z_q, indices = model.encode(x)
#     x_recon2 = model.decode(indices, class_labels)
#     print(f"Encoding shape: {z_q.shape}")
#     print(f"Indices shape: {indices.shape}")
#     print(f"Decoding shape: {x_recon2.shape}")


# if __name__ == "__main__":
#     main()
