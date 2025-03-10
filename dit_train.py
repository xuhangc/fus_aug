import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from diffusers import DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from accelerate import Accelerator
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from einops import rearrange, repeat

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
batch_size = 16
num_epochs = 100
learning_rate = 1e-4
num_train_timesteps = 1000
beta_start = 0.0001
beta_end = 0.02
img_size = 128
in_channels = 1
embed_dim = 1024
num_heads = 16
depth = 24
mlp_ratio = 4.0
save_model_epochs = 10
save_image_epochs = 5
patch_size = 8  # Size of image patches

# Dataset class


class FunctionalUltrasoundDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        """
        Args:
            data: Numpy array of ultrasound images (N, 1, 128, 128)
            labels: Binary labels (N, 1)
            transform: Optional transform to be applied on a sample
        """
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        # Normalize to [-1, 1]
        image = (image / 127.5) - 1.0

        return {"image": image, "label": label}

# TODO: Load your actual data here
# For demonstration purposes, creating dummy data


def load_dummy_data(num_samples=1000):
    # Create dummy ultrasound data and binary labels
    data = np.random.rand(num_samples, 1, img_size,
                          img_size).astype(np.float32)
    labels = np.random.randint(0, 2, size=(num_samples, 1)).astype(np.float32)
    return data, labels


# Load data
# Replace with your actual data loading function
data, labels = load_dummy_data()

# Split data into train and validation sets
train_data, val_data, train_labels, val_labels = train_test_split(
    data, labels, test_size=0.2, random_state=42
)

# Create datasets
train_dataset = FunctionalUltrasoundDataset(train_data, train_labels)
val_dataset = FunctionalUltrasoundDataset(val_data, val_labels)

# Create data loaders
train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
)
val_dataloader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
)

# Initialize accelerator
accelerator = Accelerator(
    mixed_precision="fp16",
    gradient_accumulation_steps=1,
    log_with="tensorboard",
    project_dir="logs",
)

# Diffusion Transformer (DiT) implementation


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(
            half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, img_size=128, patch_size=8, in_chans=1, embed_dim=1024):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == W == self.img_size, f"Input image size ({H}*{W}) doesn't match model ({self.img_size}*{self.img_size})"

        # BCHW -> BNC
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class MLP(nn.Module):
    """MLP with GeLU activation"""

    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    """Multi-head self-attention"""

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C //
                                  self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer block with pre-norm"""

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim,
                       hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x, cond=None):
        # Apply conditioning via adaptive layer norm if provided
        if cond is not None:
            x = x + self.attn(self.norm1(x + cond))
            x = x + self.mlp(self.norm2(x + cond))
        else:
            x = x + self.attn(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
        return x


class DiT(nn.Module):
    """
    Diffusion Transformer (DiT) for ultrasound image generation
    """

    def __init__(
        self,
        img_size=128,
        patch_size=8,
        in_channels=1,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.1,
        attn_drop_rate=0.1,
        num_classes=2,
    ):
        super().__init__()

        # Image Patch Embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_channels,
            embed_dim=embed_dim,
        )
        self.num_patches = self.patch_embed.num_patches

        # Positional Embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim))

        # Time Embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(embed_dim),
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )

        # Class Embedding for binary labels
        self.class_embed = nn.Embedding(num_classes, embed_dim)

        # Transformer Blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
            )
            for _ in range(depth)
        ])

        # Final Layer Norm
        self.norm = nn.LayerNorm(embed_dim)

        # Output projection to reconstruct image
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, patch_size * patch_size * in_channels)
        )

        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def unpatchify(self, x):
        """
        x: (B, L, patch_size**2 * in_channels)
        """
        p = self.patch_embed.patch_size
        h = w = int(x.shape[1] ** 0.5)
        in_chans = self.proj[0].out_features // (p ** 2)

        # (B, L, patch_size**2 * in_channels) -> (B, h, w, patch_size, patch_size, in_channels)
        x = x.reshape(shape=(x.shape[0], h, w, p, p, in_chans))

        # (B, h, w, patch_size, patch_size, in_channels) -> (B, in_channels, h*patch_size, w*patch_size)
        x = torch.einsum('nhwpqc->nchpwq', x)
        x = x.reshape(shape=(x.shape[0], in_chans, h * p, w * p))

        return x

    def forward(self, x, timesteps, class_labels=None):
        # Get batch dimensions
        B = x.shape[0]

        # Tokenize image
        x = self.patch_embed(x)

        # Add positional encoding
        x = x + self.pos_embed

        # Time conditioning
        time_embed = self.time_embed(timesteps)
        time_embed = repeat(time_embed, 'b d -> b n d', n=x.shape[1])

        # Class conditioning
        if class_labels is not None:
            class_labels = class_labels.long().squeeze()
            class_embed = self.class_embed(class_labels)
            class_embed = repeat(class_embed, 'b d -> b n d', n=x.shape[1])
            cond = time_embed + class_embed
        else:
            cond = time_embed

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, cond)

        # Final normalization and output projection
        x = self.norm(x)
        x = self.proj(x)

        # Unpatchify to reconstruct image
        output = self.unpatchify(x)

        return output


# Initialize the DiT model
model = DiT(
    img_size=img_size,
    patch_size=patch_size,
    in_channels=in_channels,
    embed_dim=embed_dim,
    depth=depth,
    num_heads=num_heads,
    mlp_ratio=mlp_ratio,
).to(device)

# Create noise scheduler
noise_scheduler = DDPMScheduler(
    num_train_timesteps=num_train_timesteps,
    beta_start=beta_start,
    beta_end=beta_end,
    beta_schedule="scaled_linear",
    clip_sample=False,
)

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Learning rate scheduler
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=500,
    num_training_steps=len(train_dataloader) * num_epochs,
)

# Prepare everything with accelerator
model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
    model, optimizer, train_dataloader, val_dataloader, lr_scheduler
)

# Helper functions for generating and saving samples


def save_samples(model, noise_scheduler, epoch, num_samples=4, output_dir="samples"):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        # Get a batch from validation set
        batch = next(iter(val_dataloader))
        clean_images = batch["image"].to(device)[:num_samples]
        labels = batch["label"].to(device)[:num_samples]

        # Create noise and add to images
        noise = torch.randn_like(clean_images)
        timesteps = torch.ones(clean_images.shape[0], device=device).long(
        ) * (noise_scheduler.num_train_timesteps - 1)
        noisy_images = noise_scheduler.add_noise(
            clean_images, noise, timesteps)

        # Sample with the model
        image_list = []
        noisy_sample = noisy_images.clone()

        for t in reversed(range(0, noise_scheduler.num_train_timesteps, 100)):
            timesteps = torch.tensor([t], device=device).repeat(
                noisy_sample.shape[0])
            model_output = model(noisy_sample, timesteps, labels)

            # 0 is for pred_original_sample
            step_output = noise_scheduler.step(model_output, t, noisy_sample)
            noisy_sample = step_output.prev_sample
            image_list.append(noisy_sample.detach().cpu())

        # Create a grid of images
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Original image
        axes[0, 0].imshow(
            clean_images[0, 0].detach().cpu().numpy(), cmap='gray')
        axes[0, 0].set_title(f"Original, Label: {labels[0].item()}")

        # Noisy image
        axes[0, 1].imshow(
            noisy_images[0, 0].detach().cpu().numpy(), cmap='gray')
        axes[0, 1].set_title("Noisy")

        # Final denoised
        axes[0, 2].imshow(image_list[-1][0, 0].numpy(), cmap='gray')
        axes[0, 2].set_title("Denoised")

        # Intermediate steps
        for i, idx in enumerate([0, 1, 2]):
            if idx < len(image_list) - 1:
                axes[1, i].imshow(image_list[idx][0, 0].numpy(), cmap='gray')
                axes[1, i].set_title(f"Denoising Step {idx}")

        plt.tight_layout()
        plt.savefig(f"{output_dir}/sample_epoch_{epoch}.png")
        plt.close()

# Training loop


def train_loop():
    global_step = 0

    for epoch in range(num_epochs):
        # Training
        model.train()
        progress_bar = tqdm(total=len(train_dataloader),
                            disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        train_loss = 0.0

        for batch in train_dataloader:
            clean_images = batch["image"]
            labels = batch["label"]

            # Sample noise to add to the images
            noise = torch.randn_like(clean_images)
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.num_train_timesteps, (bs,), device=accelerator.device
            ).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            noisy_images = noise_scheduler.add_noise(
                clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                # Predict noise with the model (we predict the noise component)
                model_output = model(noisy_images, timesteps, labels)

                # Calculate loss (we're predicting the clean image directly)
                loss = F.mse_loss(model_output, clean_images)
                accelerator.backward(loss)

                # Clip gradients
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            train_loss += loss.detach().item()
            global_step += 1

        progress_bar.close()
        avg_train_loss = train_loss / len(train_dataloader)

        # Validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_dataloader:
                clean_images = batch["image"]
                labels = batch["label"]

                # Sample noise to add to the images
                noise = torch.randn_like(clean_images)
                bs = clean_images.shape[0]

                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0, noise_scheduler.num_train_timesteps, (bs,), device=accelerator.device
                ).long()

                # Add noise to the clean images according to the noise magnitude at each timestep
                noisy_images = noise_scheduler.add_noise(
                    clean_images, noise, timesteps)

                # Predict the clean image
                model_output = model(noisy_images, timesteps, labels)

                # Calculate loss
                loss = F.mse_loss(model_output, clean_images)
                val_loss += loss.detach().item()

        avg_val_loss = val_loss / len(val_dataloader)

        # Log and print metrics
        logs = {
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "epoch": epoch,
            "step": global_step,
        }

        accelerator.log(logs, step=global_step)
        print(
            f"Epoch {epoch} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Save model checkpoint
        if epoch % save_model_epochs == 0 or epoch == num_epochs - 1:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                # Get a copy of the model without parallel processing wrappers
                unwrapped_model = accelerator.unwrap_model(model)

                # Save model
                model_path = f"./checkpoints/dit_model_epoch_{epoch}.pt"
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                torch.save(unwrapped_model.state_dict(), model_path)

        # Generate and save sample images
        if epoch % save_image_epochs == 0 or epoch == num_epochs - 1:
            if accelerator.is_main_process:
                save_samples(
                    accelerator.unwrap_model(model),
                    noise_scheduler,
                    epoch,
                    num_samples=4
                )

# Sample generation function for image augmentation


def generate_augmented_samples(model, noise_scheduler, num_samples=1, labels=None, device="cuda"):
    """
    Generate augmented ultrasound images using the trained DiT model

    Args:
        model: Trained DiT model
        noise_scheduler: DDPM scheduler
        num_samples: Number of samples to generate
        labels: Class labels for conditional generation (binary)
        device: Device to use for generation

    Returns:
        Generated ultrasound images
    """
    model.eval()

    # Create random noise
    noise = torch.randn(num_samples, in_channels,
                        img_size, img_size).to(device)

    # If no labels provided, generate random binary labels
    if labels is None:
        labels = torch.randint(0, 2, (num_samples, 1)).to(device)

    # Initialize with noise
    sample = noise

    # Sampling loop
    for t in tqdm(reversed(range(0, noise_scheduler.num_train_timesteps)), desc="Sampling"):
        timesteps = torch.full(
            (num_samples,), t, device=device, dtype=torch.long)

        # Forward pass
        with torch.no_grad():
            model_output = model(sample, timesteps, labels)

        # Compute previous sample using scheduler
        step_output = noise_scheduler.step(model_output, t, sample)
        sample = step_output.prev_sample

    # Normalize samples to [0, 1] range
    sample = (sample + 1) / 2

    return sample


# Start training
if __name__ == "__main__":
    # Initialize accelerator
    accelerator.init_trackers("diffusion_transformer_training")

    # Create directories
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("samples", exist_ok=True)

    train_loop()

    # Final cleanup
    accelerator.end_training()

    print("Training completed!")

    # Example of generating augmented samples
    print("Generating example augmented samples...")
    model = accelerator.unwrap_model(model)
    augmented_samples = generate_augmented_samples(
        model=model,
        noise_scheduler=noise_scheduler,
        num_samples=4,
        labels=torch.tensor([[0], [1], [0], [1]], device=device),
    )

    # Save the generated samples
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(augmented_samples[i, 0].cpu().numpy(), cmap='gray')
        ax.set_title(f"Generated (Label: {i % 2})")
        ax.axis('off')

    plt.tight_layout()
    plt.savefig("samples/final_augmented_samples.png")
    plt.close()
