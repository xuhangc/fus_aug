import os
import random
import numpy as np
import torch
from models.styleganxl import StyleGANXLGenerator, StyleGANXLDiscriminator
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
from data import NPZDataLoader


def set_seed(seed=3407):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Loss functions
def generator_loss(fake_pred):
    return -torch.mean(fake_pred)


def discriminator_loss(real_pred, fake_pred):
    return torch.mean(F.relu(1 - real_pred)) + torch.mean(F.relu(1 + fake_pred))

# R1 gradient penalty


def r1_penalty(real_pred, real_img):
    grad_real = torch.autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True
    )[0]
    grad_penalty = grad_real.pow(2).reshape(
        grad_real.shape[0], -1).sum(1).mean()
    return grad_penalty


# Training function
def train_stylegan_xl(session, batch_size=4, num_epochs=100, lr=0.002, device="cuda"):
    # Initialize dataset and dataloader
    dataset = NPZDataLoader(f'{session}_train.npz')
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=4)

    # Initialize generator and discriminator
    generator = StyleGANXLGenerator().to(device)
    discriminator = StyleGANXLDiscriminator().to(device)

    # Initialize optimizers
    g_optim = optim.Adam(generator.parameters(), lr=lr, betas=(0.0, 0.99))
    d_optim = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.0, 0.99))

    # Create output directory
    os.makedirs("StyleGANXL", exist_ok=True)

    # Calculate mean latent vector for truncation
    with torch.no_grad():
        mean_latent = generator.mean_latent(n_latent=10000)

    # Training loop
    for epoch in range(num_epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for fus_images, labels in pbar:
            fus_images = fus_images.to(device)
            labels = labels.to(device)
            batch_size = fus_images.shape[0]

            # Train discriminator
            discriminator.zero_grad()

            # Real images
            real_pred = discriminator(fus_images, labels)

            # Generate fake images
            z = torch.randn(batch_size, 512, device=device)
            fake_images = generator(
                z, labels, truncation=0.7, truncation_latent=mean_latent)
            fake_pred = discriminator(fake_images.detach(), labels)

            # Discriminator loss
            d_loss = discriminator_loss(real_pred, fake_pred)

            # R1 regularization
            real_images_requires_grad = fus_images.requires_grad_(True)
            real_pred_for_penalty = discriminator(
                real_images_requires_grad, labels)
            r1_grad_penalty = r1_penalty(
                real_pred_for_penalty, real_images_requires_grad)

            d_loss = d_loss + 10 * r1_grad_penalty  # Weight for R1 penalty

            d_loss.backward()
            d_optim.step()

            # Train generator
            generator.zero_grad()

            # Generate new fake images
            z = torch.randn(batch_size, 512, device=device)
            fake_images = generator(
                z, labels, truncation=0.7, truncation_latent=mean_latent)
            fake_pred = discriminator(fake_images, labels)

            # Generator loss
            g_loss = generator_loss(fake_pred)

            g_loss.backward()
            g_optim.step()

            # Update progress bar
            pbar.set_postfix({
                "G_loss": g_loss.item(),
                "D_loss": d_loss.item()
            })

        # Save model checkpoints
        torch.save(generator.state_dict(),
                       f"StyleGANXL/{session}_generator_epoch_{epoch+1}.pth")

        # Generate and save sample images
        with torch.no_grad():
            sample_z = torch.randn(16, 512, device=device)
            sample_labels = torch.randint(0, 2, (16, 1), device=device)
            sample_images = generator(
                sample_z, sample_labels, truncation=0.7, truncation_latent=mean_latent)

            save_image(
                sample_images,
                f"StyleGANXL/samples_{epoch+1}.png",
                nrow=4,
                normalize=True,
                value_range=(-1, 1),
            )


# Main execution
if __name__ == "__main__":
    # Set random seed for reproducibility
    set_seed(42)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    session = 'S2'

    # Train the model
    train_stylegan_xl(
        session=session,  # Update with your file path
        batch_size=4,
        num_epochs=100,
        lr=0.002,
        device=device
    )
