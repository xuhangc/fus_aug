import torch
import os
import random
import numpy as np
from models.unetgan import UNetGAN
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


# Training functions
def train_unetgan(model, train_loader, session, num_epochs=100, lr=0.0002, beta1=0.5, beta2=0.999):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Define optimizers
    optimizer_G = torch.optim.Adam(
        model.generator.parameters(), lr=lr, betas=(beta1, beta2))
    optimizer_D = torch.optim.Adam(
        model.discriminator.parameters(), lr=lr, betas=(beta1, beta2))

    # Loss functions
    adversarial_loss = torch.nn.BCEWithLogitsLoss()
    l1_loss = torch.nn.L1Loss()

    # Training loop
    for epoch in range(num_epochs):
        for i, (real_images, labels) in enumerate(train_loader):
            batch_size = real_images.size(0)
            real_images = real_images.to(device)
            labels = labels.to(device)

            # Real and fake labels
            real_labels = torch.ones(batch_size, 1).to(device)  # [B, 1]
            fake_labels = torch.zeros(batch_size, 1).to(device)  # [B, 1]

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            # Generate fake images
            noise = torch.randn(batch_size, 100).to(device)
            fake_images = model.generate(real_images, labels, noise)

            # Discriminate real images
            real_outputs = model.discriminate(real_images, labels)
            real_loss = adversarial_loss(real_outputs, real_labels)

            # Discriminate fake images
            fake_outputs = model.discriminate(fake_images.detach(), labels)
            fake_loss = adversarial_loss(fake_outputs, fake_labels)

            # Total discriminator loss
            d_loss = real_loss + fake_loss
            d_loss.backward()
            optimizer_D.step()

            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()

            # Generate fake images again (since they might have changed)
            fake_images = model.generate(real_images, labels, noise)

            # Try to fool the discriminator
            fake_outputs = model.discriminate(fake_images, labels)
            g_adversarial_loss = adversarial_loss(fake_outputs, real_labels)

            # L1 loss for reconstruction
            g_l1_loss = l1_loss(fake_images, real_images) * \
                100.0  # Weight for reconstruction

            # Total generator loss
            g_loss = g_adversarial_loss + g_l1_loss
            g_loss.backward()
            optimizer_G.step()

            # Print progress
            if i % 50 == 0:
                print(f"[Epoch {epoch}/{num_epochs}] [Batch {i}] "
                      f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")
        # Save models periodically
        if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
            torch.save(model.state_dict(),
                       f"UNetGAN/{session}_generator_epoch_{epoch+1}.pth")


if __name__ == '__main__':
    set_seed(42)

    session = 'S2'

    os.makedirs('UNetGAN', exist_ok=True)

    # Load the data
    print("Loading and preparing data...")
    train_dataset = NPZDataLoader(f'{session}_train.npz')
    val_dataset = NPZDataLoader(f'{session}_test.npz')

    # Create data loaders
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=16, num_workers=4)

    print(
        f"Training on {len(train_dataset)} samples, validating on {len(val_dataset)} samples")
    
    # Initialize U-Net GAN model
    unetgan = UNetGAN(num_classes=2)

    # Train the U-Net GAN model
    train_unetgan(unetgan, train_dataloader, session, num_epochs=100)
