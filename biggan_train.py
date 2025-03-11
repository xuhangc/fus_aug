import torch
import os
import random
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from data import NPZDataLoader
from models.biggan import Generator, Discriminator


# Set random seeds for reproducibility
def set_seed(seed=3407):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Device configuration
def train(dataloader, session):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    z_dim = 128
    num_classes = 2  # Update based on your dataset
    lr_g = 0.0002
    lr_d = 0.0002
    beta1 = 0.0
    beta2 = 0.999
    num_epochs = 100

    # Initialize models
    generator = Generator(z_dim=z_dim, num_classes=num_classes).to(device)
    discriminator = Discriminator(num_classes=num_classes).to(device)

    # Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=lr_g, betas=(beta1, beta2))
    optimizer_D = optim.Adam(discriminator.parameters(),
                             lr=lr_d, betas=(beta1, beta2))

    # Training loop
    for epoch in range(num_epochs):
        for i, (imgs, labels) in enumerate(dataloader):
            # Move data to device
            real_imgs = imgs.to(device)
            labels = labels.to(device)

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            # Sample noise and generate fake images
            z = torch.randn(imgs.size(0), z_dim).to(device)
            fake_imgs = generator(z, labels)

            # Calculate loss
            real_validity = discriminator(real_imgs, labels)
            fake_validity = discriminator(fake_imgs.detach(), labels)

            # Hinge loss
            d_loss = torch.mean(F.relu(1. - real_validity)) + \
                torch.mean(F.relu(1. + fake_validity))
            d_loss.backward()
            optimizer_D.step()

            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()

            # Generate new fake images
            z = torch.randn(imgs.size(0), z_dim).to(device)
            fake_imgs = generator(z, labels)
            fake_validity = discriminator(fake_imgs, labels)

            g_loss = -torch.mean(fake_validity)
            g_loss.backward()
            optimizer_G.step()

            # Print progress
            if i % 100 == 0:
                print(f"[Epoch {epoch}/{num_epochs}] [Batch {i}/{len(dataloader)}] "
                      f"D_loss: {d_loss.item():.4f}", f"G_loss: {g_loss.item():.4f}")
                
        # Save models periodically
        if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
            torch.save(generator.state_dict(),
                       f"BIGGAN/{session}_generator_epoch_{epoch+1}.pth")
            torch.save(discriminator.state_dict(),
                       f"BIGGAN/{session}_discriminator_epoch_{epoch+1}.pth")

                

if __name__ == '__main__':
    set_seed(42)

    session = 'S1'

    os.makedirs('BIGGAN', exist_ok=True)

    # Load the data
    print("Loading and preparing data...")
    train_dataset = NPZDataLoader(f'{session}_train.npz')
    val_dataset = NPZDataLoader(f'{session}_test.npz')

    # Create data loaders
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=2, shuffle=True, num_workers=16)

    train(train_dataloader, session)
    print('Done training.')