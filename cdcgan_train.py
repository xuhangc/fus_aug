import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import torch
from data import NPZDataLoader
from models.cdcgan import Generator, Discriminator
import torch.optim as optim
import random
import numpy as np


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


if __name__ == "__main__":
    # Set random seed for reproducibility
    set_seed(42)

    session = 'S2'
    model = 'CDCGAN'

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

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

    # Hyperparameters
    latent_dim = 100
    image_size = 128
    channels = 1
    num_classes = 2
    lr = 0.0002
    beta1 = 0.5
    beta2 = 0.999
    num_epochs = 100

    # Initialize the generator and discriminator
    generator = Generator(image_size=image_size, channels=channels,
                          num_classes=num_classes, latent_dim=latent_dim).to(device)
    discriminator = Discriminator(image_size=image_size, channels=channels, num_classes=num_classes).to(device)

    # Initialize optimizers
    optimizer_G = optim.Adam(generator.parameters(),
                             lr=lr, betas=(beta1, beta2))
    optimizer_D = optim.Adam(discriminator.parameters(),
                             lr=lr, betas=(beta1, beta2))

    # Loss function
    criterion = torch.nn.BCEWithLogitsLoss()

    # Create directories for saving results
    os.makedirs("generated_images", exist_ok=True)
    os.makedirs(model, exist_ok=True)

    # Training loop
    print("Starting training...")
    for epoch in range(num_epochs):
        generator.train()
        discriminator.train()

        running_d_loss = 0.0
        running_g_loss = 0.0

        progress_bar = tqdm(enumerate(train_dataloader),
                            total=len(train_dataloader))

        for i, (real_images, labels) in progress_bar:
            # Move data to device
            real_images = real_images.float().to(device)
            labels = labels.long().to(device)
            batch_size = real_images.size(0)

            # Ensure images are in [-1, 1] range (for tanh)
            if real_images.max() > 1.0:
                real_images = (real_images / 255.0) * 2 - 1
            elif real_images.max() <= 1.0 and real_images.min() >= 0:
                real_images = real_images * 2 - 1

            # Create labels for real and fake samples
            real_targets = torch.ones(batch_size, 1, device=device)
            fake_targets = torch.zeros(batch_size, 1, device=device)

            # ---------------------
            # Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            # Real images forward pass
            outputs_real = discriminator(real_images, labels)
            d_loss_real = criterion(outputs_real, real_targets)

            # Fake images forward pass
            z = torch.randn(batch_size, latent_dim, device=device)
            fake_images = generator(z, labels)
            outputs_fake = discriminator(fake_images.detach(), labels)
            d_loss_fake = criterion(outputs_fake, fake_targets)

            # Calculate total discriminator loss
            d_loss = (d_loss_real + d_loss_fake) / 2
            d_loss.backward()
            optimizer_D.step()

            # ---------------------
            # Train Generator
            # ---------------------
            optimizer_G.zero_grad()

            # Generate new fake images
            z = torch.randn(batch_size, latent_dim, device=device)
            fake_images = generator(z, labels)
            outputs = discriminator(fake_images, labels)

            # Calculate generator loss (want discriminator to think fakes are real)
            g_loss = criterion(outputs, real_targets)
            g_loss.backward()
            optimizer_G.step()

            # Update running losses
            running_d_loss += d_loss.item()
            running_g_loss += g_loss.item()

            # Update progress bar
            progress_bar.set_postfix({
                'D Loss': f"{d_loss.item():.4f}",
                'G Loss': f"{g_loss.item():.4f}"
            })

        # Print epoch statistics
        avg_d_loss = running_d_loss / len(train_dataloader)
        avg_g_loss = running_g_loss / len(train_dataloader)
        print(
            f"Epoch {epoch+1}/{num_epochs} - D Loss: {avg_d_loss:.4f}, G Loss: {avg_g_loss:.4f}")

        # Generate and save sample images
        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == num_epochs - 1:
            generator.eval()
            with torch.no_grad():
                # Generate samples for both classes
                n_samples = 5
                z = torch.randn(n_samples * 2, latent_dim, device=device)

                # Generate class 0 samples
                class_0_labels = torch.zeros(
                    n_samples, dtype=torch.long, device=device)
                class_0_samples = generator(z[:n_samples], class_0_labels)

                # Generate class 1 samples
                class_1_labels = torch.ones(
                    n_samples, dtype=torch.long, device=device)
                class_1_samples = generator(z[n_samples:], class_1_labels)

                # Combine samples
                samples = torch.cat([class_0_samples, class_1_samples], dim=0)
                samples = samples.cpu()

                # Convert from [-1, 1] to [0, 1] for visualization
                samples = (samples + 1) / 2.0

                # Create grid of images
                fig, axs = plt.subplots(2, n_samples, figsize=(12, 5))

                for j in range(n_samples):
                    # Class 0 samples
                    axs[0, j].imshow(samples[j, 0].numpy(), cmap='gray')
                    axs[0, j].axis('off')
                    if j == 0:
                        axs[0, j].set_title('Class 0')

                    # Class 1 samples
                    axs[1, j].imshow(
                        samples[n_samples + j, 0].numpy(), cmap='gray')
                    axs[1, j].axis('off')
                    if j == 0:
                        axs[1, j].set_title('Class 1')

                plt.tight_layout()
                plt.savefig(f"generated_images/{session}_{model}_epoch_{epoch+1}.png")
                plt.close()

        # Save models periodically
        if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
            torch.save(generator.state_dict(),
                       f"{model}/{session}_generator_epoch_{epoch+1}.pth")
            torch.save(discriminator.state_dict(),
                       f"{model}/{session}_discriminator_epoch_{epoch+1}.pth")

    print("Training completed!")

    # Function to generate samples from trained model
    def generate_samples(gen_model, n_per_class=10):
        gen_model.eval()
        with torch.no_grad():
            # Generate latent vectors
            z = torch.randn(n_per_class * 2, latent_dim, device=device)

            # Generate class 0 samples
            class_0_labels = torch.zeros(
                n_per_class, dtype=torch.long, device=device)
            class_0_samples = gen_model(z[:n_per_class], class_0_labels)

            # Generate class 1 samples
            class_1_labels = torch.ones(
                n_per_class, dtype=torch.long, device=device)
            class_1_samples = gen_model(z[n_per_class:], class_1_labels)

            # Combine samples and convert to [0, 1] range
            samples = torch.cat([class_0_samples, class_1_samples], dim=0)
            samples = samples.cpu()
            samples = (samples + 1) / 2.0

            # Plot results
            fig, axs = plt.subplots(2, n_per_class, figsize=(15, 6))

            for j in range(n_per_class):
                # Class 0 samples
                axs[0, j].imshow(samples[j, 0].numpy(), cmap='gray')
                axs[0, j].axis('off')
                if j == 0:
                    axs[0, j].set_title('Class 0')

                # Class 1 samples
                axs[1, j].imshow(
                    samples[n_per_class + j, 0].numpy(), cmap='gray')
                axs[1, j].axis('off')
                if j == 0:
                    axs[1, j].set_title('Class 1')

            plt.tight_layout()
            plt.savefig("final_generated_samples.png")
            # plt.show()

    # Generate final samples
    generate_samples(generator)
