import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import torch.nn.functional as F
from models.cvae import CVAE
import torch
from data import NPZDataLoader
from torch.utils.data import DataLoader
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


# Define the loss function for VAE

def vae_loss_function(recon_x, x, mu, log_var, kl_weight=1.0):
    """
    Calculate VAE loss with KL annealing
    """
    # Flatten input and reconstruction for loss calculation
    x_flat = x.view(x.size(0), -1)
    recon_x_flat = recon_x.view(recon_x.size(0), -1)

    # Reconstruction loss (using binary cross entropy)
    BCE = F.binary_cross_entropy(recon_x_flat, x_flat, reduction='sum')

    # KL-divergence
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    # Total loss with KL weight
    return BCE + kl_weight * KLD, BCE, KLD


def visualize_results(model, dataloader, device, epoch, kl_weight):
    """Visualize original images, reconstructions, and generated samples"""
    model.eval()

    # Get batch of validation images
    data, labels = next(iter(dataloader))
    data = data.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        # Get reconstructions
        recon_data, mu, log_var, z = model(data, labels)

        # Generate samples
        batch_size = data.size(0)
        random_z = torch.randn(batch_size, model.latent_size).to(device)
        samples = model.inference(random_z, labels)

    # Plot results
    fig, axes = plt.subplots(3, 5, figsize=(15, 9))

    # Original images
    for i in range(5):
        if i < data.size(0):
            axes[0, i].imshow(data[i, 0].cpu().numpy(), cmap='gray')
            axes[0, i].set_title(f"Original (Class {labels[i].item()})")
            axes[0, i].axis('off')

    # Reconstructions
    for i in range(5):
        if i < recon_data.size(0):
            axes[1, i].imshow(recon_data[i, 0].cpu().numpy(), cmap='gray')
            axes[1, i].set_title(f"Reconstruction")
            axes[1, i].axis('off')

    # Generated samples
    for i in range(5):
        if i < samples.size(0):
            axes[2, i].imshow(samples[i, 0].cpu().numpy(), cmap='gray')
            axes[2, i].set_title(f"Generated (Class {labels[i].item()})")
            axes[2, i].axis('off')

    plt.suptitle(f"Epoch {epoch+1} (KL Weight: {kl_weight:.2f})")
    plt.tight_layout()
    plt.savefig(f'checkpoints/results_epoch_{epoch+1}.png')
    plt.close()


def plot_training_curves(train_losses, val_losses):
    """Plot training and validation loss curves"""
    train_total, train_recon, train_kl = zip(*train_losses)
    val_total, val_recon, val_kl = zip(*val_losses)

    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(15, 10))

    # Total loss
    plt.subplot(3, 1, 1)
    plt.plot(epochs, train_total, 'b-', label='Train Loss')
    plt.plot(epochs, val_total, 'r-', label='Validation Loss')
    plt.title('Total Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Reconstruction loss
    plt.subplot(3, 1, 2)
    plt.plot(epochs, train_recon, 'b-', label='Train Reconstruction')
    plt.plot(epochs, val_recon, 'r-', label='Validation Reconstruction')
    plt.title('Reconstruction Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # KL divergence
    plt.subplot(3, 1, 3)
    plt.plot(epochs, train_kl, 'b-', label='Train KL Divergence')
    plt.plot(epochs, val_kl, 'r-', label='Validation KL Divergence')
    plt.title('KL Divergence')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('loss_curves.png')
    plt.close()


def generate_samples(model, device, num_samples=10, num_classes=2):
    """Generate and visualize samples from random latent vectors"""
    model.eval()

    # Create figure
    fig, axes = plt.subplots(num_classes, num_samples,
                             figsize=(num_samples*2, num_classes*2))

    with torch.no_grad():
        for c in range(num_classes):
            # Generate latent vectors
            z = torch.randn(num_samples, model.latent_size).to(device)
            labels = torch.full((num_samples,), c, dtype=torch.long).to(device)

            # Generate samples
            samples = model.inference(z, labels.unsqueeze(1))

            # Display samples
            for i in range(num_samples):
                if num_classes > 1:
                    ax = axes[c, i]
                else:
                    ax = axes[i]

                ax.imshow(samples[i, 0].cpu().numpy(), cmap='gray')
                ax.set_title(f"Class {c}")
                ax.axis('off')

    plt.tight_layout()
    plt.savefig('generated_samples.png')
    plt.close()


if __name__ == "__main__":
    # Set random seed for reproducibility
    set_seed(42)

    session = 'S1'
    model = 'CVAE'

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load your functional ultrasound data here
    # Replace this with your actual data loading code
    print("Loading and preparing data...")

    train_dataset = NPZDataLoader(f'{session}_train.npz')
    val_dataset = NPZDataLoader(f'{session}_test.npz')

    # Create data loaders
    train_dataloader = DataLoader(
        train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=16, num_workers=4)

    print(
        f"Training on {len(train_dataset)} samples, validating on {len(val_dataset)} samples")

    # Train the model
    print("Starting model training...")

    VAE = CVAE().to(device)

    learning_rate = 2e-4
    num_epochs = 200

    optimizer = torch.optim.AdamW(
        VAE.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, num_epochs, eta_min=1e-6)

    # Create directory for saving models
    os.makedirs(model, exist_ok=True)

    # For tracking metrics
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    # KL annealing parameters
    kl_weight = 0.0
    kl_start = 10  # Start increasing KL weight after this many epochs
    kl_anneal_epochs = 50  # Reach full KL weight after this many additional epochs

    # Main training loop
    for epoch in range(num_epochs):
        # Update KL weight with annealing schedule
        if epoch >= kl_start:
            kl_weight = min(1.0, (epoch - kl_start) / kl_anneal_epochs)

        # Training phase
        VAE.train()
        train_loss = 0
        train_recon_loss = 0
        train_kl_loss = 0

        # Use tqdm for progress bar
        train_pbar = tqdm(train_dataloader,
                          desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for data, labels in train_pbar:
            data = data.to(device)
            labels = labels.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            recon_data, mu, log_var, z = VAE(data, labels)

            # Calculate loss
            loss, recon_loss, kl_loss = vae_loss_function(
                recon_data, data, mu, log_var, kl_weight)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Accumulate losses
            train_loss += loss.item()
            train_recon_loss += recon_loss.item()
            train_kl_loss += kl_loss.item()

            # Update progress bar
            train_pbar.set_postfix({
                'loss': loss.item() / data.size(0),
                'recon': recon_loss.item() / data.size(0),
                'kl': kl_loss.item() / data.size(0)
            })

        # Calculate average training losses
        avg_train_loss = train_loss / len(train_dataset)
        avg_train_recon = train_recon_loss / len(train_dataset)
        avg_train_kl = train_kl_loss / len(train_dataset)

        # Validation phase
        VAE.eval()
        val_loss = 0
        val_recon_loss = 0
        val_kl_loss = 0

        with torch.no_grad():
            val_pbar = tqdm(
                val_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
            for data, labels in val_pbar:
                data = data.to(device)
                labels = labels.to(device)

                # Forward pass
                recon_data, mu, log_var, z = VAE(data, labels)

                # Calculate loss
                loss, recon_loss, kl_loss = vae_loss_function(
                    recon_data, data, mu, log_var, kl_weight)

                # Accumulate losses
                val_loss += loss.item()
                val_recon_loss += recon_loss.item()
                val_kl_loss += kl_loss.item()

                # Update progress bar
                val_pbar.set_postfix({
                    'loss': loss.item() / data.size(0),
                    'recon': recon_loss.item() / data.size(0),
                    'kl': kl_loss.item() / data.size(0)
                })

        # Calculate average validation losses
        avg_val_loss = val_loss / len(val_dataset)
        avg_val_recon = val_recon_loss / len(val_dataset)
        avg_val_kl = val_kl_loss / len(val_dataset)

        # Record losses
        train_losses.append((avg_train_loss, avg_train_recon, avg_train_kl))
        val_losses.append((avg_val_loss, avg_val_recon, avg_val_kl))

        # Update learning rate
        scheduler.step()

        # Print epoch summary
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {avg_train_loss:.4f} (Recon: {avg_train_recon:.4f}, KL: {avg_train_kl:.4f}) | "
              f"Val Loss: {avg_val_loss:.4f} (Recon: {avg_val_recon:.4f}, KL: {avg_val_kl:.4f}) | "
              f"KL Weight: {kl_weight:.2f} | LR: {scheduler.get_last_lr()[0]:.6f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': VAE.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_val_loss,
            }, f'{VAE}/{session}_best_cvae_model.pth')
            print(
                f"✓ Saved best model with validation loss: {avg_val_loss:.4f}")

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': VAE.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_val_loss,
            }, f'{VAE}/{session}_epoch_{epoch+1}.pth')

            # Visualize reconstructions and generated samples
            visualize_results(VAE, val_dataloader, device, epoch, kl_weight)

    print("Training completed!")

    # Plot loss curves
    plot_training_curves(train_losses, val_losses)

    # Generate samples from the trained model
    print("Generating samples from trained model...")
    generate_samples(VAE, device, num_samples=10, num_classes=2)
