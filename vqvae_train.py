import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import torch.nn.functional as F
from models.vqvae import ClassConditionalVQVAE
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


if __name__ == "__main__":
    # Set random seed for reproducibility
    set_seed(42)

    session = 'S1'
    model = 'VQVAE'

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

    VAE = ClassConditionalVQVAE().to(device)

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

    # Main training loop
    for epoch in range(num_epochs):

        # Training phase
        VAE.train()
        train_loss = 0
        # Use tqdm for progress bar
        train_pbar = tqdm(train_dataloader,
                          desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for data, labels in train_pbar:
            data = data.to(device)
            labels = labels.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            x_recon, vq_loss, indices = VAE(data, labels)

            # Backward pass and optimize
            vq_loss.backward()
            optimizer.step()

            # Accumulate losses
            train_loss += vq_loss.item()

            # Update progress bar
            train_pbar.set_postfix({
                'loss': vq_loss.item() / data.size(0),
            })

        # Calculate average training losses
        avg_train_loss = train_loss / len(train_dataset)

        # Validation phase
        VAE.eval()
        val_loss = 0

        with torch.no_grad():
            val_pbar = tqdm(
                val_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
            for data, labels in val_pbar:
                data = data.to(device)
                labels = labels.to(device)

                # Forward pass
                x_recon, vq_loss, indices = VAE(data, labels)

                # Accumulate losses
                val_loss += vq_loss.item()

                # Update progress bar
                val_pbar.set_postfix({
                    'loss': val_loss.item() / data.size(0),
                })

        # Calculate average validation losses
        avg_val_loss = val_loss / len(val_dataset)

        # Record losses
        train_losses.append((avg_train_loss))
        val_losses.append((avg_val_loss))

        # Update learning rate
        scheduler.step()

        # Print epoch summary
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': VAE.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_val_loss,
            }, f'{model}/{session}_best_cvae_model.pth')
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
            }, f'{model}/{session}_epoch_{epoch+1}.pth')

            # Visualize reconstructions and generated samples
            # visualize_results(VAE, val_dataloader, device, epoch, kl_weight)

    print("Training completed!")

    # Plot loss curves
    # plot_training_curves(train_losses, val_losses)

    # Generate samples from the trained model
    # print("Generating samples from trained model...")
    # generate_samples(VAE, device, num_samples=10, num_classes=2)
