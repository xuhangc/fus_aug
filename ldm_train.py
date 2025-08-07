import torch

import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from data import NPZDataLoader
from models.ldm import VAE, UNetModel, LatentDiffusionModel

def plot_images(images, title="images", save_path=None):
    images = images.cpu().detach()
    grid = make_grid(images, nrow=4)
    plt.figure(figsize=(8, 8))
    plt.imshow(grid.permute(1, 2, 0))
    plt.title(title)
    plt.axis("off")
    if save_path:
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")
    plt.show()
    plt.close()

if __name__ == '__main__':
    # --- Configuration ---
    SESSION = "S2"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 16
    NUM_WORKERS = 4
    VAE_EPOCHS = 50
    LDM_EPOCHS = 200
    VAE_LR = 1e-4
    LDM_LR = 1e-4
    VAE_KL_WEIGHT = 1e-6
    
    # --- Create directories ---
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    VAE_MODEL_PATH = f"models/{SESSION}_vae.pth"
    LDM_MODEL_PATH = f"models/{SESSION}_ldm.pth"

    # --- DataLoaders ---
    train_dataset = NPZDataLoader(f'{SESSION}_train.npz')
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    
    # --- Stage 1: Train VAE ---
    print("="*20 + " Stage 1: Training VAE " + "="*20)
    vae_config = {
        "in_channels": 1, "out_channels": 1, "ch": 128, "ch_mult": [1, 2, 4],
        "num_res_blocks": 2, "dropout": 0.0, "z_channels": 4
    }
    vae = VAE(vae_config).to(DEVICE)
    optimizer_vae = torch.optim.Adam(vae.parameters(), lr=VAE_LR)

    for epoch in range(VAE_EPOCHS):
        epoch_loss = 0
        for i, (x, _) in enumerate(tqdm(train_loader, desc=f"VAE Epoch {epoch+1}/{VAE_EPOCHS}")):
            x = x.to(DEVICE)
            optimizer_vae.zero_grad()
            
            recon, mu, logvar = vae(x)
            
            recon_loss = F.mse_loss(recon, x)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + VAE_KL_WEIGHT * kl_loss
            
            loss.backward()
            optimizer_vae.step()
            epoch_loss += loss.item()

        print(f"VAE Epoch {epoch+1}, Loss: {epoch_loss / len(train_loader):.4f}")

        # Save a reconstruction sample
        if (epoch + 1) % 5 == 0:
            with torch.no_grad():
                sample_x, _ = next(iter(train_loader))
                sample_x = sample_x[:4].to(DEVICE)
                recon_sample, _, _ = vae(sample_x)
                combined = torch.cat([sample_x, recon_sample], dim=0)
                plot_images(combined, title=f"VAE Recon Epoch {epoch+1}", save_path=f"results/{SESSION}_vae_recon_epoch_{epoch+1}.png")

    torch.save(vae.state_dict(), VAE_MODEL_PATH)
    print(f"VAE model saved to {VAE_MODEL_PATH}")

    # --- Stage 2: Train Latent Diffusion Model ---
    print("\n" + "="*20 + " Stage 2: Training LDM " + "="*20)
    
    # Re-initialize VAE and load trained weights
    vae = VAE(vae_config).to(DEVICE)
    vae.load_state_dict(torch.load(VAE_MODEL_PATH, map_location=DEVICE))
    
    # Initialize U-Net
    unet = UNetModel(
        in_channels=4, # from z_channels in VAE
        model_channels=128,
        out_channels=4,
        num_res_blocks=2,
        attention_resolutions=[2, 4, 8], # Resolutions where to apply attention
        dropout=0.1,
        channel_mult=[1, 2, 4],
        num_classes=2 # Assuming 2 classes from the data loader
    ).to(DEVICE)
    
    ldm = LatentDiffusionModel(vae=vae, unet=unet).to(DEVICE)
    optimizer_ldm = torch.optim.Adam(ldm.unet.parameters(), lr=LDM_LR)

    for epoch in range(LDM_EPOCHS):
        epoch_loss = 0
        for i, (x, y) in enumerate(tqdm(train_loader, desc=f"LDM Epoch {epoch+1}/{LDM_EPOCHS}")):
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer_ldm.zero_grad()
            
            loss = ldm(x, y)
            loss.backward()
            optimizer_ldm.step()
            epoch_loss += loss.item()
            
        print(f"LDM Epoch {epoch+1}, Loss: {epoch_loss / len(train_loader):.4f}")

        # Generate and save samples
        if (epoch + 1) % 10 == 0:
            ldm.eval()
            with torch.no_grad():
                # Generate 4 samples for each class
                labels_class_0 = torch.zeros(4, dtype=torch.long, device=DEVICE)
                samples_0 = ldm.sample(num_samples=4, class_labels=labels_class_0, device=DEVICE)
                
                labels_class_1 = torch.ones(4, dtype=torch.long, device=DEVICE)
                samples_1 = ldm.sample(num_samples=4, class_labels=labels_class_1, device=DEVICE)

                all_samples = torch.cat([samples_0, samples_1], dim=0)
                plot_images(all_samples, title=f"LDM Samples Epoch {epoch+1}", save_path=f"results/{SESSION}_ldm_samples_epoch_{epoch+1}.png")
            ldm.train()

    torch.save(ldm.state_dict(), LDM_MODEL_PATH)
    print(f"LDM model saved to {LDM_MODEL_PATH}")
