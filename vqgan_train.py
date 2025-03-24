import os
import torch
from data import NPZDataLoader
from models.vqgan import VQGAN
import torch.nn.functional as F
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


def train_vqgan(dataloader, session, num_epochs=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vqgan = VQGAN().to(device)
    optimizer_G = torch.optim.Adam(list(vqgan.encoder.parameters()) 
                                + list(vqgan.decoder.parameters()) 
                                + list(vqgan.codebook.parameters()), lr=1e-4)
    optimizer_D = torch.optim.Adam(vqgan.discriminator.parameters(), lr=1e-4)

    for epoch in range(num_epochs):
        for i, (fus, lab) in enumerate(dataloader):
            fus, lab = fus.to(device), lab.squeeze().to(device)  # lab shape: (B,)

            # Update discriminator
            optimizer_D.zero_grad()
            recon, _, _, _ = vqgan(fus, lab)
            real_logits = vqgan.discriminator(fus)
            fake_logits = vqgan.discriminator(recon.detach())
            d_loss = torch.mean(F.relu(1. - real_logits) +
                                F.relu(1. + fake_logits))
            d_loss.backward()
            optimizer_D.step()

            # Update generator
            optimizer_G.zero_grad()
            recon_loss, _, g_loss, codebook_loss = vqgan.compute_losses(fus, lab)
            total_g_loss = recon_loss + 0.1 * g_loss + codebook_loss
            total_g_loss.backward()
            optimizer_G.step()

            print(f"Epoch [{epoch}/100], Step [{i}/{len(dataloader)}], "
                f"Recon Loss: {recon_loss.item():.4f}, D Loss: {d_loss.item():.4f}")
        
        # Save models periodically
        if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
            torch.save(vqgan.state_dict(),
                       f"VQGAN/{session}_generator_epoch_{epoch+1}.pth")



if __name__ == "__main__":
    # Set random seed for reproducibility
    set_seed(42)

    session = 'S2'
    
    os.makedirs("VQGAN", exist_ok=True)

    # Load the data
    print("Loading and preparing data...")
    train_dataset = NPZDataLoader(f'{session}_train.npz')

    # Create data loaders
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=16, shuffle=True, num_workers=4)
    
    print(f"Training on {len(train_dataset)} samples")

    # Train the VQGAN model
    train_vqgan(train_dataloader, session)