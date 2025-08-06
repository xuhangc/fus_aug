import torch
import torch.nn.functional as F
from torchvision.utils import make_grid
import os
import matplotlib.pyplot as plt
from data import NPZDataLoader
from tqdm import tqdm
import numpy as np

from models.stable_diffusion import UNet, DiffusionModel


def plot_images(pred, original=None):
    n = pred.size(0)
    pred = pred * 0.5 + 0.5
    pred = pred.clamp(0, 1)
    img = pred.cpu().detach()

    if original is not None:
        original = original * 0.5 + 0.5
        original = original.clamp(0, 1)
        original = original.cpu().detach()
        img = torch.cat([original, img], dim=0)

    img_grid = make_grid(img, nrow=n)
    img_grid = img_grid.permute(1, 2, 0).numpy()
    img_grid = (img_grid * 255).astype("uint8")
    plt.imshow(img_grid)
    plt.axis("off")


if __name__ == "__main__":
    session = "S1"

    model_name = "StableDiffusion"
    os.makedirs(model_name, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create UNet model for noise prediction
    unet = UNet(
        in_channels=1,  # 1 for image, 1 for time embedding
        model_channels=128,
        out_channels=1,
        num_res_blocks=2,
        attention_resolutions=(16, 8),
        channel_mult=[1, 2, 4, 8],
        num_heads=8,
        n_classes=2  # Changed from context_dim to n_classes
    )
    
    # Create diffusion model
    diffusion = DiffusionModel(
        model=unet,
        image_size=128,
        timesteps=1000,
        sampling_timesteps=250,
        loss_type='l1'
    )
    
    optimizer = torch.optim.AdamW(diffusion.parameters(), lr=1e-4)
    
    train_dataset = NPZDataLoader(f'{session}_train.npz')
    test_dataset = NPZDataLoader(f'{session}_test.npz')

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=8, shuffle=True, num_workers=16)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=4, shuffle=False, num_workers=16)
    
    diffusion = diffusion.to(device)
    
    print(f"Diffusion Model Parameters: {sum(p.numel() for p in diffusion.parameters())/1e6:.2f}M")
    
    best_loss = float('inf')
    best_checkpoint = ""
    
    # Training loop
    for epoch in range(200):
        diffusion.train()
        epoch_loss = 0
        
        for i, (x, c) in enumerate(tqdm(train_loader)):
            x, c = x.to(device), c.to(device).flatten()
            
            optimizer.zero_grad()
            
            # Forward pass
            loss = diffusion(x, cond=c)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        epoch_loss /= len(train_loader)
        print(f"Epoch: {epoch}, Loss: {epoch_loss}")
        
        # Save checkpoint
        torch.save(diffusion.state_dict(), f"{model_name}/{session}_{epoch}.pth")
        
        # Validation
        if epoch % 5 == 0:
            diffusion.eval()
            with torch.no_grad():
                total_loss = 0
                
                # Compute validation loss
                for i, (x, c) in enumerate(tqdm(test_loader)):
                    x, c = x.to(device), c.to(device).flatten()
                    loss = diffusion(x, cond=c)
                    total_loss += loss.item()
                
                total_loss /= len(test_loader)
                print(f"Epoch: {epoch}, Validation Loss: {total_loss}")
                
                # Save best model
                if total_loss < best_loss:
                    best_loss = total_loss
                    best_checkpoint = f"{model_name}/{session}_{epoch}.pth"
                    torch.save(diffusion.state_dict(), best_checkpoint)
                
                # Generate samples for visualization
                c_test = torch.tensor([0, 1], device=device)  # Generate samples for both classes
                samples = diffusion.sample(batch_size=len(c_test), cond=c_test)
                
                plot_images(pred=samples)
                plt.savefig(f"{model_name}/{session}_samples_{epoch}.png")
                plt.close()
                
                # Generate and plot reconstructions
                x_test = next(iter(test_loader))[0][:4].to(device)
                reconstructions = diffusion.reconstruct(x_test)
                
                plot_images(pred=reconstructions, original=x_test)
                plt.savefig(f"{model_name}/{session}_reconstruction_{epoch}.png")
                plt.close()
    
    print("Training completed!")
    print(f"Best model saved at: {best_checkpoint}")
    
    # Final evaluation
    diffusion.load_state_dict(torch.load(best_checkpoint))
    diffusion.eval()
    
    # Generate class-conditional samples
    for class_idx in range(2):
        c = torch.tensor([class_idx] * 10, device=device)
        samples = diffusion.sample(batch_size=10, cond=c)
        
        plot_images(pred=samples)
        plt.savefig(f"{model_name}/{session}_final_class{class_idx}.png")
        plt.close()