import torch
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
import os
import matplotlib.pyplot as plt
from data import NPZDataLoader
from tqdm import tqdm
from models.stable_diffusion import UNet

def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)

def plot_images(images, path, **kwargs):
    grid = make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    plt.imshow(ndarr)
    plt.axis('off')
    plt.savefig(path)
    plt.close()

if __name__ == "__main__":
    session = "S2"
    model_name = "StableDiffusion"
    os.makedirs(model_name, exist_ok=True)
    os.makedirs(os.path.join(model_name, "results"), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Model
    model = UNet(
        dim=64,
        dim_mults=(1, 2, 4),
        n_res_blocks=2
    ).to(device)

    # Training settings
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    epochs = 200
    batch_size = 16
    timesteps = 1000

    # Dataloader
    train_dataset = NPZDataLoader(f'{session}_train.npz')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # Noise schedule
    betas = linear_beta_schedule(timesteps=timesteps).to(device)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

    def q_sample(x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    @torch.no_grad()
    def p_sample(model, x, t, t_index):
        betas_t = betas[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_recip_alphas_t = sqrt_recip_alphas[t].reshape(-1, 1, 1, 1)
        
        model_mean = sqrt_recip_alphas_t * (x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t)

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = posterior_variance[t].reshape(-1, 1, 1, 1)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def sample(model, image_size, batch_size=16, channels=1):
        img = torch.randn((batch_size, channels, image_size, image_size), device=device)
        imgs = []

        for i in tqdm(reversed(range(0, timesteps)), desc='sampling loop time step', total=timesteps):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            img = p_sample(model, img, t, i)
            imgs.append(img.cpu())
        return imgs

    # Training loop
    for epoch in range(epochs):
        for step, (batch, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
            optimizer.zero_grad()

            batch = batch.to(device)
            t = torch.randint(0, timesteps, (batch.size(0),), device=device).long()
            
            noise = torch.randn_like(batch)
            x_noisy = q_sample(x_start=batch, t=t, noise=noise)
            predicted_noise = model(x_noisy, t)
            
            loss = F.mse_loss(noise, predicted_noise)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{epochs} Loss: {loss.item()}")

        if (epoch + 1) % 10 == 0:
            print("Saving model and generating samples...")
            torch.save(model.state_dict(), f"{model_name}/{session}_{epoch+1}_sd.pth")
            
            # Generate and save samples
            model.eval()
            with torch.no_grad():
                imgs = sample(model, image_size=128, batch_size=4, channels=1)
                final_img = imgs[-1]
                plot_images(final_img, os.path.join(model_name, "results", f"epoch_{epoch+1}_sample.png"), nrow=2)
            model.train()