import torch
import torch.nn.functional as F
from torchvision.utils import make_grid
import os
import matplotlib.pyplot as plt
from data import NPZDataLoader
from tqdm import tqdm

from models.vqvae import VQVAE, VQVAEConfig

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

    model_name = "VQVAE"
    os.makedirs(model_name, exist_ok=True)

    patch_sizes = [1, 2, 3, 4, 5, 6, 8, 10, 13, 16, 32]
    
    max_len = sum(p**2 for p in patch_sizes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = VQVAEConfig(
        resolution=128,
        in_channels=1,
        dim=128,
        ch_mult=[1, 2, 4],
        num_res_blocks=2,
        z_channels=64,
        out_ch=1,
        vocab_size=8192,
        patch_sizes=patch_sizes,
    )

    vq_model = VQVAE(config)
    optimizer = torch.optim.AdamW(vq_model.parameters(), lr=3e-4)

    train_dataset = NPZDataLoader(f'{session}_train.npz')
    test_dataset = NPZDataLoader(f'{session}_test.npz')

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=2, shuffle=True, num_workers=16)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=16)
    
    vq_model = vq_model.to(device)
    for epoch in range(100):
        epoch_loss = 0
        epoch_recon_loss = 0
        for i, (x, c) in enumerate(tqdm(train_loader)):
            x, c = x.to(device), c.to(device).flatten()
            optimizer.zero_grad()
            xhat, r_maps, idxs, scales, q_loss = vq_model(x)
            recon_loss = F.mse_loss(xhat, x)
            loss = recon_loss + q_loss
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_recon_loss += recon_loss.item()

        epoch_loss /= len(train_loader)
        epoch_recon_loss /= len(train_loader)
        print(f"Epoch: {epoch}, Loss: {epoch_loss}, Recon Loss: {epoch_recon_loss}")
        torch.save(vq_model.state_dict(), f"{model_name}/{session}_{epoch}_vqvae.pth")

        if epoch % 5 == 0:
            with torch.no_grad():
                total_loss = 0
                total_recon_loss = 0
                for i, (x, c) in enumerate(tqdm(test_loader)):
                    x, c = x.to(device), c.to(device)
                    xhat, r_maps, idxs, scales, q_loss = vq_model(x)
                    recon_loss = F.mse_loss(xhat, x)
                    loss = recon_loss + q_loss
                    total_loss += loss.item()
                    total_recon_loss += recon_loss.item()

                total_loss /= len(test_loader)
                total_recon_loss /= len(test_loader)

                print(f"Epoch: {epoch}, Test Loss: {total_loss}, Test Recon Loss: {total_recon_loss}")

                x = x[:10, :].to(device)
                x_hat = vq_model(x)[0]

                plot_images(pred=x_hat, original=x)
                plt.savefig(f"{model_name}/{session}_vqvae_{epoch}.png")
                plt.close()

    torch.save(vq_model.state_dict(), f"{model_name}/{session}_vqvae.pth")