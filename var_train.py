import torch
import torch.nn.functional as F
from torchvision.utils import make_grid
import os
import matplotlib.pyplot as plt
from data import NPZDataLoader
from tqdm import tqdm

from models.var import VQVAE, VAR, VQVAEConfig


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

    model_name = "VAR"
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

    temp_loss = 100
    best_vqvae_checkpoint = ""
    for epoch in range(200):
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

        if epoch % 1 == 0:
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

                if total_recon_loss < temp_loss:
                    temp_loss = total_recon_loss
                    torch.save(vq_model.state_dict(), f"{model_name}/{session}_{epoch}_vqvae.pth")

                    best_vqvae_checkpoint = f"{model_name}/{session}_{epoch}_vqvae.pth"

                    x = x[:10, :].to(device)
                    x_hat = vq_model(x)[0]

                    plot_images(pred=x_hat, original=x)
                    plt.savefig(f"{model_name}/{session}_vqvae_{epoch}.png")
                    plt.close()

                print(f"Epoch: {epoch}, Test Loss: {total_loss}, Test Recon Loss: {total_recon_loss}")


    del vq_model, optimizer, x, x_hat, train_loader, test_loader
    torch.cuda.empty_cache()

    temp_loss = 100

    print("=" * 10 + "Training VAR" + "=" * 10)
    vqvae = VQVAE(config)
    vqvae.load_state_dict(torch.load(best_vqvae_checkpoint, weights_only=True))
    vqvae = vqvae.to(device)
    vqvae.eval()

    for param in vqvae.parameters():
        param.requires_grad = False

    var_model = VAR(vqvae=vqvae, dim=128, n_heads=8, n_layers=3, patch_sizes=patch_sizes, n_classes=2)
    optimizer = torch.optim.AdamW(var_model.parameters(), lr=1e-3)

    print(f"VQVAE Parameters: {sum(p.numel() for p in vqvae.parameters())/1e6:.2f}M")
    print(f"VAR Parameters: {sum(p.numel() for p in var_model.parameters())/1e6:.2f}M")

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=2, shuffle=True, num_workers=16)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=16)
    var_model = var_model.to(device)
    
    for epoch in range(200):
        epoch_loss = 0
        for i, (x, c) in enumerate(tqdm(train_loader)):
            x, c = x.to(device), c.to(device).flatten()
            optimizer.zero_grad()

            _, _, idxs_R_BL, scales_BlC, _ = vqvae(x)
            idx_BL = torch.cat(idxs_R_BL, dim=1)
            scales_BlC = scales_BlC.to(device)
            logits_BLV = var_model(scales_BlC, cond=c)
            loss = F.cross_entropy(logits_BLV.view(-1, logits_BLV.size(-1)), idx_BL.view(-1))

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= len(train_loader)
        print(f"Epoch: {epoch}, Loss: {epoch_loss}")

        if epoch % 1 == 0:
            with torch.no_grad():
                total_loss = 0

                for i, (x, c) in enumerate(tqdm(test_loader)):
                    x, c = x.to(device), c.to(device).flatten()
                    out_B3HW = var_model.generate(c, 0)
                    recon_loss = F.mse_loss(xhat, x)
                    loss = recon_loss
                    total_loss += loss.item()

                if total_loss < temp_loss:
                    temp_loss = total_loss
                    torch.save(var_model.state_dict(), f"{model_name}/{session}_{epoch}_var.pth")

                    cond = torch.arange(2).to(device)
                    out_B3HW = var_model.generate(cond, 0)
                    plot_images(pred=out_B3HW)

                    plt.savefig(f"{model_name}/{session}_var_{epoch}.png")
                    plt.close()
