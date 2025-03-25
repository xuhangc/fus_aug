import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid
import random
import numpy as np
from data import NPZDataLoader
from models.vqvae import VQVAE, VQVAEConfig


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
    # Set random seed for reproducibility
    set_seed(42)

    session = 'S2'
    model_name = "VQVAE"
    os.makedirs(f"{model_name}", exist_ok=True)

    patch_sizes = [1, 2, 3, 4, 5, 6, 8, 10, 13, 16, 32]
    max_len = sum(p**2 for p in patch_sizes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize model configuration
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

    # Load test dataset
    test_dataset = NPZDataLoader(f'{session}_test.npz')
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=16)

    # Initialize model
    vq_model = VQVAE(config).to(device)
    
    # Load trained model
    checkpoint = torch.load(f"{model_name}/{session}_5_vqvae.pth", weights_only=True,
                           map_location=device if torch.cuda.is_available() else 'cpu')
    vq_model.load_state_dict(checkpoint)
    vq_model.eval()

    # Evaluate model
    data_list = []
    label_list = []
    metrics = {
        "loss": 0.0,
        "recon_loss": 0.0,
        "quant_loss": 0.0
    }
    
    with torch.no_grad():
        for i, (x, c) in enumerate(tqdm(test_loader)):
            x, c = x.to(device), c.to(device)
            
            # Forward pass
            xhat, r_maps, idxs, scales, q_loss = vq_model(x)
            
            # Calculate losses
            recon_loss = F.mse_loss(xhat, x)
            loss = recon_loss + q_loss
            
            # Update metrics
            metrics["loss"] += loss.item()
            metrics["recon_loss"] += recon_loss.item()
            metrics["quant_loss"] += q_loss.item()
            
            # For visualization
            if i < 10:  # Save first 10 reconstructions
                plt.figure(figsize=(12, 6))
                plot_images(pred=xhat, original=x)
                plt.savefig(f"{model_name}/{session}_vqvae_recon_{i}.png")
                plt.close()
            
            # For dataset creation (similar to test_biggan.py)
            reconstructed = xhat.squeeze(0).permute(1, 2, 0)
            label = c.permute(1, 0)
            
            data_list.append(reconstructed)
            label_list.append(label)

    # Calculate average metrics
    for key in metrics:
        metrics[key] /= len(test_loader)
    
    # Print evaluation results
    print(f"Test Results:")
    print(f"Total Loss: {metrics['loss']:.6f}")
    print(f"Reconstruction Loss: {metrics['recon_loss']:.6f}")
    print(f"Quantization Loss: {metrics['quant_loss']:.6f}")
    
    # Save reconstructed data
    data_tensor = torch.cat(data_list, dim=2).cpu().numpy()
    label_tensor = torch.cat(label_list, dim=1).cpu().numpy()
    
    np.savez(f"{model_name}/{model_name}_{session}.npz", 
             fus=data_tensor, label=label_tensor)
    
    # Save metrics to file
    with open(f"{model_name}/{model_name}_{session}_metrics.txt", "w") as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value:.6f}\n")