import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid
import random
import numpy as np
from data import NPZDataLoader
from models.var import VQVAE, VAR, VQVAEConfig


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

    session = 'S1'
    model_name = "VAR"

    # Ensure output directory exists
    os.makedirs(f"{model_name}", exist_ok=True)

    # Define patch sizes as in training
    patch_sizes = [1, 2, 3, 4, 5, 6, 8, 10, 13, 16, 32]
    max_len = sum(p**2 for p in patch_sizes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create the configuration for VQVAE
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

    # Load the test dataset
    test_dataset = NPZDataLoader(f'{session}_test.npz')
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=8)

    # Initialize and load VQVAE model
    vqvae = VQVAE(config)
    vqvae.load_state_dict(torch.load(f"{model_name}/{session}_vqvae.pth", 
                                      weights_only=True, 
                                      map_location=device))
    vqvae = vqvae.to(device)
    vqvae.eval()

    for param in vqvae.parameters():
        param.requires_grad = False

    # Initialize and load VAR model
    var_model = VAR(vqvae=vqvae, dim=128, n_heads=8, n_layers=3, patch_sizes=patch_sizes, n_classes=2)
    var_model.load_state_dict(torch.load(f"{model_name}/{session}_var.pth", 
                                          weights_only=True, 
                                          map_location=device))
    var_model = var_model.to(device)
    var_model.eval()

    # Generate samples for evaluation
    data_list = []
    label_list = []
    with torch.no_grad():
        # Generate samples for each class
        for class_idx in range(2):  # For 2 classes
            print(f"Generating samples for class {class_idx}")
            # Generate multiple samples per class
            for i in range(10):  # Generate 10 samples per class
                cond = torch.tensor([class_idx]).to(device)
                # Generate sample with temperature 0.8
                fake_img = var_model.generate(cond, temperature=0.8)
                
                # Plot and save individual generated images
                if i < 5:  # Save first 5 samples as images
                    plot_images(pred=fake_img)
                    plt.savefig(f"{model_name}/{session}_var_test_class{class_idx}_sample{i}.png")
                    plt.close()
                
                # Reshape for data collection
                fake_img = fake_img.squeeze(0).permute(1, 2, 0)
                data_list.append(fake_img)
                label_list.append(torch.tensor([[class_idx]]))
        
        # Also test reconstruction quality on test set
        for i, (imgs, labels) in enumerate(tqdm(test_loader)):
            if i >= 10:  # Limit to 10 samples to avoid too many outputs
                break
                
            real_imgs = imgs.to(device)
            labels = labels.to(device)
            
            # Reconstruction through VQVAE
            recon_imgs, _, _, _, _ = vqvae(real_imgs)
            
            # Plot and save comparison
            plot_images(pred=recon_imgs, original=real_imgs)
            plt.savefig(f"{model_name}/{session}_var_test_recon_{i}.png")
            plt.close()
    
    # Convert to numpy and save generated samples
    if data_list:
        data_tensor = torch.stack(data_list, dim=0)
        label_tensor = torch.cat(label_list, dim=0)
        
        data_array = data_tensor.cpu().numpy()
        label_array = label_tensor.cpu().numpy()
        
        np.savez(f"{model_name}/{model_name}_{session}.npz", fus=data_array, label=label_array)
        print(f"Generated data saved to {model_name}/{model_name}_{session}.npz")
    
    # Evaluate reconstruction quality on the test set
    with torch.no_grad():
        total_loss = 0
        total_recon_loss = 0
        for i, (imgs, labels) in enumerate(tqdm(test_loader)):
            real_imgs = imgs.to(device)
            labels = labels.to(device)
            
            # Get reconstruction and quantization loss
            recon_imgs, _, idxs_R_BL, scales_BlC, q_loss = vqvae(real_imgs)
            recon_loss = F.mse_loss(recon_imgs, real_imgs)
            
            # Calculate VAR loss (prediction of quantized tokens)
            idx_BL = torch.cat(idxs_R_BL, dim=1)
            logits_BLV = var_model(scales_BlC, cond=labels.flatten())
            var_loss = F.cross_entropy(logits_BLV.view(-1, logits_BLV.size(-1)), idx_BL.view(-1))
            
            total_loss += var_loss.item()
            total_recon_loss += recon_loss.item()
        
        total_loss /= len(test_loader)
        total_recon_loss /= len(test_loader)
        
        print(f"Test VAR Loss: {total_loss}")
        print(f"Test Reconstruction Loss: {total_recon_loss}")