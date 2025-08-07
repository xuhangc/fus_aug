import os
import torch
import numpy as np
from tqdm import tqdm
from models.ldm import VAE, UNetModel, LatentDiffusionModel
from data import NPZDataLoader

# Set random seeds for reproducibility
def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    # Set random seed for reproducibility
    set_seed(42)

    # --- Configuration ---
    session = 'S1'
    model_name = "LDM"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- DataLoader ---
    # Note: Batch size is 1 to generate one image per label from the test set
    val_dataset = NPZDataLoader(f'{session}_test.npz')
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=8)

    # --- Model Hyperparameters (must match training configuration) ---
    vae_config = {
        "in_channels": 1, "out_channels": 1, "ch": 128, "ch_mult": [1, 2, 4],
        "num_res_blocks": 2, "dropout": 0.0, "z_channels": 4
    }
    unet_config = {
        "in_channels": 4,
        "model_channels": 128,
        "out_channels": 4,
        "num_res_blocks": 2,
        "attention_resolutions": [2, 4, 8],
        "dropout": 0.1,
        "channel_mult": [1, 2, 4],
        "num_classes": 2
    }
    ldm_timesteps = 1000
    
    # --- Initialize and Load Models ---
    # Initialize the full LDM architecture
    vae = VAE(vae_config).to(device)
    unet = UNetModel(**unet_config).to(device)
    ldm = LatentDiffusionModel(vae=vae, unet=unet, timesteps=ldm_timesteps).to(device)
    
    # Load the trained LDM state dictionary
    ldm_model_path = f'models/{session}_ldm.pth'
    checkpoint = torch.load(ldm_model_path, map_location=device)
    ldm.load_state_dict(checkpoint)
    ldm.eval()

    # Create output directory
    os.makedirs(f"{model_name}", exist_ok=True)

    # --- Inference Loop ---
    data_list = []
    label_list = []
    with torch.no_grad():
        for i, (imgs, labels) in enumerate(tqdm(val_dataloader, desc=f"Generating with {model_name}")):
            # The LDM's UNet expects class indices (long type), not one-hot vectors.
            # We convert the label from the dataloader to a class index.
            class_indices = torch.argmax(labels, dim=1).long().to(device)

            # Generate images by sampling from the LDM
            fake_imgs = ldm.sample(num_samples=imgs.size(0), class_labels=class_indices, device=device)
            
            # Process the generated data to match the format in test_cgan.py
            # Output shape from sample(): (B, C, H, W) -> (H, W, C)
            fake_imgs = fake_imgs.squeeze(0).permute(1, 2, 0)
            
            # Process labels to match the output format [num_classes, num_samples]
            labels = labels.permute(1, 0)

            data_list.append(fake_imgs)
            label_list.append(labels)
    
    # --- Save Results ---
    # Concatenate all generated images and labels
    data_list = torch.cat(data_list, dim=2).cpu().numpy()
    label_list = torch.cat(label_list, dim=1).cpu().numpy()

    # Save in the same .npz format as the CGAN script
    output_path = f"{model_name}/{model_name}_{session}.npz"
    np.savez(output_path, fus=data_list, label=label_list)
    print(f"Generated data saved to {output_path}")

