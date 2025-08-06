import os
import torch
import torch.nn.functional as F
import random
import numpy as np
from tqdm import tqdm
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

from models.stable_diffusion import UNet  # Assuming the model is in models/stable_diffusion.py
from data import NPZDataLoader

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

def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    """
    Generates a linear schedule for beta values.
    """
    return torch.linspace(start, end, timesteps)

def plot_images(images, path, **kwargs):
    """
    Helper function to plot and save a grid of images.
    """
    grid = make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    plt.imshow(ndarr)
    plt.axis('off')
    plt.savefig(path)
    plt.close()

if __name__ == "__main__":
    # Set random seed for reproducibility
    set_seed(42)

    # --- Configuration ---
    session = 'S1'
    model_name = "StableDiffusion"
    checkpoint_epoch = 200  # The epoch of the checkpoint to load
    checkpoint_path = f"{model_name}/{session}_{checkpoint_epoch}_sd.pth"
    
    output_dir = f"{model_name}_inference"
    os.makedirs(output_dir, exist_ok=True)

    # --- Device and Dataloader ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    val_dataset = NPZDataLoader(f'{session}_test.npz')
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    # --- Model Initialization ---
    model = UNet(
        dim=64,
        dim_mults=(1, 2, 4),
        n_res_blocks=2
    ).to(device)
    
    # Load the trained model checkpoint
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    except FileNotFoundError:
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("Please make sure you have a trained model checkpoint.")
        exit()

    model.eval()

    # --- Diffusion Parameters ---
    timesteps = 1000
    betas = linear_beta_schedule(timesteps=timesteps).to(device)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

    # --- Sampling Functions ---
    @torch.no_grad()
    def p_sample(model, x, t, t_index):
        """
        Performs one step of the reverse diffusion process.
        """
        betas_t = betas[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_recip_alphas_t = sqrt_recip_alphas[t].reshape(-1, 1, 1, 1)
        
        # Predict the mean of the posterior distribution
        model_mean = sqrt_recip_alphas_t * (x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t)

        if t_index == 0:
            return model_mean
        else:
            # Add noise to get the final sample for this timestep
            posterior_variance_t = posterior_variance[t].reshape(-1, 1, 1, 1)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def sample(model, image_size, batch_size=1, channels=1):
        """
        The main sampling loop to generate an image from random noise.
        """
        img = torch.randn((batch_size, channels, image_size, image_size), device=device)
        
        for i in tqdm(reversed(range(0, timesteps)), desc='Sampling loop', total=timesteps, leave=False):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            img = p_sample(model, img, t, i)
        
        # Denormalize the image from [-1, 1] to [0, 1]
        img = (img + 1) * 0.5
        return img

    # --- Inference Loop ---
    data_list = []
    label_list = []
    with torch.no_grad():
        for i, (_, labels) in enumerate(tqdm(val_dataloader, desc="Generating images")):
            
            # Generate fake images using the diffusion model
            # The number of images generated is based on the batch size of the dataloader
            fake_imgs = sample(model, image_size=128, batch_size=labels.size(0), channels=1)
            
            # Save a sample image for visualization
            if i < 5: # Save first 5 generated images
                plot_images(fake_imgs, path=os.path.join(output_dir, f"generated_sample_{i}.png"))

            # Process the generated data to match the format in test_cgan.py
            fake_imgs = fake_imgs.squeeze(0).permute(1, 2, 0)
            labels = labels.permute(1, 0)

            data_list.append(fake_imgs)
            label_list.append(labels)
    
    # --- Save Results ---
    print("Concatenating results...")
    data_list = torch.cat(data_list, dim=2).cpu().numpy()
    label_list = torch.cat(label_list, dim=1).cpu().numpy()

    output_filename = f"{output_dir}/{model_name}_{session}.npz"
    print(f"Saving generated data to {output_filename}")
    np.savez(output_filename, fus=data_list, label=label_list)
    print("Inference complete.")
