import os
import torch
import torch.nn as nn
import random
import numpy as np
from tqdm import tqdm
from diffusers import UNet2DModel, DDPMScheduler, DDIMScheduler
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


# Custom Conditional UNet that incorporates the class label
class ConditionalUNet(nn.Module):
    def __init__(self, unet, num_classes=2):
        super().__init__()
        self.unet = unet
        self.class_embedding = nn.Embedding(
            num_classes, unet.config.block_out_channels[0])

    def forward(self, x, t, class_labels):
        # Embed class labels
        class_emb = self.class_embedding(class_labels.flatten())
        # Expand class embedding to match batch and spatial dimensions
        batch_size = x.shape[0]
        class_emb = class_emb.unsqueeze(1).unsqueeze(1)
        class_emb = class_emb.expand(batch_size, -1, x.shape[2], x.shape[3])

        # Concatenate the class embedding as an additional channel
        x_with_class = torch.cat([x, class_emb], dim=1)

        # Pass through UNet
        return self.unet(x_with_class, t)


if __name__ == "__main__":
    # Set random seed for reproducibility
    set_seed(42)

    session = 'S2'
    model = "DDIM"  # Using model as the variable name to match test_biggan.py

    val_dataset = NPZDataLoader(f'{session}_test.npz')

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=8)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create the UNet model for diffusion
    base_unet = UNet2DModel(
        sample_size=128,
        in_channels=2,
        out_channels=1,
        layers_per_block=4,
        block_out_channels=(64, 128, 256, 512),
        down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
    )
    generator = ConditionalUNet(base_unet, num_classes=2)
    generator.to(device)

    # Create the noise scheduler
    if model == "DDPM":
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear",
            clip_sample=False
        )
    elif model == "DDIM":
        noise_scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear",
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
            prediction_type="epsilon"
        )

    # Load the trained model
    model_path = f'{session}_{model}/best_model.pt'
    generator.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
    generator.eval()

    os.makedirs(f"{model}", exist_ok=True)

    data_list = []
    label_list = []
    
    # Set the number of inference steps for the scheduler
    num_inference_steps = 50
    noise_scheduler.set_timesteps(num_inference_steps)
    
    with torch.no_grad():
        for i, (imgs, labels) in enumerate(tqdm(val_dataloader)):
            # Move data to device
            real_imgs = imgs.to(device)
            labels = labels.to(device)
            
            # Start with random noise
            image = torch.randn((imgs.size(0), 1, 128, 128), device=device)
            
            # Diffusion process (reverse)
            for t in noise_scheduler.timesteps:
                # Expand the timesteps for batch dimension
                timesteps = torch.full((imgs.size(0),), t, device=device, dtype=torch.long)
                
                # Predict noise residual
                noise_pred = generator(image, timesteps, labels).sample
                
                # Update image with the predicted noise
                image = noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=t,
                    sample=image
                ).prev_sample
            
            # Process the generated data to match the format in test_biggan.py
            fake_imgs = image.squeeze(0).permute(1, 2, 0)
            labels = labels.permute(1, 0)
            
            data_list.append(fake_imgs)
            label_list.append(labels)
    
    data_list = torch.cat(data_list, dim=2).cpu().numpy()
    label_list = torch.cat(label_list, dim=1).cpu().numpy()
    
    np.savez(f"{model}/{model}_{session}.npz", fus=data_list, label=label_list)
    print(f"Generated samples saved to {model}/{model}_{session}.npz")