import os
import torch
import random
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data import NPZDataLoader
import numpy as np
from diffusers import DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from accelerate import Accelerator
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from models.dit import DiT


def set_seed(seed=3407):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)

session = 'S1'
# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
batch_size = 4
num_epochs = 100
learning_rate = 1e-4
num_train_timesteps = 1000
beta_start = 0.0001
beta_end = 0.02
img_size = 128
in_channels = 1
embed_dim = 1024
num_heads = 16
depth = 24
mlp_ratio = 4.0
save_model_epochs = 10
save_image_epochs = 5
patch_size = 8  # Size of image patches


# Create datasets
train_dataset = NPZDataLoader(f'{session}_train.npz')
val_dataset = NPZDataLoader(f'{session}_test.npz')

# Create data loaders
train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
)
val_dataloader = DataLoader(
    val_dataset, batch_size=1, shuffle=False, num_workers=4
)

# Initialize accelerator
accelerator = Accelerator(
    mixed_precision="fp16",
    gradient_accumulation_steps=1,
    log_with="tensorboard",
    project_dir="logs",
)

# Initialize the DiT model
model = DiT(
    img_size=img_size,
    patch_size=patch_size,
    in_channels=in_channels,
    embed_dim=embed_dim,
    depth=depth,
    num_heads=num_heads,
    mlp_ratio=mlp_ratio,
).to(device)

# Create noise scheduler
noise_scheduler = DDPMScheduler(
    num_train_timesteps=num_train_timesteps,
    beta_start=beta_start,
    beta_end=beta_end,
    beta_schedule="scaled_linear",
    clip_sample=False,
)

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Learning rate scheduler
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=500,
    num_training_steps=len(train_dataloader) * num_epochs,
)

# Prepare everything with accelerator
model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
    model, optimizer, train_dataloader, val_dataloader, lr_scheduler
)

# Helper functions for generating and saving samples


def save_samples(model, noise_scheduler, epoch, num_samples=4, output_dir="samples"):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        # Get a batch from validation set
        batch = next(iter(val_dataloader))
        clean_images = batch[0].to(device)[:num_samples]
        labels = batch[1].to(device)[:num_samples]

        # Create noise and add to images
        noise = torch.randn_like(clean_images)
        timesteps = torch.ones(clean_images.shape[0], device=device).long(
        ) * (noise_scheduler.num_train_timesteps - 1)
        noisy_images = noise_scheduler.add_noise(
            clean_images, noise, timesteps)

        # Sample with the model
        image_list = []
        noisy_sample = noisy_images.clone()

        for t in reversed(range(0, noise_scheduler.num_train_timesteps, 100)):
            timesteps = torch.tensor([t], device=device).repeat(
                noisy_sample.shape[0])
            model_output = model(noisy_sample, timesteps, labels)

            # 0 is for pred_original_sample
            step_output = noise_scheduler.step(model_output, t, noisy_sample)
            noisy_sample = step_output.prev_sample
            image_list.append(noisy_sample.detach().cpu())

        # Create a grid of images
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Original image
        axes[0, 0].imshow(
            clean_images[0, 0].detach().cpu().numpy(), cmap='gray')
        axes[0, 0].set_title(f"Original, Label: {labels[0].item()}")

        # Noisy image
        axes[0, 1].imshow(
            noisy_images[0, 0].detach().cpu().numpy(), cmap='gray')
        axes[0, 1].set_title("Noisy")

        # Final denoised
        axes[0, 2].imshow(image_list[-1][0, 0].numpy(), cmap='gray')
        axes[0, 2].set_title("Denoised")

        # Intermediate steps
        for i, idx in enumerate([0, 1, 2]):
            if idx < len(image_list) - 1:
                axes[1, i].imshow(image_list[idx][0, 0].numpy(), cmap='gray')
                axes[1, i].set_title(f"Denoising Step {idx}")

        plt.tight_layout()
        plt.savefig(f"{output_dir}/sample_epoch_{epoch}.png")
        plt.close()

# Training loop
def train_loop():
    global_step = 0

    for epoch in range(num_epochs):
        # Training
        model.train()
        progress_bar = tqdm(total=len(train_dataloader),
                            disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        train_loss = 0.0

        for batch in train_dataloader:
            clean_images = batch[0]
            labels = batch[1]

            # Sample noise to add to the images
            noise = torch.randn_like(clean_images)
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.num_train_timesteps, (bs,), device=accelerator.device
            ).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            noisy_images = noise_scheduler.add_noise(
                clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                # Predict noise with the model (we predict the noise component)
                model_output = model(noisy_images, timesteps, labels)

                # Calculate loss (we're predicting the clean image directly)
                loss = F.mse_loss(model_output, clean_images)
                accelerator.backward(loss)

                # Clip gradients
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            train_loss += loss.detach().item()
            global_step += 1

        progress_bar.close()
        avg_train_loss = train_loss / len(train_dataloader)

        # Validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_dataloader:
                clean_images = batch[0]
                labels = batch[1]

                # Sample noise to add to the images
                noise = torch.randn_like(clean_images)
                bs = clean_images.shape[0]

                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0, noise_scheduler.num_train_timesteps, (bs,), device=accelerator.device
                ).long()

                # Add noise to the clean images according to the noise magnitude at each timestep
                noisy_images = noise_scheduler.add_noise(
                    clean_images, noise, timesteps)

                # Predict the clean image
                model_output = model(noisy_images, timesteps, labels)

                # Calculate loss
                loss = F.mse_loss(model_output, clean_images)
                val_loss += loss.detach().item()

        avg_val_loss = val_loss / len(val_dataloader)

        # Log and print metrics
        logs = {
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "epoch": epoch,
            "step": global_step,
        }

        accelerator.log(logs, step=global_step)
        print(
            f"Epoch {epoch} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Save model checkpoint
        if epoch % save_model_epochs == 0 or epoch == num_epochs - 1:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                # Get a copy of the model without parallel processing wrappers
                unwrapped_model = accelerator.unwrap_model(model)

                # Save model
                model_path = f"./DiT/{session}_dit_model_epoch_{epoch}.pt"
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                torch.save(unwrapped_model.state_dict(), model_path)

        # Generate and save sample images
        if epoch % save_image_epochs == 0 or epoch == num_epochs - 1:
            if accelerator.is_main_process:
                save_samples(
                    accelerator.unwrap_model(model),
                    noise_scheduler,
                    epoch,
                    num_samples=4
                )

# Sample generation function for image augmentation
def generate_augmented_samples(model, noise_scheduler, num_samples=1, labels=None, device="cuda"):
    """
    Generate augmented ultrasound images using the trained DiT model

    Args:
        model: Trained DiT model
        noise_scheduler: DDPM scheduler
        num_samples: Number of samples to generate
        labels: Class labels for conditional generation (binary)
        device: Device to use for generation

    Returns:
        Generated ultrasound images
    """
    model.eval()

    # Create random noise
    noise = torch.randn(num_samples, in_channels,
                        img_size, img_size).to(device)

    # If no labels provided, generate random binary labels
    if labels is None:
        labels = torch.randint(0, 2, (num_samples, 1)).to(device)

    # Initialize with noise
    sample = noise

    # Sampling loop
    for t in tqdm(reversed(range(0, noise_scheduler.num_train_timesteps)), desc="Sampling"):
        timesteps = torch.full(
            (num_samples,), t, device=device, dtype=torch.long)

        # Forward pass
        with torch.no_grad():
            model_output = model(sample, timesteps, labels)

        # Compute previous sample using scheduler
        step_output = noise_scheduler.step(model_output, t, sample)
        sample = step_output.prev_sample

    # Normalize samples to [0, 1] range
    sample = (sample + 1) / 2

    return sample


# Start training
if __name__ == "__main__":
    # Initialize accelerator
    accelerator.init_trackers("diffusion_transformer_training")

    # Create directories
    os.makedirs("DiT", exist_ok=True)
    os.makedirs("samples", exist_ok=True)

    train_loop()

    # Final cleanup
    accelerator.end_training()

    print("Training completed!")

    # Example of generating augmented samples
    print("Generating example augmented samples...")
    model = accelerator.unwrap_model(model)
    augmented_samples = generate_augmented_samples(
        model=model,
        noise_scheduler=noise_scheduler,
        num_samples=4,
        labels=torch.tensor([[0], [1], [0], [1]], device=device),
    )

    # Save the generated samples
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(augmented_samples[i, 0].cpu().numpy(), cmap='gray')
        ax.set_title(f"Generated (Label: {i % 2})")
        ax.axis('off')

    plt.tight_layout()
    plt.savefig("samples/final_augmented_samples.png")
    plt.close()
