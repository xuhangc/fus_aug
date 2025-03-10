import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from data import NPZDataLoader
from torch.utils.data import DataLoader
from diffusers import UNet2DModel, DDPMScheduler, DDIMScheduler
from diffusers.optimization import get_scheduler
import numpy as np
from tqdm.auto import tqdm
import random


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
        class_emb = self.class_embedding(class_labels)

        # Expand class embedding to match batch and spatial dimensions
        batch_size = x.shape[0]
        class_emb = class_emb.unsqueeze(-1).unsqueeze(-1)
        class_emb = class_emb.expand(batch_size, -1, x.shape[2], x.shape[3])

        # Concatenate the class embedding as an additional channel
        x_with_class = torch.cat([x, class_emb], dim=1)

        # Pass through UNet
        return self.unet(x_with_class, t)

# Function to train the diffusion model


def train_diffusion_model(
    train_dataloader,
    val_dataloader=None,
    num_epochs=100,
    learning_rate=2e-4,
    save_dir="ddim",
    device="cuda" if torch.cuda.is_available() else "cpu",
    patience=20  # Early stopping patience
):
    # Create the UNet model for diffusion
    base_unet = UNet2DModel(
        sample_size=128,      # The target image resolution
        # Number of input channels (1 for image + 1 for class embedding)
        in_channels=2,
        # Number of output channels (predicting noise for the image)
        out_channels=1,
        layers_per_block=2,
        block_out_channels=(128, 256, 512, 512),
        down_block_types=("DownBlock2D", "DownBlock2D",
                          "DownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
    )
    model = ConditionalUNet(base_unet, num_classes=2)
    model.to(device)

    # Create the noise scheduler
    # noise_scheduler = DDPMScheduler(
    #     num_train_timesteps=1000,
    #     beta_start=0.0001,
    #     beta_end=0.02,
    #     beta_schedule="linear",
    #     clip_sample=False
    # )
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

    # Create the optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-5)

    # Setup the learning rate scheduler
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=len(train_dataloader) * num_epochs
    )

    # Create directories for saving models and visualizations
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "samples"), exist_ok=True)

    # Training loop
    global_step = 0
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_path = os.path.join(save_dir, "best_model.pt")
    early_stopping_counter = 0

    for epoch in range(num_epochs):
        model.train()
        epoch_losses = []
        progress_bar = tqdm(total=len(train_dataloader))
        progress_bar.set_description(f"Epoch {epoch}")

        for batch in train_dataloader:
            # Get the inputs
            clean_images, class_labels = batch
            clean_images = clean_images.to(device)
            class_labels = class_labels.to(device)
            batch_size = clean_images.shape[0]

            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape, device=device)

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.num_train_timesteps, (batch_size,), device=device
            ).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            noisy_images = noise_scheduler.add_noise(
                clean_images, noise, timesteps)

            # Predict the noise residual
            noise_pred = model(noisy_images, timesteps, class_labels).sample

            # Calculate the loss
            loss = F.mse_loss(noise_pred, noise)
            epoch_losses.append(loss.item())

            # Update the model parameters
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(
            ), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            global_step += 1

        # Calculate average loss for the epoch
        avg_train_loss = sum(epoch_losses) / len(epoch_losses)
        train_losses.append(avg_train_loss)
        print(f"Epoch {epoch} - Average Training Loss: {avg_train_loss:.4f}")

        # Validation
        if val_dataloader is not None:
            model.eval()
            val_epoch_losses = []
            with torch.no_grad():
                for batch in val_dataloader:
                    clean_images, class_labels = batch
                    clean_images = clean_images.to(device)
                    class_labels = class_labels.to(device)
                    batch_size = clean_images.shape[0]

                    noise = torch.randn(clean_images.shape, device=device)
                    timesteps = torch.randint(
                        0, noise_scheduler.num_train_timesteps, (batch_size,), device=device
                    ).long()

                    noisy_images = noise_scheduler.add_noise(
                        clean_images, noise, timesteps)
                    noise_pred = model(
                        noisy_images, timesteps, class_labels).sample

                    loss = F.mse_loss(noise_pred, noise)
                    val_epoch_losses.append(loss.item())

            current_val_loss = sum(val_epoch_losses) / len(val_epoch_losses)
            val_losses.append(current_val_loss)
            print(f"Epoch {epoch} - Validation Loss: {current_val_loss:.4f}")

            # Check if this is the best model so far (based on validation loss)
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                torch.save(model.state_dict(), best_model_path)
                print(
                    f"New best model saved with validation loss: {best_val_loss:.4f}")
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                print(
                    f"Early stopping counter: {early_stopping_counter}/{patience}")

                # Early stopping
                if early_stopping_counter >= patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs")
                    break

        # Save the model periodically
        if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
            torch.save(model.state_dict(), os.path.join(
                save_dir, f"model_epoch_{epoch}.pt"))

            # Generate and save some sample images
            # for class_label in [0, 1]:
            #     generated_images = generate_images(
            #         model=model,
            #         noise_scheduler=noise_scheduler,
            #         class_label=class_label,
            #         num_images=4,
            #         device=device
            #     )

            #     Plot and save the generated images
            #     fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            #     for i, image in enumerate(generated_images):
            #         axes[i].imshow(image[0], cmap='gray')
            #         axes[i].set_title(f"Class {class_label}")
            #         axes[i].axis('off')

            #     plt.tight_layout()
            #     plt.savefig(os.path.join(save_dir, "samples",
            #                 f"epoch_{epoch}_class_{class_label}.png"))
            #     plt.close()

    # Load the best model if available
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        print(f"Loaded best model with validation loss: {best_val_loss:.4f}")

    # Plot and save the training curves
    # plt.figure(figsize=(10, 5))
    # plt.plot(train_losses, label='Training Loss')
    # if val_dataloader is not None:
    #     plt.plot(val_losses, label='Validation Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.title('Training and Validation Losses')
    # plt.savefig(os.path.join(save_dir, "loss_curve.png"))
    # plt.close()

    return model, noise_scheduler

# Function to generate new images using the trained model


# def generate_images(
#     model,
#     noise_scheduler,
#     class_label,
#     num_inference_steps=50,
#     num_images=4,
#     device="cuda" if torch.cuda.is_available() else "cpu"
# ):
#     # Put model in evaluation mode
#     model.eval()

#     # Create class labels tensor
#     class_labels = torch.tensor([class_label] * num_images, device=device)

#     # Start with random noise
#     image_shape = (num_images, 1, 128, 128)
#     image = torch.randn(image_shape, device=device)

#     # Use DDIM scheduler for faster and higher quality sampling
#     ddim_scheduler = DDIMScheduler.from_config(noise_scheduler.config)
#     ddim_scheduler.set_timesteps(num_inference_steps)

#     # Diffusion process (reverse)
#     for t in tqdm(ddim_scheduler.timesteps):
#         # Expand the timesteps for batch dimension
#         timesteps = torch.full(
#             (num_images,), t, device=device, dtype=torch.long)

#         # Predict noise residual
#         with torch.no_grad():
#             noise_pred = model(image, timesteps, class_labels).sample

#         # Update image with the predicted noise
#         image = ddim_scheduler.step(noise_pred, t, image).prev_sample

#     # Normalize image to [0, 1]
#     image = (image + 1) / 2
#     image = image.clamp(0, 1)

#     return image.cpu().numpy()

def generate_images(
    model,
    noise_scheduler,
    class_label,
    num_inference_steps=50,
    eta=0.0,  # 0.0 for deterministic sampling, 1.0 for stochastic
    num_images=4,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    # Put model in evaluation mode
    model.eval()

    # Create class labels tensor
    class_labels = torch.tensor([class_label] * num_images, device=device)

    # Start with random noise
    image_shape = (num_images, 1, 128, 128)
    image = torch.randn(image_shape, device=device)

    # Set number of inference steps for the scheduler
    noise_scheduler.set_timesteps(num_inference_steps)

    # Diffusion process (reverse)
    for t in tqdm(noise_scheduler.timesteps):
        # Expand the timesteps for batch dimension
        timesteps = torch.full(
            (num_images,), t, device=device, dtype=torch.long)

        # Predict noise residual
        with torch.no_grad():
            noise_pred = model(image, timesteps, class_labels).sample

        # DDIM update step with specified eta parameter
        image = noise_scheduler.step(
            model_output=noise_pred,
            timestep=t,
            sample=image,
            # Control stochasticity (0 = deterministic, 1 = stochastic)
            eta=eta
        ).prev_sample

    # Normalize image to [0, 1]
    image = (image + 1) / 2
    image = image.clamp(0, 1)

    return image.cpu().numpy()


# Function to generate a batch of augmented data
def generate_augmented_dataset(
    model,
    noise_scheduler,
    num_samples_per_class,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    # Generate images for class 0
    print("Generating samples for class 0...")
    class0_images = generate_images(
        model=model,
        noise_scheduler=noise_scheduler,
        class_label=0,
        num_inference_steps=50,
        num_images=num_samples_per_class,
        device=device
    )

    # Generate images for class 1
    print("Generating samples for class 1...")
    class1_images = generate_images(
        model=model,
        noise_scheduler=noise_scheduler,
        class_label=1,
        num_inference_steps=50,
        num_images=num_samples_per_class,
        device=device
    )

    # Combine the generated data
    generated_images = np.concatenate([class0_images, class1_images], axis=0)
    generated_labels = np.concatenate([
        np.zeros(num_samples_per_class, dtype=np.int64),
        np.ones(num_samples_per_class, dtype=np.int64)
    ])

    return generated_images, generated_labels

# Main function to set up and run the training


def main():
    # Set random seed for reproducibility
    set_seed(42)

    session = "S1"
    model = "ddim"

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load your functional ultrasound data here
    # Replace this with your actual data loading code
    # Placeholder for demonstration:
    print("Loading and preparing data...")

    train_dataset = NPZDataLoader(f'{session}_train.npz')
    val_dataset = NPZDataLoader(f'{session}_test.npz')

    # Create data loaders
    train_dataloader = DataLoader(
        train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=16, num_workers=4)

    print(
        f"Training on {len(train_dataset)} samples, validating on {len(val_dataset)} samples")

    # Train the model
    print("Starting model training...")
    model, noise_scheduler = train_diffusion_model(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        num_epochs=100,
        learning_rate=1e-4,
        save_dir=session + "_" + model,
        device=device,
        patience=20
    )

    # Generate augmented dataset for training other models
    print("Generating augmented dataset...")
    augmented_images, augmented_labels = generate_augmented_dataset(
        model=model,
        noise_scheduler=noise_scheduler,
        num_samples_per_class=500,
        device=device
    )

    # Save the augmented dataset
    print("Saving augmented dataset...")
    os.makedirs("augmented_data", exist_ok=True)
    np.save("augmented_data/augmented_images.npy", augmented_images)
    np.save("augmented_data/augmented_labels.npy", augmented_labels)

    print("Done!")


if __name__ == "__main__":
    main()
