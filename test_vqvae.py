import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import torch
from models.vqvae import VQVAE
import random
import numpy as np
from data import NPZDataLoader
from torch.utils.data import DataLoader


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


if __name__ == "__main__":
    # Set random seed for reproducibility
    set_seed(42)

    session = 'S1'

    model = "VQVAE"

    val_dataset = NPZDataLoader(f'{session}_test.npz')

    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize the generator and discriminator
    generator = VQVAE().to(device)
    
    checkpoint = torch.load(f'VQVAE/{session}_best_vqvae_model.pth', weights_only=True, map_location=device
                            if torch.cuda.is_available() else 'cpu')
    generator.load_state_dict(checkpoint['model_state_dict'])
    generator.eval()

    n_class0 = 8
    n_class1 = 8

    samples_class0 = []
    samples_class1 = []

    with torch.no_grad():

        for i in tqdm(val_dataloader):
            data, labels = i
            data = data.to(device)
            labels = labels.to(device)

            if labels.item() == 0 and n_class0 > 0:
                n_class0 -= 1
            elif labels.item() == 1 and n_class1 > 0:
                n_class1 -= 1
            else:
                continue

            recon_data, vq_loss, perplexity, classification = generator(data)

            samples = recon_data.reshape(1, 1, 128, 128)

            if labels.item() == 0:
                samples_class0.append(samples)
            else:
                samples_class1.append(samples)

    samples_class0 = torch.cat(samples_class0, dim=0)
    samples_class1 = torch.cat(samples_class1, dim=0)

    samples_class0 = samples_class0.cpu()
    samples_class1 = samples_class1.cpu()

    samples_class0 = (samples_class0 + 1) / 2.0
    samples_class1 = (samples_class1 + 1) / 2.0

    # Plot results
    fig, axs = plt.subplots(2, 8, figsize=(15, 6))

    for j in range(8):
        # Class 0 samples
        axs[0, j].imshow(samples_class0[j, 0].numpy(), cmap='hot')
        axs[0, j].axis('off')
        if j == 0:
            axs[0, j].set_title('Class 0')

        # Class 1 samples
        axs[1, j].imshow(
            samples_class1[j, 0].numpy(), cmap='hot')
        axs[1, j].axis('off')
        if j == 0:
            axs[1, j].set_title('Class 1')
    
    plt.tight_layout()
    plt.savefig(f"{session}_{model}_final_generated_samples.png")