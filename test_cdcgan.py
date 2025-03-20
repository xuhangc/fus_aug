import os
import torch
import random
import numpy as np
from tqdm import tqdm
from models.cdcgan import Generator
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


if __name__ == "__main__":
    # Set random seed for reproducibility
    set_seed(42)

    session = 'S2'

    model = "CDCGAN"

    val_dataset = NPZDataLoader(f'{session}_test.npz')

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=8)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Hyperparameters
    latent_dim = 100
    image_size = 128
    channels = 1
    num_classes = 2
    lr = 0.0002
    beta1 = 0.5
    beta2 = 0.999
    num_epochs = 100

    # Initialize the generator
    generator = Generator(image_size=image_size, channels=channels,
                         num_classes=num_classes, latent_dim=latent_dim).to(device)
    
    checkpoint = torch.load(f'CDCGAN/{session}_generator_epoch_100.pth', weights_only=True, map_location=device
                           if torch.cuda.is_available() else 'cpu')
    generator.load_state_dict(checkpoint)
    generator.eval()

    os.makedirs(f"{model}", exist_ok=True)

    data_list = []
    label_list = []
    with torch.no_grad():
        for i, (imgs, labels) in enumerate(tqdm(val_dataloader)):
            real_imgs = imgs.to(device)
            labels = labels.to(device)

            # Generate random latent vectors
            z = torch.randn(imgs.size(0), latent_dim, device=device)
            
            # Generate fake images
            fake_imgs = generator(z, labels)
            
            # Process the generated data to match the format in test_biggan.py
            fake_imgs = fake_imgs.squeeze(0).permute(1, 2, 0)
            labels = labels.permute(1, 0)

            data_list.append(fake_imgs)
            label_list.append(labels)
    
    data_list = torch.cat(data_list, dim=2).cpu().numpy()
    label_list = torch.cat(label_list, dim=1).cpu().numpy()

    np.savez(f"{model}/{model}_{session}.npz", fus=data_list, label=label_list)