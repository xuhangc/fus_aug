import os
import torch
import random
import numpy as np
from tqdm import tqdm
from models.cvae import CVAE
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

    session = 'S2'

    model = "CVAE"

    val_dataset = NPZDataLoader(f'{session}_test.npz')

    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize the generator
    generator = CVAE().to(device)
    
    checkpoint = torch.load(f'CVAE/{session}_best_cvae_model.pth', weights_only=True, map_location=device
                            if torch.cuda.is_available() else 'cpu')
    generator.load_state_dict(checkpoint['model_state_dict'])
    generator.eval()

    os.makedirs(f"{model}", exist_ok=True)

    data_list = []
    label_list = []
    with torch.no_grad():
        for i, (imgs, labels) in enumerate(tqdm(val_dataloader)):
            data = imgs.to(device)
            labels = labels.to(device)

            # Generate samples using random latent vectors
            batch_size = data.size(0)
            random_z = torch.randn(batch_size, generator.latent_size).to(device)
            fake_imgs = generator.inference(random_z, labels).view(data.shape)
            
            # Process the generated data to match the format in test_biggan.py
            fake_imgs = fake_imgs.squeeze(0).permute(1, 2, 0)
            labels = labels.permute(1, 0)
            
            data_list.append(fake_imgs)
            label_list.append(labels)
    
    data_list = torch.cat(data_list, dim=2).cpu().numpy()
    label_list = torch.cat(label_list, dim=1).cpu().numpy()

    np.savez(f"{model}/{model}_{session}.npz", fus=data_list, label=label_list)