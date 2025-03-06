from models.cvae import CVAE
import torch

if __name__ == "__main__":
    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the conditional variational autoencoder
    cvae = CVAE().to(device)

    # Print the conditional variational autoencoder
    print(cvae)