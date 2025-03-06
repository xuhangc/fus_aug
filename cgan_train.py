from models.cgan import Generator, Discriminator
import torch


if __name__ == "__main__":

    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the generator
    G = Generator().to(device)

    # Define the discriminator
    D = Discriminator().to(device)

    # Print the generator
    print(G)

    # Print the discriminator
    print(D)