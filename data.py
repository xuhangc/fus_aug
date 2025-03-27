import numpy as np
import torch
from torch.utils.data import Dataset


class NPZDataLoader(Dataset):
    def __init__(self, npz_file):
        super(NPZDataLoader, self).__init__()

        with np.load(npz_file) as data:
            self.fus_files = np.transpose(data['fus'], (2, 0, 1))
            self.lab_files = np.transpose(data['label'], (1, 0))

        self.sizex = len(self.fus_files)  # get the size of target

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex

        fus = self.fus_files[index_]
        fus = fus[:128, :]
        lab = self.lab_files[index_]

        fus_tensor = torch.from_numpy(fus).float()
        lab_tensor = torch.from_numpy(lab).long()

        # min_vals = torch.min(fus_tensor)
        max_vals = torch.max(fus_tensor)

        # fus_tensor = (fus_tensor - min_vals) / (max_vals - min_vals + 1e-8)
        fus_tensor /= max_vals

        return fus_tensor.unsqueeze(0), lab_tensor


# train_dataset = NPZDataLoader('S1_train.npz')
# from torch.utils.data import DataLoader
# dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)

# # # Example usage (remove in production)
# for i, (batch_features, batch_labels) in enumerate(dataloader):
#     print(f"Batch {i+1}:")
#     print(f"Feature batch shape: {batch_features.shape}") 
#     print(f"Label batch shape: {batch_labels.shape}")
#     if i == 0:  # Show only first batch for demonstration
#         break