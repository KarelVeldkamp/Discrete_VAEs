from torch.utils.data import Dataset
import pandas as pd
import torch


class MemoryDataset(Dataset):
    """
    Torch dataset for item response data in numpy array
    """
    def __init__(self, X, device='cpu'):
        """
        initialize
        :param file_name: path to csv that contains NXI matrix of responses from N people to I items
        """
        # Read csv and ignore rownames
        self.x_train = torch.tensor(X, dtype=torch.float32)

    def __len__(self):
        return len(self.x_train)

    def __getitem__(self, idx):
        return self.x_train[idx]


