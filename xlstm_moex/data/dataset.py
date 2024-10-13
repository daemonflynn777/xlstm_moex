import numpy as np
import torch
from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
    """Dataset in format for neural networks:
    X:                      y:
    [                       [
        [[1], [2], [3]],        [10],
        [[4], [5], [6]],        [11],
        [[7], [8], [9]],        [12],
    ]                       ]
    """
    def __init__(self, X: np.array, y: np.array, device: str):
        super().__init__()
        seq_lenght = X.shape[1]
        self.X = (
            torch
            .from_numpy(X.astype('float32'))
            .to(device=device)
            .reshape(-1, seq_lenght , 1)
        )  # shape is (num_examples, seq_lenght, in_features) 
        self.y = torch.from_numpy(y.astype('float32')).to(device=device)

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index], self.y[index].unsqueeze(0)