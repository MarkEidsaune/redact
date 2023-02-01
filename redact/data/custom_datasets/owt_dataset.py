import torch
from torch.utils.data import Dataset
import numpy as np
import os

class OWTDataset(Dataset):
    '''
    
    '''
    def __init__(self, dir, split, block_size):
        self.dir = dir
        self.split = split
        self.block_size = block_size
        self.data = np.memmap(os.path.join(dir, 'train.bin' if split == 'train' else 'val.bin'), dtype=np.uint16, mode='r')
    
    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = torch.from_numpy((self.data[idx:idx+self.block_size]).astype(np.int64))
        y = torch.from_numpy((self.data[idx+1:idx+1+self.block_size]).astype(np.int64))
        return x, y