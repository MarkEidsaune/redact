import numpy as np
import torch
import os

# Simple dataloader w/ np.memmap
data = np.memmap(os.path.join('/media/nvme2/openwebtext' 'train.bin'), dtype=np.uint16, mode='r')
def get_batch():
    ix = torch.randint(len(data) - 1024, (15,))
    x = torch.stack([torch.from_numpy((data[i:i+args.block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+args.block_size]).astype(np.int64)) for i in ix])
    # x, y = x.to(device), y.to(device)
    return x, y

print(f'Size of numpy memmap before retrieving batches: {data.nbytes}')
for i in range(10000):
    X, Y = get_batch()

print(f'Size of numpy memmap after retrieving batches: {data.nbytes}')

