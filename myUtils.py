import pickle
import torch
import numpy as np
from torch.utils.data import Dataset

def save_obj(obj, path):
    if not path.endswith('.pkl'):
        path = path + '.pkl'
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def load_obj(path):
    if not path.endswith('.pkl'):
        path = path + '.pkl'
    with open(path, 'rb') as f:
        return pickle.load(f)

def set_rng_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

def check_gpu():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

class myDataset(Dataset):
    def __init__(self, X, Y=None, transform=None, train=True):
        self.X = X
        self.Y = Y
        self.transform = transform
        self.train = train

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.transform and self.train:
            x = self.transform(x)
        if self.Y is None:
            return x,
        else:
            y = self.Y[idx]
            return x,y
