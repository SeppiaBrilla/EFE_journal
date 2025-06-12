import torch, random
import numpy as np
from torch.utils.data import Dataset as ds

class Dataset(ds):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def dict_lists_to_list_of_dicts(input_dict:dict):
    """
    A function that convert a dictionary of lists into a list of dictionaries
    Parameters
    ----------
    input_dict:dict
        The dictionary to convert
    
    Outputs
    -------
    The list of dictionaries
    """
    keys = input_dict.keys()
    list_lengths = [len(input_dict[key]) for key in keys]

    if len(set(list_lengths)) > 1:
        raise ValueError("All lists in the input dictionary must have the same length.")

    list_of_dicts = [{key: input_dict[key][i] for key in keys} for i in range(list_lengths[0])]

    return list_of_dicts

def set_seed(seed=42):
    random.seed(seed)                     # Python built-in random
    np.random.seed(seed)                  # Numpy
    torch.manual_seed(seed)               # PyTorch CPU
    torch.cuda.manual_seed(seed)          # PyTorch GPU (if using)
    torch.cuda.manual_seed_all(seed)      # All GPUs (if using multi-GPU)
    
    # For deterministic behavior (may affect performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def split_data(x, fold, buckets=None):
    BUCKETS = 10 if buckets is None else buckets

    N_ELEMENTS = len(x)

    BUCKET_SIZE = N_ELEMENTS // BUCKETS

    x_local = x.copy()
    x_test = []

    idx = fold * BUCKET_SIZE
    for _ in range(BUCKET_SIZE):
        x_test.append(x_local.pop(idx))

    x_train = x_local
    
    return  x_train, x_test

def is_competitive(vb, option):
    return (option < 10 or vb * 2 >= option) and option < 3600
