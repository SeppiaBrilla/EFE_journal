import argparse
from predictor.old__clustering_predictor import Kmeans_predictor
from predictor.autofolio_predictor import Autofolio_predictor
import torch
import numpy as np
import random

def get_predictor(predictor_type:'str', 
                  train_data:'list[dict]', 
                  **kwargs) -> 'Kmeans_predictor|Autofolio_predictor':
    if predictor_type == "kmeans":
        if "idx2comb" not in kwargs:
            raise Exception(f"predictor_type {predictor_type} needs idx2comb. idx2comb cannot be None")
        if "features" not in kwargs:
            raise Exception(f"predictor_type {predictor_type} needs features. features cannot be None")
        model_name = kwargs.get("name", None)
        max_threads = kwargs["max_threads"] if "max_threads" in kwargs else 12
        hyperparameters =  kwargs["hperparameters"] if "hyperparameters" in kwargs else None
        return Kmeans_predictor(training_data=train_data, idx2comb=kwargs["idx2comb"], features=kwargs["features"], max_threads=max_threads, hyperparameters=hyperparameters, model_name=model_name) 
    elif predictor_type == "autofolio":
        if "features" not in kwargs:
            raise Exception(f"predictor_type {predictor_type} needs features. features cannot be None")
        max_threads = kwargs["max_threads"] if "max_threads" in kwargs else 12
        pre_trained_model = kwargs["pre_trained_model"] if "pre_trained_model" in kwargs else None
        model_name = kwargs.get("name", None)
        return Autofolio_predictor(training_data=train_data, 
                                   features=kwargs["features"], 
                                   max_threads=max_threads, 
                                   pre_trained_model=pre_trained_model, 
                                   tune=kwargs["tune"],
                                   model_name=model_name)
    else:
        raise Exception(f"predictor_type {predictor_type} unrecognised")

def is_competitive(vb, option):
        return (option < 10 or vb * 2 >= option) and option < 3600

def get_sb_vb(train:'list[dict]', validation:'list[dict]', test:'list[dict]') -> 'tuple[tuple[float,float],tuple[float,float],tuple[float,float]]':
    sb_train, sb_val, sb_test = 0, 0, 0
    vb_train, vb_val, vb_test = 0, 0, 0
    combinations = [t["combination"] for t in train[0]["all_times"]]
    comb_vals = {comb:0 for comb in combinations}

    for datapoint in train:
        vb_train += datapoint["time"]
        for t in datapoint["all_times"]:
             comb_vals[t["combination"]] += t["time"]
    sb_train = min(comb_vals.values())

    comb_vals = {comb:0 for comb in combinations}
    for datapoint in validation:
        vb_val += datapoint["time"]
        for t in datapoint["all_times"]:
             comb_vals[t["combination"]] += t["time"]
    sb_val = min(comb_vals.values())

    comb_vals = {comb:0 for comb in combinations}
    for datapoint in test:
        vb_test += datapoint["time"]
        for t in datapoint["all_times"]:
             comb_vals[t["combination"]] += t["time"]
    sb_test = min(comb_vals.values())

    return (sb_train, vb_train), (sb_val, vb_val), (sb_test, vb_test)

def positive_int(value):
    ivalue = int(value)
    if ivalue < 0:
        raise argparse.ArgumentTypeError(f"{value} is not a positive integer")
    return ivalue

def pad(*args) -> tuple:
    str_args = [str(arg) for arg in args]
    max_len_arg = max([len(arg) for arg in str_args])
    str_args = [f"{' '* (max_len_arg - len(arg))}{arg}" for arg in str_args]
    return tuple(str_args)

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
