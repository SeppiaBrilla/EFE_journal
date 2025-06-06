from typing import Any
from os.path import exists
import torch.nn as nn
import argcomplete
import numpy as np
import argparse
import torch
from json import loads, dump
from helper import dict_lists_to_list_of_dicts, is_competitive, set_seed, split_data, is_competitive, Dataset
from torch.utils.data import DataLoader
from transformers import  BertTokenizer
import pandas as pd
import numpy as np
from kmeans_as import Kmeans_predictor
from sklearn.decomposition import PCA
from competitive_model import CompetitiveModel
from training import train
from tqdm import tqdm

BERT_TYPE = "bert-base-uncased"

def initialize_with_activation_pca(source_layer: nn.Linear, target_layer: nn.Linear, input_data: torch.Tensor):
    """
    Initializes the target layer using PCA on the activations of the source layer.

    Args:
        source_layer (nn.Linear): Pretrained or initialized large layer.
        target_layer (nn.Linear): Smaller layer to initialize.
        input_data (torch.Tensor): Representative input data of shape [N, in_features].
    """
    assert input_data.shape[1] == source_layer.in_features, "Input data must match input features of source layer"
    assert source_layer.weight.shape[0] > target_layer.weight.shape[0], "Source layer must be larger than target"

    # Get activations from the source layer
    with torch.no_grad():
        source_activations = source_layer(input_data).cpu().numpy()  # shape: [N, source_out_features]

        # Apply PCA to reduce activations
        pca = PCA(n_components=target_layer.out_features)
        reduced_activations = pca.fit_transform(source_activations)  # shape: [N, target_out_features]

        # Learn a projection from input_data to reduced_activations using least squares
        X = input_data.cpu().numpy()  # shape: [N, in_features]
        Y = reduced_activations       # shape: [N, target_out_features]

        # Solve for W in Y ≈ X @ W.T  =>  W = (X^T X)^(-1) X^T Y
        W = torch.linalg.lstsq(torch.tensor(X), torch.tensor(Y)).solution.T  # shape: [target_out_features, in_features]

        # Initialize the target layer
        target_layer.weight.data = W.to(target_layer.weight.dtype)

def prepare_nn_data(data:'list[dict]'):

    tokenizer = BertTokenizer.from_pretrained(BERT_TYPE, clean_up_tokenization_spaces=True, model_max_length=2048)
    instances_and_model = [d["instance_value_json"] for d in data]

    x = dict_lists_to_list_of_dicts(tokenizer(instances_and_model, padding=True, truncation=True, return_tensors='pt'))
    y = []

    combinations = [d["combination"] for d in sorted(data[0]["all_times"], key= lambda x: x["combination"])]
    for datapoint in data:
        y_datapoint = sorted(datapoint["all_times"], key= lambda x: x["combination"])
        datapoint["all_times"] = y_datapoint
        ordered_times = [d["time"] for d in datapoint["all_times"]]
        ordered_times = sorted(ordered_times)
        vb = min([d["time"] for d in y_datapoint])
        competitivness = [1. if is_competitive(vb, d["time"]) else 0. for d in y_datapoint if d["combination"] in combinations]
        y.append(torch.Tensor(competitivness))

    return (x, y), combinations

def prerpare_k_means_data(dataset, features):

    idx2comb = {idx:comb for idx, comb in enumerate(sorted([t["combination"] for t in dataset[0]["all_times"]]))}
    train_data = []
    for datapoint in dataset:
        if features[features["inst"] == datapoint["instance_name"]].empty or features[features["inst"] == datapoint["instance_name"]].isna().any().any():
            continue
        train_data.append({
            "inst": datapoint["instance_name"],
            "times": {t["combination"]:t["time"] for t in datapoint["all_times"]}
        })

    return train_data, idx2comb

def get_bounds(lst, number):
    lst = sorted(set(lst))
    lower = 0
    upper = number

    for val in lst:
        if val < number:
            lower = val
        elif val > upper:
            upper = val
            break
        else:
            upper = number

    return lower, upper

import torch

def cross_validation_evaluation(model_hyperparam:dict, 
                                learning_rate:float, 
                                x:list, y:list,
                                k_means_data:list,
                                epochs:int, 
                                batch_size:int,
                                device, 
                                initialization:bool=False,
                                init_models:None|list[CompetitiveModel]=None) -> tuple[list[CompetitiveModel], float, list[dict]]:

    total = (5 * epochs) + (5 * 1368) 
    if init_models is not None and initialization:
        total += epochs // 2
        print("initializing the encoder and the feature network")
        assert len(init_models) == 5, f'too few models found: {len(init_models)}'

    history = []
    scores = []
    models = []
    with tqdm(total=total) as pbar:
        for fold in range(5):
            x_train, x_val = split_data(x, fold, buckets=5)
            y_train, y_val = split_data(y, fold, buckets=5)

            train_dataloader = DataLoader(Dataset(x_train, y_train), batch_size=batch_size, shuffle=True)
            validation_dataloader = DataLoader(Dataset(x_val, y_val), batch_size=batch_size, shuffle=True)

            model = CompetitiveModel(**model_hyperparam)
            history_pretrain = None
            if init_models is not None and initialization:
                model.bert.load_state_dict(init_models[fold].bert.state_dict())
                for param in model.bert.parameters():
                    param.requires_grad = False
                training_data = torch.tensor(init_models[fold].get_encoder_features(x_train, device), device=device)
                features_layer = init_models[fold].features.to(device)
                initialize_with_activation_pca(features_layer, model.features, training_data)
                del features_layer
                for param in model.features.parameters():
                    param.requires_grad = False
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate/2)
                model, history_pretrain, _ = train(model=model, 
                                                train_dataset=train_dataloader, 
                                                validation_dataset=validation_dataloader, 
                                                optimizer=optimizer, 
                                                loss=nn.functional.binary_cross_entropy_with_logits, 
                                                epochs=epochs // 10, 
                                                device=device, 
                                                hyperparam=model_hyperparam,
                                                patience=5,
                                                shared_tqdm=pbar,
                                                model_class=CompetitiveModel)
                for param in model.features.parameters():
                    param.requires_grad = True
                for param in model.bert.parameters():
                    param.requires_grad = True

            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            model, training_history, _ = train(model=model, 
                                                train_dataset=train_dataloader, 
                                                validation_dataset=validation_dataloader, 
                                                optimizer=optimizer, 
                                                loss=nn.functional.binary_cross_entropy_with_logits, 
                                                epochs=epochs, 
                                                device=device, 
                                                hyperparam=model_hyperparam,
                                                patience=5,
                                                shared_tqdm=pbar,
                                                model_class=CompetitiveModel)

            cpu_model = CompetitiveModel(**model_hyperparam)
            cpu_model.load_state_dict(model.state_dict())
            models.append(cpu_model)
            total_history:dict[str,Any] = {'training':training_history}
            if history_pretrain is not None:
                total_history['pretrain'] = history_pretrain
            # features = pd.DataFrame(model.get_features(k_means_data, device))
            # del model
            # torch.cuda.empty_cache()
            # k_data, idx2comb = prerpare_k_means_data(k_means_data, features)
            # k_train, k_val = split_data(k_data, fold, buckets=5)

            # kmeans_model = Kmeans_predictor(training_data=k_train,
            #                                 validation_data=k_val,
            #                                 idx2comb=idx2comb, 
            #                                 features=features, 
            #                                 shared_tqdm=pbar,
            #                                 max_threads=12)
            score =1 # kmeans_model.par10score
            scores.append(score)
            total_history['score'] = score
            history.append(total_history)
    return (models, float(np.mean(scores)), history)

def progressively_train(k_means_data:list[dict], 
                        starting_feature_size:int, 
                        learning_rate:float, 
                        nn_data:tuple[list[dict],list[dict]], 
                        epochs:int, 
                        batch_size:int,
                        device, 
                        combinations:list[str], 
                        max_retries:int=10,
                        partial_scores:dict|None=None,
                        initialization:bool=False,
                        partial_scores_file:str|None=None) -> tuple[int, float, list[CompetitiveModel], dict]:
    x, y = nn_data 

    length = len(combinations)
    total_history = {}
    if partial_scores is None:
        feature_size = starting_feature_size
        scores = {} 
        current_try = 0
        best_score = (0, np.inf)
    else:
        current_try = len(partial_scores.keys())
        partial_scores = {int(k): v for k, v in partial_scores.items()}
        scores = partial_scores 
        best_score = sorted(partial_scores.items(), key=lambda x: x[1])[0]
        tried_sizes = list(sorted(partial_scores.keys()))
        if best_score[0] == min(partial_scores.keys()):
            feature_size = best_score[0] // 2
        else:
            second_bound = tried_sizes[tried_sizes.index(best_score[0]) - 1]
            feature_size = (best_score[0] + second_bound) // 2
        print(f'trying feature siz: {feature_size}')

    best_models:list[CompetitiveModel]|None = None
    while True:
        trained_models, score, historties = cross_validation_evaluation({'feature_size':feature_size, 'output_size':length},
                                                                        learning_rate=learning_rate,
                                                                        x=x,
                                                                        y=y,
                                                                        batch_size=batch_size,
                                                                        k_means_data=k_means_data,
                                                                        epochs=epochs,
                                                                        device=device,
                                                                        initialization=initialization,
                                                                        init_models=best_models)
        scores[feature_size] = score

        if feature_size == 100:
            for i, model in enumerate(trained_models):
                torch.save(model.state_dict(), f'Initial_model_{i}')
            raise Exception()


        tried_features = list(scores)
        l, u = get_bounds(tried_features, feature_size)

        total_history[feature_size] = historties
        last_size = feature_size
        if score < best_score[1]:
            best_score = (feature_size, score)
            best_models = []
            for model in trained_models:
                best_model = CompetitiveModel(feature_size, length)
                best_model.load_state_dict(model.state_dict())
                best_models.append(best_model)
            feature_size = (feature_size + l) // 2
        elif score == best_score[1]:
            best_score = (feature_size, score)
            best_models = []
            for model in trained_models:
                best_model = CompetitiveModel(feature_size, length)
                best_model.load_state_dict(model.state_dict())
                best_models.append(best_model)
            feature_size = (feature_size + u) // 2
        elif best_score[1] < score:
            feature_size = (feature_size + u) // 2

        if current_try >= max_retries:
            print("max number of trains reached. Exiting")
            assert isinstance(best_score[0], int) and isinstance(best_score[1], float) and isinstance(best_models, list)
            return (best_score[0], best_score[1], best_models, total_history)

        if feature_size in scores:
            assert isinstance(best_score[0], int) and isinstance(best_score[1], float) and isinstance(best_models, list)
            print(f"feature size {feature_size} already tried. Exiting")
            return (best_score[0], best_score[1], best_models, total_history)
        if abs(feature_size - last_size) < 5:
            assert isinstance(best_score[0], int) and isinstance(best_score[1], float) and isinstance(best_models, list)
            print(f"feature size {feature_size} too close to the last tried ({last_size}). Exiting")
            return (best_score[0], best_score[1], best_models, total_history)

        print(f"Feature size {last_size} scored {score}. The best score is: {best_score[1]} with {best_score[0]} features. Trying new feature size: {feature_size}")
        if partial_scores_file is not None:
            with open(partial_scores_file, 'w') as f:
                dump(scores, f)

        current_try += 1

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True)
parser.add_argument("--epochs", type=int, required=True)
parser.add_argument("--learning_rate", type=float, required=True)
parser.add_argument("--save", required=True)
parser.add_argument("--history", required=False)
parser.add_argument("--initial-size", required=False, type=int)
parser.add_argument("--batch-size", required=False, type=int, default=4)
parser.add_argument("--partial-scores", required=False, type=str)
parser.add_argument("--initialization", required=False, action='store_true')

argcomplete.autocomplete(parser)
def main():

    arguments = parser.parse_args()
    dataset = arguments.dataset
    epochs = arguments.epochs
    learning_rate = arguments.learning_rate
    save_file = arguments.save
    history_file = arguments.history
    feature_size = arguments.initial_size
    partial_scores_file = arguments.partial_scores
    initialization = arguments.initialization
    
    partial_scores:dict|None = None


    if partial_scores_file is not None:
        if exists(partial_scores_file):
            with open(partial_scores_file) as f:
                partial_scores = loads(f.read())

    batch_size = arguments.batch_size

    f = open(dataset)
    data = loads(f.read())
    f.close()

    set_seed(42)

    (x, y), combinations = prepare_nn_data(data)
    tokenizer = BertTokenizer.from_pretrained(BERT_TYPE, clean_up_tokenization_spaces=True, model_max_length=2048)
    for datapoint in data:
        datapoint["token"] = tokenizer(datapoint["instance_value_json"], truncation=True, return_tensors="pt")



    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("operating on device:", device)
    result =  progressively_train(k_means_data=data,
                                  starting_feature_size=feature_size,
                                  batch_size=batch_size,
                                  learning_rate=learning_rate,
                                  nn_data=(x,y),
                                  epochs=epochs,
                                  device=device,
                                  partial_scores=partial_scores,
                                  partial_scores_file=partial_scores_file,
                                  initialization=initialization,
                                  combinations=combinations)
    (feature_size, score, best_models, history) = result
    print(f'finished training with score: {score} and feature size: {feature_size}')
    with open(save_file, 'w') as f:
        dump({'feature_size':feature_size, 'score':score}, f)
    for i in range(5):
        torch.save(best_models[i].state_dict(), f'model_{i}')

    with open(history_file, 'w') as f:
        dump(history, f)
    
main()