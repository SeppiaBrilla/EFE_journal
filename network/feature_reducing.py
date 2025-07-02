from typing import Any
import torch.nn as nn
import argcomplete
import numpy as np
import argparse
import pandas as pd
import torch
from json import loads, dump
from helper import dict_lists_to_list_of_dicts, set_seed, split_data, Dataset
from torch.utils.data import DataLoader
from transformers import  BertTokenizer
import numpy as np
from kmeans_as import Kmeans_predictor
from models import CompetitiveModel
from training import remove, train, to
from tqdm import tqdm
import torch.nn.functional as F

BERT_TYPE = "bert-base-uncased"

class ReducerAdapter(nn.Module):
    def __init__(self, input_size:int, reduced_size:int) -> None:
        super().__init__()
        middle_size = (input_size + reduced_size) // 2
        self.input = nn.Linear(input_size, middle_size)
        self.reducing = nn.Linear(middle_size, reduced_size)
        self.reduced = nn.Linear(reduced_size, middle_size)
        self.output = nn.Linear(middle_size, input_size) 

    def forward(self, inputs) -> torch.Tensor:
        ins = F.relu(self.input(inputs))
        reducing = F.tanh(self.reducing(ins))
        reduced = F.relu(self.reduced(reducing))
        out = F.tanh(self.output(reduced))
        return out

class MorfedModel(nn.Module):
    def __init__(self, baseModel:CompetitiveModel, adapter:ReducerAdapter) -> None:
        super().__init__()
        self.bert = baseModel.bert
        self.feature_layer = baseModel.features
        self.adapter_input = adapter.input
        self.adapter_reducing = adapter.reducing

    def forward(self, inputs) -> torch.Tensor:
        _, encoded_input = self.bert(**inputs, return_dict = False)
        out = self.feature_layer(encoded_input)
        out = F.tanh(out)
        adapter_in = F.relu(self.adapter_input(out))
        return F.tanh(self.adapter_reducing(adapter_in))

    def get_features(self, dataset:list[dict], device):
        model = self.to(device)
        features = []
        for datapoint in dataset:
            tokenized_instance = {k:datapoint["token"][k].to(device) for k in datapoint["token"].keys()}
            feats = model(tokenized_instance)[0].detach().cpu()
            keys = list(tokenized_instance.keys())
            for k in keys:
                del tokenized_instance[k]
            out = {}
            for i in range(len(feats)):
                out[f"feat{i}"] = round(float(feats[i]), 3)

            out["inst"] = datapoint["instance_name"]
            features.append(out)
        del model
        return features

def get_model_features(model, inputs):
    _, encoded_input = model.bert(**inputs, return_dict = False)
    out = model.features(encoded_input)
    features = model.activation(out)
    return features.cpu()[0]

def prepare_nn_data(data:list[dict], model:CompetitiveModel, device:torch.device) -> tuple[list, list]:

    tokenizer = BertTokenizer.from_pretrained(BERT_TYPE, clean_up_tokenization_spaces=True, model_max_length=2048)
    instances_and_model = [d["instance_value_json"] for d in data]

    tokenized_instances = dict_lists_to_list_of_dicts(tokenizer(instances_and_model, padding=True, truncation=True, return_tensors='pt'))
    x, y = [], []
    model = model.to(device)
    for instance in tqdm(tokenized_instances):
        instance_device = to({k: v.view(1,-1) for k,v in instance.items()} , device)
        with torch.no_grad():
            model_predictions = get_model_features(model, instance_device)
        remove(instance_device)
        x.append(model_predictions)
        y.append(model_predictions)
    del model
    return (x, y)

def prepare_k_means_data(dataset, features):

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
                                base_model:CompetitiveModel,
                                device: torch.device) -> tuple[list[ReducerAdapter], float, list[dict]]:

    total = (10 * epochs) + (10 * 1368) 

    history = []
    scores = []
    models = []
    with tqdm(total=total) as pbar:
        for fold in range(10):
            x_train, x_val = split_data(x, fold, buckets=10)
            y_train, y_val = split_data(y, fold, buckets=10)

            train_dataloader = DataLoader(Dataset(x_train, y_train), batch_size=batch_size, shuffle=True)
            validation_dataloader = DataLoader(Dataset(x_val, y_val), batch_size=batch_size, shuffle=True)

            model = ReducerAdapter(**model_hyperparam)

            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            model, training_history, best_loss = train(model=model, 
                                                train_dataset=train_dataloader, 
                                                validation_dataset=validation_dataloader, 
                                                optimizer=optimizer, 
                                                loss=nn.functional.mse_loss, 
                                                epochs=epochs, 
                                                device=device, 
                                                hyperparam=model_hyperparam,
                                                patience=5,
                                                shared_tqdm=pbar,
                                                model_class=ReducerAdapter)

            print('best loss:', best_loss)
            cpu_model = ReducerAdapter(**model_hyperparam)
            cpu_model.load_state_dict(model.state_dict())
            models.append(cpu_model)
            total_history:dict[str,Any] = {'training':training_history}
            assert isinstance(model, ReducerAdapter)
            morfed_model = MorfedModel(base_model, model)
            features = pd.DataFrame(morfed_model.get_features(k_means_data, device))
            del model
            del morfed_model
            torch.cuda.empty_cache()
            k_data, idx2comb = prepare_k_means_data(k_means_data, features)
            k_train, k_val = split_data(k_data, fold, buckets=10)

            kmeans_model = Kmeans_predictor(training_data=k_train,
                                            validation_data=k_val,
                                            idx2comb=idx2comb, 
                                            features=features, 
                                            shared_tqdm=pbar,
                                            max_threads=12)
            score = kmeans_model.par10score
            scores.append(score)
            total_history['score'] = score
            history.append(total_history)
    return (models, float(np.mean(scores)), history)

def progressively_train(k_means_data:list[dict], 
                        input_size:int, 
                        learning_rate:float, 
                        nn_data:tuple[list[dict],list[dict]], 
                        epochs:int, 
                        batch_size:int,
                        device:torch.device, 
                        base_model:CompetitiveModel,
                        max_retries:int=10) -> tuple[int, float, list[ReducerAdapter], dict]:
    x, y = nn_data 

    total_history = {}
    feature_size = input_size
    scores = {} 
    current_try = 0
    best_score = (0, np.inf)
    best_models = []
    while True:
        trained_models, score, historties = cross_validation_evaluation({'input_size':100, 'reduced_size':feature_size},
                                                                        learning_rate=learning_rate,
                                                                        x=x,
                                                                        y=y,
                                                                        batch_size=batch_size,
                                                                        k_means_data=k_means_data,
                                                                        epochs=epochs,
                                                                        base_model=base_model,
                                                                        device=device)
        scores[feature_size] = score

        tried_features = list(scores)
        l, u = get_bounds(tried_features, feature_size)

        total_history[feature_size] = historties
        last_size = feature_size
        if score < best_score[1]:
            best_score = (feature_size, score)
            best_models = []
            for model in trained_models:
                best_model = ReducerAdapter(input_size, feature_size)
                best_model.load_state_dict(model.state_dict())
                best_models.append(best_model)
            feature_size = (feature_size + l) // 2
        elif score == best_score[1]:
            best_score = (feature_size, score)
            best_models = []
            for model in trained_models:
                best_model = ReducerAdapter(input_size, feature_size)
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
        current_try += 1

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True)
parser.add_argument("--epochs", type=int, required=True)
parser.add_argument("--learning_rate", type=float, required=True)
parser.add_argument("--save", required=True)
parser.add_argument("--history", required=False)
parser.add_argument("--initial-size", required=False, type=int)
parser.add_argument("--batch-size", required=False, type=int, default=4)
parser.add_argument("--base-model", required=True, type=str)

argcomplete.autocomplete(parser)
def main():

    arguments = parser.parse_args()
    dataset = arguments.dataset
    epochs = arguments.epochs
    learning_rate = arguments.learning_rate
    save_file = arguments.save
    history_file = arguments.history
    feature_size = arguments.initial_size
    batch_size = arguments.batch_size

    f = open(dataset)
    data = loads(f.read())
    f.close()

    combinations = [d["combination"] for d in sorted(data[0]["all_times"], key= lambda x: x["combination"])]
    base_model = CompetitiveModel(feature_size=100, output_size=len(combinations))
    base_model.load_state_dict(torch.load(arguments.base_model, weights_only=True))
    set_seed(42)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("operating on device:", device)

    x, y = prepare_nn_data(data, base_model, device)
    tokenizer = BertTokenizer.from_pretrained(BERT_TYPE, clean_up_tokenization_spaces=True, model_max_length=2048)
    for datapoint in data:
        datapoint["token"] = tokenizer(datapoint["instance_value_json"], truncation=True, return_tensors="pt")

    result =  progressively_train(k_means_data=data,
                                  input_size=feature_size,
                                  batch_size=batch_size,
                                  learning_rate=learning_rate,
                                  nn_data=(x,y),
                                  epochs=epochs,
                                  device=device,
                                  base_model=base_model)
    (feature_size, score, best_models, history) = result
    print(f'finished training with score: {score} and feature size: {feature_size}')
    with open(save_file, 'w') as f:
        dump({'feature_size':feature_size, 'score':score}, f)
    for i in range(10):
        torch.save(best_models[i].state_dict(), f'model_{i}')

    with open(history_file, 'w') as f:
        dump(history, f)
    
main()
