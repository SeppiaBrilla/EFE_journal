from ConfigSpace import Configuration, ConfigurationSpace, Categorical, Integer
import torch
import os
import json
from competitive_model import CompetitiveModel
from kmeans_as import Kmeans_predictor
import argparse, argcomplete
from transformers import BertTokenizer
import numpy as np
import pandas as pd
from smac import HyperparameterOptimizationFacade, Scenario
from torch.utils.data import DataLoader
from training import train
from helper import set_seed, dict_lists_to_list_of_dicts, Dataset, split_data, is_competitive

def prepare_nn_data(data:'list[dict]') -> tuple[tuple[list, list], list]:

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", clean_up_tokenization_spaces=True, model_max_length=2048)
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

def prerpare_k_means_data(dataset, features, fold):
    x_train, x_validation, x_test = split_data(dataset, fold)

    idx2comb = {idx:comb for idx, comb in enumerate(sorted([t["combination"] for t in x_train[0]["all_times"]]))}
    train_data = []
    for datapoint in x_train + x_validation:
        if features[features["inst"] == datapoint["instance_name"]].empty or features[features["inst"] == datapoint["instance_name"]].isna().any().any():
            continue
        train_data.append({
            "trues": [0 if is_competitive(datapoint["time"], t["time"]) else 1 for t in sorted(datapoint["all_times"], key=lambda x: x["combination"])],
            "inst": datapoint["instance_name"],
            "times": {t["combination"]:t["time"] for t in datapoint["all_times"]}
        })

    times = {}
    for datapoint in x_train + x_validation + x_test:
        times[datapoint["instance_name"]] = {t["combination"]:t["time"] for t in datapoint["all_times"]}

    return train_data, idx2comb

def get_sb_vb(data, sb_alg, fold) -> tuple[tuple[float,float], tuple[float,float]]:

    train_elements = (len(data) // 10) * 9
    train = data[train_elements*fold :train_elements * (fold + 1)]

    validation = data[train_elements:]
    vb_train, sb_train = 0, 0 
    for val in train:
        times = {t['combination']: t['time'] for t in val['all_times']}
        sb_train += times[sb_alg]
        vb_train += min(times.values())
    vb_val, sb_val = 0, 0
    for val in validation:
        times = {t['combination']: t['time'] for t in val['all_times']}
        sb_val += times[sb_alg]
        vb_val += min(times.values())

    return (sb_train, vb_train), (sb_val, vb_val)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True)
parser.add_argument("--epochs", type=int, required=True)
parser.add_argument("--lr", type=float, required=True)
parser.add_argument("--save", required=True)
# parser.add_argument("--pre_trained", required=False)
parser.add_argument("--history", required=False)
parser.add_argument("--starting-size", required=False, type=int)
parser.add_argument("--batch-size", required=False, type=int, default=4)

argcomplete.autocomplete(parser)

def main():
    args = parser.parse_args()   
    dataset = args.dataset
    epochs = args.epochs
    lr = args.lr
    save_file = args.save
    history_file = args.history
    features_size = args.starting_size
    batch_size = args.batch_size


    f = open(dataset)
    data = json.loads(f.read())
    f.close()

    (x, y), combinations = prepare_nn_data(data)
    test_data = int(len(x) / 100 * 20)
    x_test, y_test = x[len(x) - test_data:], y[len(x) - test_data:]
    assert len(x_test) == test_data, (len(x_test), test_data)
    x_train, y_train = x[:len(x) - test_data], y[:len(x) - test_data]
    dataloaders = [get_dataloader(x_train,y_train, batch_size, fold, n_buckets=5, test=False) for fold in range(5)]
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", clean_up_tokenization_spaces=True)
    times = {}
    for datapoint in data:
        datapoint["token"] = tokenizer(datapoint["instance_value_json"], truncation=True, return_tensors="pt")
        for t in datapoint['all_times']:
            if not t['combination'] in times:
                times[t['combination']] = 0
            times[t['combination']] += t['time']
    sb_alg = min(times.items(), key= lambda x: x[1])[0]


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("operating on device:", device)
    history = {}
    def loss(pred, true):
        logits = torch.nn.functional.binary_cross_entropy_with_logits(pred, true, reduction="none")
        return torch.mean(logits)

    def trainer(config: Configuration, seed: int = 42) -> float:
        set_seed(seed)
        losses = []
        scores = []

        for fold, (train_dataloader, validation_dataloader, _) in enumerate(dataloaders):
            model = CompetitiveModel(feature_size=config['feature_size'], output_size=len(combinations))
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            best_model, training_history, best_loss = train(model=model,
                                                   train_dataset=train_dataloader,
                                                   validation_dataset=validation_dataloader,
                                                   optimizer=optimizer,
                                                   loss=loss,
                                                   epochs=epochs,
                                                   patience=3,
                                                   device=device,
                                                   model_class=CompetitiveModel,
                                                   hyperparam={"feature_size":config['feature_size'], "output_size":len(combinations)})
            # history[config['feature_size']] = training_history
            assert isinstance(best_model, CompetitiveModel)
            features = pd.DataFrame(best_model.get_features(data, device))
            k_means_train_data, idx2comb = prerpare_k_means_data(data, features, 0)

            train_elements = int(len(k_means_train_data) * .9)
            train_data = k_means_train_data[:train_elements]
            val_data = k_means_train_data[train_elements:]
            print(len(val_data))

            kmeans_model = Kmeans_predictor(training_data=train_data, validation_data=val_data, idx2comb=idx2comb, features=features, max_threads=20, seed=seed)
            score = kmeans_model.par10score
            _, (sb, vb) = get_sb_vb(data, sb_alg, fold)
            print(sb, vb, score)

            scores.append((score-vb)/(sb-vb))
            losses.append(best_loss)
        
        print(f"model with {config['feature_size']} features has a loss value of {np.mean(losses):.3f} and a par10 score of {np.mean(scores):.3f}")

        return float(np.mean(scores))

    feature_sizes = range(features_size//2, features_size + 1)
    configspace = ConfigurationSpace({"feature_size": Integer("feature_size", (features_size//2, features_size))})
    scenario = Scenario(configspace, deterministic=True, n_trials=len(feature_sizes))

    smac = HyperparameterOptimizationFacade(scenario, trainer)
    incumbent = smac.optimize()
    print("best configuration:")
    print(incumbent)
    path = ""
    for dir in save_file.split("/")[:-1]:
        path = os.path.join(path,dir)
        if not os.path.exists(path):
            os.mkdir(path)
    with open(save_file,'w') as f:
        config = []
        if isinstance(incumbent, list):
            for inc in incumbent:
                config.append({"feature_size":int(inc['feature_size'])})
        else:
            config = {"feature_size":int(incumbent['feature_size'])}
        json.dump(config, f)
    path = ""
    for dir in history_file.split("/")[:-1]:
        path = os.path.join(path,dir)
        if not os.path.exists(path):
            os.mkdir(path)
    with open(history_file,'w') as f:
        json.dump(history, f)

if __name__ == "__main__":
    main()
