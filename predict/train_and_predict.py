import argparse
import argcomplete
import pandas as pd
import json
import os

from sklearn.model_selection import train_test_split
from predictor.kmeans_predictor import Kmeans_predictor
from kmeans_as import Kmeans_predictor
from predictor.autofolio_predictor import Autofolio_predictor
from tqdm import tqdm

from helper import set_seed

SEED = 42

def get_features(instances, features) -> 'list[dict]':
    return [{
        "inst": inst[0], 
        "features": features[features["inst"] == inst[0]].to_numpy()[0][1:].tolist(), 
        "times": {t["combination"]: t["time"] for t in inst[1]}
    } for inst in instances]

def prerpare_k_means_data(dataset, features):
    train_data = []
    for datapoint in dataset:
        if features[features["inst"] == datapoint["instance_name"]].empty or features[features["inst"] == datapoint["instance_name"]].isna().any().any():
            continue
        train_data.append({
            "inst": datapoint["instance_name"],
            "times": {t["combination"]:t["time"] for t in datapoint["all_times"]}
        })

    return train_data

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--type", choices=["kmeans", "autofolio"], help="the heuristic to use to make a choiche", required=True)
parser.add_argument("-f", "--features", type=str, help="The features to use (in csv format) with the heuristic", required=True)
parser.add_argument("-d", "--dataset", type=str, help="The dataset to use (in json format)", required=True)
parser.add_argument("--history", type=str, help="The history file (in json format)", required=False)
parser.add_argument("-v", "--validation-split", type=float, help="The amount of data to use in validation", required=False, default=.1)
parser.add_argument("--max_threads", type=int, help="The maximum number of threads to use with Autofolio. Default is 12", required=False)
parser.add_argument("--time", default=False, help="Whether the script shoud print the time required to get the predictions or not. Default = False", action='store_true')
parser.add_argument("-b","--base_folder", type=str, help="base folder")
parser.add_argument("-n","--name", type=str, help="model name", required=False)
parser.add_argument("-s","--random-seed", type=int, help="random seed to use", required=False, default=42)
parser.add_argument("--tune", default=False, help="If tune autofolio or not. Default = False", action='store_true')
argcomplete.autocomplete(parser)

def main():
    arguments = parser.parse_args()
    f = open(arguments.dataset + '_train.json')
    train_dataset = json.load(f)
    f.close()
    f = open(arguments.dataset + '_test.json')
    test_dataset = json.load(f)
    f.close()

    history = None
    if arguments.history is not None:
        f = open(arguments.history)
        history = json.load(f)
        f.close()

    set_seed(arguments.random_seed)

    validation_split = arguments.validation_split
    features_split = arguments.features.split('/')
    original_train_features = pd.read_csv('/'.join(features_split[:-1]) + '/train_' + features_split[-1])
    original_test_features = pd.read_csv('/'.join(features_split[:-1]) + '/test_' + features_split[-1])
    save_folder = arguments.base_folder
    
    # Create save folder if it doesn't exist
    os.makedirs(save_folder, exist_ok=True)
    
    # (x_train, _), (x_validation, _), (x_test, _) = get_dataloader(dataset, dataset, [fold])
    x_train, x_validation = train_test_split(train_dataset, test_size=validation_split)
    train_instances = [(x["instance_name"], x["all_times"]) for x in x_train]
    validation_instances = [(x["instance_name"], x["all_times"]) for x in x_validation]
    test_instances = [(x["instance_name"], x["all_times"]) for x in test_dataset]

    idx2comb = {idx:comb for idx, comb in enumerate(sorted([t["combination"] for t in x_train[0]["all_times"]]))}

    train_data = prerpare_k_means_data(x_train, original_train_features)

    validation_data = []
    validation_data = prerpare_k_means_data(x_validation, original_train_features)

    times = {}
    for datapoint in train_dataset + test_dataset:
        times[datapoint["instance_name"]] = {t["combination"]:t["time"] for t in datapoint["all_times"]}

    opt_times = {comb["combination"]:0 for comb in train_dataset[0]["all_times"]}
    for datapoint in train_dataset:
        for t in datapoint["all_times"]:
            opt_times[t["combination"]] += t["time"]
    sb_key = min(opt_times.items(), key = lambda x: x[1])[0]

    predictor = None
    if arguments.type == "kmeans":
        pbar = tqdm(total=1368)
        predictor = Kmeans_predictor(training_data=train_data, 
                                     validation_data=validation_data,
                                     shared_tqdm=pbar,
                                     idx2comb=idx2comb, 
                                     features=original_train_features, 
                                     seed=SEED,
                                     max_threads=arguments.max_threads if arguments.max_threads is not None else 12) 
    elif arguments.type == "autofolio":
        predictor = Autofolio_predictor(training_data=train_data, 
                                   features=original_train_features, 
                                   max_threads=arguments.max_threads if arguments.max_threads is not None else 12, 
                                   pre_trained_model=None, 
                                   tune=arguments.tune,
                                   model_name=arguments.name)
    else:
        raise Exception(f"predictor_type {arguments.type} unrecognised")

    if history is not None:
        feature_configuration_results = [fold['score'] for fold in history[str(original_train_features.shape[1] - 1)]]
        assert predictor.par10score in feature_configuration_results, (predictor.par10score, feature_configuration_results)

    total_time = 0
    sb_tot = 0
    train_features = [
        {"inst": inst[0], "features": original_train_features[original_train_features["inst"] == inst[0]].to_numpy()[0].tolist()} 
        for inst in train_instances 
        if not original_train_features[original_train_features["inst"] == inst[0]].empty
    ]

    for i in range(len(train_features)):
        train_features[i]["features"].pop(train_features[i]["features"].index(train_features[i]["inst"]))
        _ = [float(e) for e in train_features[i]["features"]]
    
    predictions = predictor.predict(train_features)
    assert isinstance(predictions, list)
    if len(predictions) !=  len(train_features):
        for inst in train_instances:
            if original_train_features[original_train_features["inst"] == inst[0]].empty:
                predictions.append({"chosen_option": sb_key, "inst": inst[0], "time": 0})

    f = open(f"{save_folder}/train_predictions_{arguments.random_seed}", "w")
    json.dump(predictions, f)
    f.close()

    val_features = [
        {"inst": inst[0], "features": original_train_features[original_train_features["inst"] == inst[0]].to_numpy()[0].tolist()} 
        for inst in validation_instances
        if not original_train_features[original_train_features["inst"] == inst[0]].empty
    ]
    for i in range(len(val_features)):
        val_features[i]["features"].pop(val_features[i]["features"].index(val_features[i]["inst"]))
        _ = [float(e) for e in val_features[i]["features"]]
    predictions = predictor.predict(val_features)
    assert isinstance(predictions, list)
    if len(predictions) !=  len(val_features):
        for inst in validation_instances:
            if original_train_features[original_train_features["inst"] == inst[0]].empty:
                predictions.append({"chosen_option": sb_key, "inst": inst[0], "time": 0})

    f = open(f"{save_folder}/validation_predictions_{arguments.random_seed}", "w")
    json.dump(predictions, f)
    f.close()

    test_features = [
        {"inst": inst[0], "features": original_test_features[original_test_features["inst"] == inst[0]].to_numpy()[0].tolist()} 
        for inst in test_instances
        if not original_test_features[original_test_features["inst"] == inst[0]].empty
    ]
    for i in range(len(test_features)):
        test_features[i]["features"].pop(test_features[i]["features"].index(test_features[i]["inst"]))
        _ = [float(e) for e in test_features[i]["features"]]
    predictions = predictor.predict(test_features)
    assert isinstance(predictions, list)
    if len(predictions) !=  len(test_features):
        for inst in test_instances:
            if original_test_features[original_test_features["inst"] == inst[0]].empty:
                predictions.append({"chosen_option": sb_key, "inst": inst[0], "time": 0})

    test_time = 0
    sb_test = 0
    for pred in predictions:
        sb_test += times[pred["inst"]][sb_key]
        test_time += times[pred["inst"]][pred["chosen_option"]]
    total_time += test_time
    sb_tot += sb_test
    f = open(f"{save_folder}/test_predictions_{arguments.random_seed}", "w")
    json.dump(predictions, f)
    f.close()
    
    print(f"""
test: {test_time/sb_test:,.2f}
          """)
   
if __name__ == "__main__":
    main()
