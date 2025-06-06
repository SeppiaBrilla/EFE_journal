import pandas as pd
import numpy as np
from sklearn.model_selection import ParameterGrid
from sklearn.cluster import KMeans
import concurrent.futures
from tqdm import tqdm
import json
from sys import stderr
from time import time

def isnan(array:list):
    for a in array:
        if str(a) == "nan":
            return True
    return False

class Kmeans_predictor:

    def __init__(self,
                 training_data:list[dict], 
                 validation_data:list[dict], 
                 idx2comb:dict[int,str], 
                 features:pd.DataFrame, 
                 shared_tqdm:tqdm,
                 max_threads:int=12,
                 n_k:int|None=None,
                 seed:int=42) -> None:
        self.training_data = training_data
        self.validation_data = validation_data
        self.idx2comb = idx2comb
        self.features_data = features
        self.max_threads = max_threads
        self.seed = seed
        self.n_k = n_k

        self._initialize_model(shared_tqdm)

    def _initialize_model(self, shared_tqdm:tqdm):
        self.comb2idx = {v:k for k,v in self.idx2comb.items()}
        train = self.training_data
        validation = self.validation_data

        training_features:list|np.ndarray = [[str(f) for f in self.features_data[self.features_data["inst"] == datapoint["inst"]].to_numpy()[0].tolist()] 
                             for datapoint in self.training_data]
        times = {opt:0 for opt in self.training_data[0]["times"].keys()}
        for i in range(len(self.training_data)):
            inst = self.training_data[i]["inst"]
            training_features[i].pop(training_features[i].index(inst))
            training_features[i] = [float(f) for f in training_features[i]]
            train[i]['features'] = training_features
            for key in self.training_data[i]["times"].keys():
                times[key] += self.training_data[i]["times"][key]

        train = self.training_data
        validation = self.validation_data


        self.sb = min(times.items())[0]

        training_features = np.array(training_features)
        hyperparameters, score = self.__get_clustering_parameters(training_features, 
                                                                  train, 
                                                                  validation, 
                                                                  self.features_data, 
                                                                  self.idx2comb,
                                                                  shared_tqdm)
        self.clustering_parameters = hyperparameters
        self.par10score = score
        self.model = KMeans(**hyperparameters)
        y_pred = self.model.fit_predict(training_features)
        stats = {i: {comb:0 for comb in self.idx2comb.values()} for i in range(hyperparameters["n_clusters"])}
        for i in range(len(train)):
            for option in train[i]["times"].keys():
                stats[int(y_pred[i])][option] += train[i]["times"][option]
        self.order = {
            str(i): 
            {k:v for k, v in sorted(stats[i].items(), key=lambda item: item[1], reverse=False)} 
            for i in range(hyperparameters["n_clusters"])
        }
        new_score = 0
        for datapoint in validation:
            datapoint_features = self.features_data[self.features_data["inst"] == datapoint["inst"]].to_numpy()[0].tolist()
            datapoint_features.pop(datapoint_features.index(datapoint["inst"]))
            datapoint_features = np.array(datapoint_features)
            preds = self.model.predict(datapoint_features.reshape(1, -1))
            datapoint_candidates = list(self.idx2comb.values())
            option = self.__get_prediction(datapoint_candidates, int(preds[0]), self.order)
            new_score += datapoint["times"][option]

        assert score == new_score

    def __get_clustering_parameters(self, training_features:'np.ndarray', 
                                    train_data:'list[dict]', 
                                    validation_data:'list[dict]', 
                                    features:'pd.DataFrame',
                                    idx2comb:'dict',
                                    shared_tqdm:tqdm) -> 'tuple[dict, float]':
        parameters = list(ParameterGrid({
            'n_clusters': range(2, 21 if self.n_k is None else self.n_k),
            'init': ['k-means++', 'random'],
            'max_iter': [100, 200, 300],
            'tol': [1e-3, 1e-4, 1e-5],
            'n_init': [5, 10, 15, 'auto'],
            'random_state': [self.seed],
            'verbose': [0]
        }))
        clusters_val = []

        with concurrent.futures.ThreadPoolExecutor(self.max_threads) as executor:
            futures = {executor.submit(
                self.__get_clustering_score, 
                params, training_features, 
                idx2comb, train_data, 
                validation_data, features): json.dumps(params)
            for params in parameters}

            for future in concurrent.futures.as_completed(futures):
                params = futures[future]
                try:
                    result = future.result()
                    clusters_val.append((json.loads(params), result))
                    shared_tqdm.update(1)
                except Exception as e:
                    print(f"An error occurred for text '{params}': {e}", file=stderr)

        best_cluster_val = min(clusters_val, key=lambda x: x[1])
        return best_cluster_val
    
    def __get_clustering_score(self, params:'dict', 
                               training_features:'np.ndarray', 
                               idx2comb:'dict', 
                               train_data:'list[dict]', 
                               validation_data:'list', 
                               features:'pd.DataFrame'):
        kmeans = KMeans(**params)
        y_pred = kmeans.fit_predict(training_features)
        stats = {i: {comb:0 for comb in idx2comb.values()} for i in range(params["n_clusters"])}
        for i in range(len(train_data)):
            for option in train_data[i]["times"].keys():
                stats[y_pred[i]][option] += train_data[i]["times"][option]
        order = {str(i): {k:v for k, v in sorted(stats[i].items(), key=lambda item: item[1], reverse=False)} for i in range(params["n_clusters"])}
        time = 0
        for datapoint in validation_data:
            datapoint_features = features[features["inst"] == datapoint["inst"]].to_numpy()[0].tolist()
            datapoint_features.pop(datapoint_features.index(datapoint["inst"]))
            datapoint_features = np.array(datapoint_features)
            preds = kmeans.predict(datapoint_features.reshape(1, -1))
            datapoint_candidates = list(idx2comb.values())
            option = self.__get_prediction(datapoint_candidates, int(preds[0]), order)
            time += datapoint["times"][option]
        return time

    def __get_prediction(self, options:'list', category:'int', order:'dict'):
        for candidate in order[str(category)]:
            if candidate in options:
                return candidate

    def predict(self, dataset:'list[dict]') -> 'list[dict]':
        """
        Original predict method.
        Given a dataset, return a list containing each prediction for each datapoint and the sum of the total predicted time.
        -------
        Parameters
            dataset:list[dict]
                A list containing, for each datapoint to predict, a list of features to use for the prediction, a dictionary containing, for each option, the corresponding time
        ------
        Output
            A tuple containing:
                - a list of dicts with, for each datapoint, the chosen option and the corresponding predicted time
        """

        predictions = []
        for datapoint in tqdm(dataset):
            feat = np.array(datapoint["features"]).reshape(1,-1)
            start = time()
            if isnan(feat.tolist()[0]):
                chosen_option = self.sb
            else:
                category = self.model.predict(feat)
                options = list(self.idx2comb.values())
                chosen_option = self.__get_prediction(options, int(category[0]), self.order)
            predictions.append({"chosen_option": chosen_option, "inst": datapoint["inst"], "time": time() - start})

        return predictions
