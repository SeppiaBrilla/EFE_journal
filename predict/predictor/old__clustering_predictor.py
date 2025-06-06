import os
import pickle
from .base_predictor import Predictor, Predictor_initializer, isnan
from tqdm import tqdm
from sys import stderr
import concurrent.futures
import pandas as pd
import json
import numpy as np
from sklearn.model_selection import ParameterGrid
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, ClassifierMixin
import joblib
from time import time

class kmeans_initializer(Predictor_initializer):
    def __init__(self, pretrained_clustering_file_path:'str', order:'dict', idx2comb:'dict') -> None:
        super().__init__()
        self.kmeans = joblib.load(pretrained_clustering_file_path)
        self.order = order
        self.idx2comb = idx2comb

class Kmeans_predictor(Predictor, BaseEstimator, ClassifierMixin):

    MODEL_NAME = "kmeans.pkl"

    def __init__(self, training_data:'list[dict]|None'=None, 
                 validation_data:'list[dict]|None'=None, 
                 idx2comb:'dict[int,str]|None'=None, 
                 features:'pd.DataFrame|None'=None, 
                 hyperparameters:'dict|None'=None,
                 model_name:'str|None'=None,
                 max_threads:'int'=12,
        ) -> 'None':
        """
        initialize an instance of the class Kmeans_predictor.
        ---------
        Parameters
            training_data:list[dict].
                Indicates the data to use to create the ordering used to break ties
            idx2comb:dict[int,str].
                A dictionary that, for each index, returns the corresponding combination
            features:pd.DataFrame.
                a dataframe with a column indicating the instances and a feature set for each feature
            hyperparameters:dict|None. Default=None
                hyperparameters of the clustering model to use. If None, a greedy search will be done to find the best hyperparameters
            max_threads:int. Default=12
                Maximum number of threads to use for hyperparameter search
            n_clusters:int. Default=10
                Number of clusters for KMeans
            init:str. Default='k-means++'
                Method for initialization
            max_iter:int. Default=300
                Maximum number of iterations of the k-means algorithm
            tol:float. Default=1e-4
                Relative tolerance with regards to Frobenius norm of the difference in the cluster centers
            n_init:int. Default=10
                Number of time the k-means algorithm will be run with different centroid seeds
            random_state:int. Default=42
                Random seed
            verbose:int. Default=0
                Verbosity mode
        -----
        Usage
        ```py
        train_data = [{"inst": "instance name", "trues":[1,0,1,0]}]
        features = pd.DataFrame([{"inst": "instance name", "feat1":0, "feat2":0, "feat3":1, "feat4":1}])
        idx2comb = {0: "combination_0", 1:"combination_1"}
        predictor = Kmeans_predictor(train_data, idx2comb, features)
        ```
        """
        Predictor.__init__(self)
        BaseEstimator.__init__(self)
        
        # Store all parameters as attributes for sklearn compatibility
        self.training_data = training_data
        self.validation = validation_data
        self.idx2comb = idx2comb
        self.features_data = features
        self.max_threads = max_threads
        self.model_name = model_name

        # Initialize model if data is provided
        if training_data is not None and idx2comb is not None and features is not None:
            self._initialize_model(hyperparameters)

    def _initialize_model(self, hyperparameters):
        """Internal method to initialize the model from the constructor parameters"""
        self.comb2idx = {v:k for k,v in self.idx2comb.items()}
        TRAIN_ELEMENTS = int(len(self.training_data) * .9)
        if self.validation is None:
            train = self.training_data[:TRAIN_ELEMENTS]
            validation = self.training_data[TRAIN_ELEMENTS:]
        else:
            validation = self.validation
        training_features = [[str(f) for f in self.features_data[self.features_data["inst"] == datapoint["inst"]].to_numpy()[0].tolist()] 
                             for datapoint in self.training_data]
        times = {opt:0 for opt in self.training_data[0]["times"].keys()}
        for i in range(len(self.training_data)):
            inst = self.training_data[i]["inst"]
            training_features[i].pop(training_features[i].index(inst))
            for key in self.training_data[i]["times"].keys():
                times[key] += self.training_data[i]["times"][key]

        self.sb = min(times.items())[0]
        training_features = np.array(training_features)

        if hyperparameters is None:
            hyperparameters = self.__get_clustering_parameters(training_features, train, validation, self.features_data, self.idx2comb)
        self.clustering_parameters = hyperparameters
        self.clustering_model = KMeans(**hyperparameters)
        y_pred = self.clustering_model.fit_predict(training_features)
        cluster_range = range(hyperparameters["n_clusters"])
        stats = {i: {comb:0 for comb in self.idx2comb.values()} for i in cluster_range}
        for i in range(len(train)):
            for option in train[i]["times"].keys():
                stats[y_pred[i]][option] += train[i]["times"][option]
        self.order = {str(i): {k:v for k, v in sorted(stats[i].items(), key=lambda item: item[1], reverse=False)} for i in cluster_range}
        if self.model_name is not None:
            os.makedirs(self.model_name, exist_ok=True)
            with open(os.path.join(self.model_name, "order"), 'w') as f:
                json.dump(self.order, f)
            with open(os.path.join(self.model_name, "model.pkl"), 'wb') as f:
                pickle.dump(self.clustering_model, f)
            with open(os.path.join(self.model_name, "hyperparams"), 'w') as f:
                json.dump(self.clustering_parameters, f)

    def __get_clustering_parameters(self, 
                                    training_features:'np.ndarray', 
                                    train_data:'list[dict]', 
                                    validation_data:'list[dict]', 
                                    features:'pd.DataFrame',
                                    idx2comb:'dict') -> 'dict':
        parameters = list(ParameterGrid({
            'n_clusters': range(2, 21),
            'init': ['k-means++', 'random'],
            'max_iter': [100, 200, 300],
            'tol': [1e-3, 1e-4, 1e-5],
            'n_init': [5, 10, 15, 'auto'],
            'random_state': [42],
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

            for future in tqdm(concurrent.futures.as_completed(futures)):
                params = futures[future]
                try:
                    result = future.result()
                    clusters_val.append((json.loads(params), result))
                except Exception as e:
                    raise Exception(f"An error occurred for text '{params}': {e}", file=stderr)

        best_cluster_val = min(clusters_val, key=lambda x: x[1])
        return best_cluster_val[0]
    
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

    @staticmethod 
    def from_pretrained(pretrained:'kmeans_initializer') -> 'Kmeans_predictor':
        predictor = Kmeans_predictor()
        predictor.idx2comb = pretrained.idx2comb
        predictor.order = pretrained.order
        predictor.clustering_model = pretrained.kmeans
        predictor.comb2idx = {v:k for k,v in pretrained.idx2comb.items()}
        predictor.clustering_parameters = {
            'n_clusters': pretrained.kmeans.n_clusters,
            'init': pretrained.kmeans.init,
            'max_iter': pretrained.kmeans.max_iter,
            'tol': pretrained.kmeans.tol,
            'n_init': pretrained.kmeans.n_init,
            'random_state': pretrained.kmeans.random_state,
            'verbose': 0
        }
        return predictor

    def __get_dataset(self, dataset:'list') -> 'list[dict]':
        if type(dataset[0]) == float:
            return [{"inst":"", "features":dataset}]
        return dataset

    def __get_prediction(self, options:'list', category:'int', order:'dict|None' = None):
        order = order if not order is None else self.order
        for candidate in order[str(category)]:
            if candidate in options:
                return candidate

    def predict(self, X, filter=False):
        """
        Implementation of sklearn's predict method.
        Returns the selected option for each input sample.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        filter : bool, default=False
            Whether to filter options based on feature values.
            
        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predicted classes.
        """
        # Check if this is our custom format or sklearn format
        if isinstance(X, list) and (len(X) == 0 or isinstance(X[0], (dict, float))):
            return self.predict_custom(X, filter)
        
        # Handle sklearn format (numpy array or similar)
        predictions = []
        for i in range(X.shape[0]):
            feat = X[i].reshape(1, -1)
            if np.isnan(feat).any():
                chosen_option = self.sb
            else:
                category = self.clustering_model.predict(feat)
                options = list(self.idx2comb.values())
                if filter:
                    options = [o for o in self.comb2idx.keys() if X[i, self.comb2idx[o]] < 0.5]
                    if len(options) == 0:
                        options = list(self.idx2comb.values())
                chosen_option = self.__get_prediction(options, int(category[0]))
            predictions.append(chosen_option)
        
        return np.array(predictions)

    def predict_custom(self, dataset:'list[dict]|list[float]', filter:'bool'=False) -> 'list[dict]|dict':
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
                - a float corresponding to the total time of the predicted options
        """
        is_single = type(dataset[0]) == float
        dataset = self.__get_dataset(dataset)
        if filter and len(dataset[0]["features"]) != len(list(self.idx2comb.keys())):
            raise Exception(f"number of features is different from number of combinations: {len(dataset[0]['features'])} != {len(list(self.idx2comb.keys()))}")

        predictions = []
        for datapoint in tqdm(dataset):
            feat = np.array(datapoint["features"]).reshape(1,-1)
            start = time()
            if isnan(feat.tolist()[0]):
                chosen_option = self.sb
            else:
                category = self.clustering_model.predict(feat)
                options = list(self.idx2comb.values())
                if filter:
                    options = [o for o in self.comb2idx.keys() if datapoint["features"][self.comb2idx[o]] < .5]
                    if len(options) == 0:
                        options = list(self.idx2comb.values())
                chosen_option = self.__get_prediction(options, int(category[0]))
            predictions.append({"chosen_option": chosen_option, "inst": datapoint["inst"], "time": time() - start})

        if is_single:
            return predictions[0]
        return predictions
    
    def fit(self, X, y=None):
        """
        Fit the KMeans clustering model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : None
            Ignored. This parameter exists only for compatibility with sklearn's API.
            
        Returns
        -------
        self : object
            Returns self.
        """
        if self.clustering_model is None:
            # Create a default clustering model
            self.clustering_parameters = {
                'n_clusters': self.n_clusters,
                'init': self.init,
                'max_iter': self.max_iter,
                'tol': self.tol,
                'n_init': self.n_init,
                'random_state': self.random_state,
                'verbose': self.verbose
            }
            self.clustering_model = KMeans(**self.clustering_parameters)
        
        # Fit the clustering model
        self.clustering_model.fit(X)
        
        # If we don't have order information yet, create a default order
        if self.order is None:
            if self.idx2comb is None:
                # Create dummy mappings if we don't have them
                unique_clusters = range(self.n_clusters)
                self.idx2comb = {i: f"option_{i}" for i in range(self.n_clusters)}
                self.order = {str(i): {f"option_{j}": j for j in range(self.n_clusters)} 
                              for i in unique_clusters}
            else:
                # Create default order using option names
                self.order = {str(i): {opt: j for j, opt in enumerate(self.idx2comb.values())} 
                             for i in range(self.n_clusters)}
                
            self.comb2idx = {v: k for k, v in self.idx2comb.items()}
            
            # Set default shortest option if missing
            if self.sb is None and self.idx2comb:
                self.sb = list(self.idx2comb.values())[0]
        
        return self
    
    def score(self, X, y=None):
        """
        Return a score based on the predicted execution time for the samples.
        For permutation_importance, higher values indicate better performance.
        
        If time information is available in y, uses that; otherwise calculates
        a proxy score based on the confidence of cluster assignment.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,) or array-like of dictionaries
            True labels for X. If provided as dictionaries with time information,
            uses that to calculate actual performance.
            
        Returns
        -------
        score : float
            Negative mean predicted time (higher is better). A higher score
            indicates better algorithm selection (shorter execution time).
        """
        predictions = self.predict(X)
        
        # If y contains time information (as in original format)
        if y is not None and isinstance(y, list) and len(y) > 0 and isinstance(y[0], dict) and "times" in y[0]:
            # Calculate actual time based on predictions
            total_time = 0
            for i, pred in enumerate(predictions):
                if pred in y[i]["times"]:
                    total_time += y[i]["times"][pred]
            return total_time/X.shape[0]
        
        # Otherwise use a proxy score based on distance to cluster centers
        elif self.clustering_model is not None:
            # Get distances to assigned clusters
            clusters = self.clustering_model.predict(X)
            distances = np.zeros(len(X))
            for i in range(len(X)):
                # Calculate distance to assigned cluster center
                center = self.clustering_model.cluster_centers_[clusters[i]]
                distances[i] = np.linalg.norm(X[i] - center)
            # Return negative mean distance (higher is better)
            return -np.mean(distances)
            
        return 0.0
    
    def get_params(self, deep=True):
        """
        Get parameters for this estimator.
        
        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.
            
        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        params = {
            'n_clusters': self.n_clusters,
            'init': self.init,
            'max_iter': self.max_iter,
            'tol': self.tol,
            'n_init': self.n_init,
            'random_state': self.random_state,
            'verbose': self.verbose,
            'max_threads': self.max_threads
        }
        return params
    
    def set_params(self, **parameters):
        """
        Set the parameters of this estimator.
        
        Parameters
        ----------
        **parameters : dict
            Estimator parameters.
            
        Returns
        -------
        self : object
            Estimator instance.
        """
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
