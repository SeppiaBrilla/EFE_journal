import argcomplete
import numpy as np
import argparse
import torch
import torch.nn as nn
from json import loads, dump

from torch.utils.data import DataLoader
from helper import Dataset, dict_lists_to_list_of_dicts, split_data
from transformers import BertTokenizer
from models import RegressionModel
from training import train
from tqdm import tqdm

BERT_TYPE = "bert-base-uncased"

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True)
parser.add_argument("--epochs", type=int, required=True)
parser.add_argument("--learning_rate", type=float, required=True)
parser.add_argument("--save", required=True)
parser.add_argument("--fold", type=int, required=True)
parser.add_argument("--pre_trained", required=False)
parser.add_argument("--history", required=False)
parser.add_argument("--features_size", required=False, type=int)
parser.add_argument("--batch-size", required=False, type=int, default=4)
parser.add_argument("--patience", required=False, type=int, default=3)

argcomplete.autocomplete(parser)
def main():

    arguments = parser.parse_args()
    dataset = arguments.dataset
    pretrained_weights = arguments.pre_trained
    epochs = arguments.epochs
    learning_rate = arguments.learning_rate
    save_weights_file = arguments.save
    fold = arguments.fold
    history_file = arguments.history
    feature_size = arguments.features_size
    batch_size = arguments.batch_size
    patience = arguments.patience

    f = open(dataset)
    data = loads(f.read())
    f.close()

    tokenizer = BertTokenizer.from_pretrained(BERT_TYPE, clean_up_tokenization_spaces=True, model_max_length=2048)
    instances_and_model = [d["instance_value_json"] for d in data]

    x = dict_lists_to_list_of_dicts(tokenizer(instances_and_model, padding=True, truncation=True, return_tensors='pt'))
    y = []
    all_times = []
    combinations = [d["combination"] for d in sorted(data[0]["all_times"], key= lambda x: x["combination"])]
    for datapoint in data:
        y_datapoint = sorted(datapoint["all_times"], key= lambda x: x["combination"])
        datapoint["all_times"] = y_datapoint
        times = [t["time"] for t in y_datapoint]
        _y = np.log1p(times)
        all_times += times
        y.append(torch.tensor(_y, dtype=torch.float32))

    all_times = torch.tensor(all_times)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("operating on device:", device)

    x_train, _ = split_data(x, fold=fold)
    y_train, _ = split_data(y, fold=fold)

    x_train, x_validation = split_data(x_train, 9)
    y_train, y_validation = split_data(y_train, 9)

    train_dataloader = DataLoader(Dataset(x_train, y_train), batch_size=batch_size)
    validation_dataloader = DataLoader(Dataset(x_validation, y_validation), batch_size=batch_size)

    length = len(combinations)

    model = RegressionModel(feature_size, length)
    if pretrained_weights != None:
        model.load_state_dict(torch.load(pretrained_weights))

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    loss = nn.MSELoss()

    pbar = tqdm(total=epochs)

    model, training_history, _ = train(model=model, 
                                            train_dataset=train_dataloader, 
                                            validation_dataset=validation_dataloader, 
                                            optimizer=optimizer, 
                                            loss=loss, 
                                            epochs=epochs, 
                                            device=device, 
                                            hyperparam={"feature_size": feature_size, "output_size": length},
                                            patience=patience,
                                            shared_tqdm=pbar,
                                            temporary_save=save_weights_file,
                                            model_class=RegressionModel)

    f = open(history_file, "w")
    dump(training_history, f)
    f.close()
    
main()
