import torch.nn as nn
import argcomplete
import argparse
import torch
from json import loads, dump
from helper import dict_lists_to_list_of_dicts, is_competitive, set_seed, split_data, is_competitive, Dataset
from torch.utils.data import DataLoader
from transformers import  BertTokenizer
from models import CompetitiveModel
from training import train
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torch

BERT_TYPE = "bert-base-uncased"

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


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--epochs", type=int, required=True)
parser.add_argument("--lr", type=float, required=True)
parser.add_argument("--save", required=True)
parser.add_argument("--history", required=True)
parser.add_argument("--feature-size", required=True, type=int)
parser.add_argument("--batch-size", required=False, type=int, default=4)
parser.add_argument("--validation-split", required=False, type=float, default=.1)
parser.add_argument("--random-seed", required=True, type=int)

argcomplete.autocomplete(parser)
def main():

    arguments = parser.parse_args()
    dataset = arguments.dataset
    epochs = arguments.epochs
    learning_rate = arguments.lr
    save_file = arguments.save
    history_file = arguments.history
    feature_size = arguments.feature_size
    batch_size = arguments.batch_size
    validation_split = arguments.validation_split
    random_seed = arguments.random_seed

    f = open(dataset)
    data = loads(f.read())
    f.close()

    set_seed(random_seed)

    (x, y), combinations = prepare_nn_data(data)
    length = len(combinations)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("operating on device:", device)

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=validation_split)

    train_dataloader = DataLoader(Dataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    validation_dataloader = DataLoader(Dataset(x_val, y_val), batch_size=batch_size, shuffle=True)

    model_hyperparam = {'feature_size':feature_size, 'output_size':length}

    model = CompetitiveModel(**model_hyperparam)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    with tqdm(total=epochs) as pbar:
        model, training_history, _ = train(model=model, 
                                            train_dataset=train_dataloader, 
                                            validation_dataset=validation_dataloader, 
                                            optimizer=optimizer, 
                                            loss=nn.functional.binary_cross_entropy_with_logits, 
                                            epochs=epochs, 
                                            device=device, 
                                            hyperparam=model_hyperparam,
                                            patience=100,
                                            shared_tqdm=pbar,
                                            model_class=CompetitiveModel)
        
    torch.save(model.state_dict(), save_file)
    with open(history_file, 'w') as f:
        dump(training_history, f)
    
main()
