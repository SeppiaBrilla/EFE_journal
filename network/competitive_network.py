import torch.nn as nn
import argcomplete
import argparse
from torch.utils.data import DataLoader
from competitive_model import CompetitiveModel
import torch
from tqdm import tqdm
from json import loads, dump
from helper import dict_lists_to_list_of_dicts, split_data, Dataset, is_competitive, set_seed
from transformers import BertTokenizer
from training import train

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
parser.add_argument("--limit", required=False, type=int, default=0)
parser.add_argument("--batch-size", required=False, type=int, default=4)

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

    f = open(dataset)
    data = loads(f.read())
    f.close()

    set_seed(42)

    tokenizer = BertTokenizer.from_pretrained(BERT_TYPE, clean_up_tokenization_spaces=True, model_max_length=2048)
    instances_and_model = [d["instance_value_json"] for d in data]

    x = dict_lists_to_list_of_dicts(tokenizer(instances_and_model, truncation=True, return_tensors='pt'))
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

    x_train, x_val = split_data(x, fold, buckets=5)
    y_train, y_val = split_data(y, fold, buckets=5)

    train_dataloader = DataLoader(Dataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    validation_dataloader = DataLoader(Dataset(x_val, y_val), batch_size=batch_size, shuffle=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("operating on device:", device)

    length = len(combinations)

    model = CompetitiveModel(feature_size, length)
    if pretrained_weights != None:
        model.load_state_dict(torch.load(pretrained_weights))

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    with tqdm(total=epochs) as pbar:
        model, train_data, _ = train(model=model, 
                                train_dataset=train_dataloader, 
                                validation_dataset=validation_dataloader, 
                                optimizer=optimizer, 
                                loss=nn.functional.binary_cross_entropy_with_logits,
                                epochs=epochs,
                                device=device, 
                                hyperparam={"feature_size": feature_size, "output_size": length},
                                patience=5,
                                model_class=CompetitiveModel,
                                shared_tqdm=pbar,
                                )
    torch.save(model.state_dict(), f"{save_weights_file}_final")

    f = open(history_file, "w")
    dump(train_data, f)
    f.close()

main()