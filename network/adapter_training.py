import torch.nn as nn
import torch.nn.functional as F
import torch
from models import CompetitiveModel
from transformers import BertTokenizer
from training import train, to, remove
from helper import dict_lists_to_list_of_dicts, set_seed, Dataset, split_data
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse, argcomplete
from json import load, dump

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

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True)
parser.add_argument("--epochs", type=int, required=True)
parser.add_argument("--lr", type=float, required=True)
parser.add_argument("--save", required=True)
parser.add_argument("--history", required=False)
parser.add_argument("--initial-size", required=False, type=int)
parser.add_argument("--batch-size", required=False, type=int, default=4)
parser.add_argument("--base-model", required=True, type=str)
parser.add_argument("--seed", required=True, type=int)

argcomplete.autocomplete(parser)

def main():

    arguments = parser.parse_args()
    dataset = arguments.dataset
    epochs = arguments.epochs
    learning_rate = arguments.lr
    save_file = arguments.save
    history_file = arguments.history
    feature_size = arguments.initial_size
    batch_size = arguments.batch_size

    f = open(dataset)
    data = load(f)
    f.close()

    combinations = [d["combination"] for d in sorted(data[0]["all_times"], key= lambda x: x["combination"])]
    base_model = CompetitiveModel(feature_size=100, output_size=len(combinations))
    base_model.load_state_dict(torch.load(arguments.base_model, weights_only=True))
    set_seed(arguments.seed)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("operating on device:", device)

    x, y = prepare_nn_data(data, base_model, device)
    x_train, x_val = split_data(x, 9, buckets=10)
    y_train, y_val = split_data(y, 9, buckets=10)

    train_dataloader = DataLoader(Dataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    validation_dataloader = DataLoader(Dataset(x_val, y_val), batch_size=batch_size, shuffle=True)


    model = ReducerAdapter(input_size=100, reduced_size=feature_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model, training_history, best_loss = train(model=model, 
                                               train_dataset=train_dataloader, 
                                               validation_dataset=validation_dataloader,
                                               optimizer=optimizer, 
                                               loss=nn.functional.mse_loss, 
                                               epochs=epochs, 
                                               device=device, 
                                               hyperparam={'input_size':100, 'reduced_size':feature_size},
                                               patience=epochs,
                                               shared_tqdm=tqdm(total=epochs),
                                               model_class=ReducerAdapter)

    print('final loss:', best_loss)
    assert isinstance(model, ReducerAdapter)
    combined = MorfedModel(baseModel=base_model, adapter=model)
    torch.save(combined.state_dict(), save_file)
    with open(history_file, 'w') as f:
        dump(training_history, f)

main()
