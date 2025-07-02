import torch.nn as nn
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import torch
from tqdm import tqdm
import json
from torch import load, device, cuda
from transformers import BertTokenizer, logging, BertModel, BertConfig
from typing import Callable
import argparse, argcomplete
logging.set_verbosity_error()

class Model(nn.Module):
    def __init__(self, feature_size:int, output_size:int, final_Activation_function:Callable|None=None) -> None:
        super().__init__()
        self.config = BertConfig(max_position_embeddings=2048, hidden_dropout_prob=0, attention_probs_dropout_prob=0)
        self.bert = BertModel(self.config)
        self.features = nn.Linear(self.bert.config.hidden_size,feature_size)
        self.dropout = nn.Dropout(.3)
        self.post_features = nn.Linear(feature_size, 200)
        self.output_layer = nn.Linear(200, output_size)
        self.activation = nn.functional.tanh
        self.final_activation = final_Activation_function

    def forward(self, inputs):
        _, encoded_input = self.bert(**inputs, return_dict = False)
        encoded_input = self.dropout(encoded_input)
        out = self.features(encoded_input)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.post_features(out)
        out = nn.functional.relu(out)
        out = self.dropout(out)
        out = self.output_layer(out)
        if self.final_activation:
            out = self.final_activation(out)
        return out

class CompetitiveModel(Model):
    def __init__(self, feature_size, output_size) -> None:
        super().__init__(feature_size, output_size, None)

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

  
class Language_features_generator:
    def __init__(self, pre_trained_weights:str, num_classes:int) -> None:
        super().__init__()
        self.device = device("cuda:0" if cuda.is_available() else "cpu")
        base_model = CompetitiveModel(feature_size=100, output_size=num_classes)
        adapter = ReducerAdapter(input_size=100, reduced_size=feature_size)
        self.model = MorfedModel(base_model, adapter)
        self.model.load_state_dict(load(pre_trained_weights, weights_only=True))
        self.model = self.model.to(self.device)
        self.model.eval()

    def generate(self, tokenized_instance: 'dict') -> 'dict[str,float]':
        tokenized_instance = {k:tokenized_instance[k].to(self.device) for k in tokenized_instance.keys()}
        with torch.no_grad():
            model_output = self.model(tokenized_instance)[0]
        keys = list(tokenized_instance.keys())
        for k in keys:
            del tokenized_instance[k]
        else:
            out = {}
            for i in range(len(model_output)):
                out[f"feat{i}"] = round(float(model_output[i]), 3)
            return out

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--model-name", type=str, required=True)
parser.add_argument("--seeds", type=str, required=True)
parser.add_argument("--feature-size", type=int, required=True)
parser.add_argument("--features-base-name", type=str, required=True)
argcomplete.autocomplete(parser)

arguments = parser.parse_args()
dataset_name = arguments.dataset
model_name = arguments.model_name
seeds = arguments.seeds.split(',')
feature_size = arguments.feature_size
feature_base = arguments.features_base_name

f = open(dataset_name)
dataset = json.load(f)
f.close()

cols = [d["combination"] for d in sorted(dataset[0]["all_times"], key= lambda x: x["combination"])]
output_size = len(cols)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", clean_up_tokenization_spaces=True, model_max_length=2048)
for datapoint in dataset:
    datapoint["token"] = tokenizer(datapoint["instance_value_json"], truncation=True, return_tensors="pt")

for i in seeds:
    new_features = []
    actual_name = model_name
    if '{i}' in model_name:
        split = model_name.split('{i}')
        actual_name = f'{split[0]}{i}{split[1]}'
    generator = Language_features_generator(actual_name, output_size)
    for datapoint in tqdm(dataset, desc=f"fold: {i}"):
        feature_gen = generator.generate(datapoint["token"])
        feature_gen["inst"] = datapoint["instance_name"]
        new_features.append(feature_gen)
    del generator.model
    del generator
    torch.cuda.empty_cache()
    new_features_df = pd.DataFrame(new_features)
    new_features_df.to_csv(f"{feature_base}_{i}.csv", index=False)
