from transformers import  BertModel, BertConfig
import torch.nn as nn
import torch

class CompetitiveModel(nn.Module):
    def __init__(self, feature_size, output_size) -> None:
        super().__init__()
        self.config = BertConfig(max_position_embeddings=2048, hidden_dropout_prob=0, attention_probs_dropout_prob=0)
        self.bert = BertModel(self.config)
        self.features = nn.Linear(self.bert.config.hidden_size,feature_size)
        self.dropout = nn.Dropout(.3)
        self.post_features = nn.Linear(feature_size, 200)
        self.output_layer = nn.Linear(200, output_size)
        self.activation = nn.functional.tanh

    def forward(self, inputs):
        _, encoded_input = self.bert(**inputs, return_dict = False)
        encoded_input = self.dropout(encoded_input)
        out = self.features(encoded_input)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.post_features(out)
        out = nn.functional.relu(out)
        out = self.dropout(out)
        return self.output_layer(out)

    def __get_feature(self, inputs):
        _, encoded_input = self.bert(**inputs, return_dict = False)
        out = self.features(encoded_input)
        features = self.activation(out)
        return features

    def __get_encoder_feature(self, inputs):
        _, encoded_input = self.bert(**inputs, return_dict = False)
        return encoded_input

    def get_encoder_features(self, dataset:list[dict], device):
        model = self.to(device)
        features = []
        for datapoint in dataset:
            tokenized_instance = {k:v.reshape((1,v.shape[0])).to(device) for k,v in datapoint.items()}
            feats = model.__get_encoder_feature(tokenized_instance)[0].detach().cpu()
            keys = list(tokenized_instance.keys())
            for k in keys:
                del tokenized_instance[k]
            del tokenized_instance
            out = []
            for i in range(len(feats)):
                out.append(float(feats[i]))
            features.append(out)
            torch.cuda.empty_cache()
        del model
        self = self.to('cpu')
        return features

    def get_features(self, dataset:list[dict], device):
        model = self.to(device)
        features = []
        for datapoint in dataset:
            tokenized_instance = {k:datapoint["token"][k].to(device) for k in datapoint["token"].keys()}
            feats = model.__get_feature(tokenized_instance)[0].detach().cpu()
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
