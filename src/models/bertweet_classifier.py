from cgitb import text
from turtle import forward
import torch as t
from transformers import AutoModel


class BertTweetClassifier(t.nn.Module):
    def __init__(self, configs, device='cpu') -> None:
        super().__init__()

        self.device = device
        self.configs = configs
        self.bert = AutoModel.from_pretrained(configs.model.text.bert).to(device)
        self.linear_one = t.nn.Linear(configs.model.text.dimentions, 512).to(device)
        self.linear_two = t.nn.Linear(512, 256).to(device)
        self.linear_three = t.nn.Linear(256, 128).to(device)
        self.linear_four = t.nn.Linear(128, len(self.configs.datasets.labels)).to(device)

    def forward(self, input):
        input_ids = input['input_ids'].to(self.device)
        attention_mask = input['attention_mask'].to(self.device)

        with t.no_grad():
            _, pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)

        x = t.relu(self.linear_one(pooled_output))
        x = t.relu(self.linear_two(x))
        x = t.relu(self.linear_three(x))
        return self.linear_four(x)

    def get_intermediate_features(self, input, layer):
        input_ids = input['input_ids'].to(self.device)
        attention_mask = input['attention_mask'].to(self.device)

        with t.no_grad():
            _, pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)

        if layer == 0:
            return pooled_output

        x = t.relu(self.linear_one(pooled_output))
        if layer == 1:
            return x

        x = t.relu(self.linear_two(x))
        if layer == 2:
            return x
        x = t.relu(self.linear_three(x))
        if layer == 3:
            return x

        return self.linear_four(x)
