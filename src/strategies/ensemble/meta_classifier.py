import os
import math
import itertools
from collections import defaultdict, Counter

import torch as t

from .ensemble import Ensemble
from src.config_reader import read_json_configs
from src.models.bert import BertClassifier
from src.models.bert_focal_loss import BertClassifier_fl
from src.models.unifiedQA import UnifiedQAClassifier

def get_model(configs, filepath, device):
    model_name = configs.model.type

    if model_name == 'bert':
        model = BertClassifier(configs, device)
    elif model_name == 'bert_fl':
        model = BertClassifier_fl(configs, device)
    elif model_name == 'unifiedQA':
        model = UnifiedQAClassifier(configs, device)
    else:
        raise Exception('Invalid model name')

    model.load_state_dict(t.load(filepath, map_location=device))
    return model

class MetaClassifier(Ensemble):
    def __init__(self, config, logger, device='cpu') -> None:
        super().__init__()
        self.device = device
        self.models = []
        self.configs = config
        self.logger = logger

        for model_config in self.configs.model.meta_classifier.models:
            c = read_json_configs(os.path.join('./configs', model_config['config']))
            model = get_model(c, model_config['path'], self.device)
            self.models.append(model)

        self.label2idx_a = self.get_label_index_a()
        self.idx2label_a = {v: k for k, v in self.label2idx_a.items()}

        self.loss_a = t.nn.CrossEntropyLoss()
        n1 = len(self.models)*3
        n2 = math.ceil((len(self.models)*3)/2)
        self.l1 = t.nn.Linear( n1, n2).to(self.device)
        self.l2 = t.nn.Linear(n2, 2).to(self.device)

    def forward(self, batch, train=True):
        predictions = defaultdict(list)
        for model in self.models:
            model.eval()
            pred, loss = model(batch, train=False)
            for rewire_id in batch['rewire_id']:
                label = self.label2idx_a[pred[rewire_id]['sexist']]
                
                predictions[rewire_id] += [label, pred[rewire_id]['confidence_s']['sexist'], pred[rewire_id]['uncertainity']['sexist']]

        x = t.relu(self.l1(t.tensor(list(predictions.values())).to(self.device)))
        pred_a = self.l2(x)

        loss = 0
        if train:
            if 'a' in self.configs.train.task:
                actual_a = t.tensor([self.label2idx_a[l] for l in batch['label_sexist']]).to(self.device)
                loss_a = self.loss_a(pred_a, actual_a)
                loss += loss_a
        
        labels = {}
        pred_a_ids = t.argmax(pred_a, dim=1)
        for i in range(len(pred_a_ids)):
            sexist_label = self.idx2label_a[pred_a_ids[i].item()]
            labels[batch['rewire_id'][i]] = {'sexist': sexist_label if 'a' in self.configs.train.task else None}
            

        return labels, loss

    def get_label_index_a(self):
        return {x: y['id'] for x, y in self.configs.datasets.labels.configs.items()}

    def get_label_index_b(self):
        label2idx, i = {}, 0
        for label_value in self.configs.datasets.labels.configs.values():
            for category in label_value['categories']:
                label2idx[category] = i
                i += 1
        return label2idx

    def get_label_index_c(self):
        label2idx, i = {}, 0
        for label_value in self.configs.datasets.labels.configs.values():
            for category in label_value['categories'].values():
                for vector in category['vectors']:
                    label2idx[vector] = i
                    i += 1
        return label2idx

    def get_trainable_parameters(self):
        optimizer_parameters = [
            {'params': [p for n, p in self.l1.named_parameters()], 'weight_decay': 0.01, 'lr': self.configs.train.optimizer.lr},

            {'params': [p for n, p in self.l2.named_parameters()], 'weight_decay': 0.01, 'lr': self.configs.train.optimizer.lr}
        ] 
        
        return optimizer_parameters


        
