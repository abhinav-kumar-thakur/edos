import os
import math
from collections import defaultdict

import torch as t

from .ensemble import Ensemble
from src.config_reader import read_json_configs
from src.models.utils import get_model


class MetaClassifier(Ensemble):
    def __init__(self, config, device='cpu') -> None:
        super().__init__()
        self.device = device
        self.models = []
        self.metrics = []
        self.configs = config
        self.memoize = {}

        for model_config in self.configs.model.meta_classifier.models:
            c = read_json_configs(os.path.join('./configs', model_config['config']))
            m = read_json_configs(model_config['metrics']).configs
            model = get_model(c, model_config['path'], self.device)
            self.models.append(model)
            self.metrics.append(m)

        self.label2idx_a = self.get_label_index_a()
        self.idx2label_a = {v: k for k, v in self.label2idx_a.items()}

        self.loss_a = t.nn.CrossEntropyLoss()
        n1 = 3 * len(self.models) * (len(self.label2idx_a))
        n2 = math.ceil(n1/2)
        self.l1 = t.nn.Linear(n1, n2).to(self.device)
        self.l2 = t.nn.Linear(n2, 2).to(self.device)

    def forward(self, batch, train=True):
        features = defaultdict(list)
        if any([k for k in batch['rewire_id'] if k not in self.memoize]):
            for i, model in enumerate(self.models):
                model.eval()
                pred, loss = model(batch, train=False)
                metrics = self.metrics[i]
                for rewire_id in batch['rewire_id']:
                    logits = list(pred[rewire_id]['scores']['sexist'].values())
                    models_metrics = []
                    for k in self.label2idx_a.keys():
                        models_metrics.append(metrics['eval_metric']['a'][k]['precision'])
                        models_metrics.append(metrics['eval_metric']['a'][k]['recall'])
                    
                    features[rewire_id].extend(logits + models_metrics)

            for rewire_id in batch['rewire_id']:
                self.memoize[rewire_id] = features[rewire_id]
        else:
            for rewire_id in batch['rewire_id']:
                features[rewire_id] = self.memoize[rewire_id]


        x = t.relu(self.l1(t.tensor(list(features.values())).to(self.device)))
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
            labels[batch['rewire_id'][i]] = {
                'sexist': sexist_label if 'a' in self.configs.train.task else None,
                'category': None,
                'vector': None,
                'scores': {
                    'sexist': None,
                    'category': None,
                    'vector': None
                },
                'confidence': {
                    'sexist': None,
                    'category': None,
                    'vector': None
                },
                'confidence_s': {
                    'sexist': None,
                    'category': None,
                    'vector': None
                },
                'uncertainity': {
                    'sexist': None,
                    'category': None,
                    'vector': None
                }
            }

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
