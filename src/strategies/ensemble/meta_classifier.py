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

        self.loss_a, self.loss_b, self.loss_c = t.nn.CrossEntropyLoss(), t.nn.CrossEntropyLoss(), t.nn.CrossEntropyLoss()
        label_len = None
        if 'a' in self.configs.train.task:
            self.label2idx_a = self.get_label_index_a()
            self.idx2label_a = {v: k for k, v in self.label2idx_a.items()} 
            label_len = len(self.label2idx_a)
        elif 'b' in self.configs.train.task:
            self.label2idx_b = self.get_label_index_b()
            self.idx2label_b = {v: k for k, v in self.label2idx_b.items()}
            if label_len is not None: assert label_len == len(self.label2idx_b), f"Incorrect Label Length: {label_len} != {len(self.label2idx_b)}"
            label_len = len(self.label2idx_b)
        elif 'c' in self.configs.train.task:
            self.label2idx_c = self.get_label_index_c()
            self.idx2label_c = {v: k for k, v in self.label2idx_c.items()}
            if label_len is not None: assert label_len == len(self.label2idx_c), f"Incorrect Label Length: {label_len} != {len(self.label2idx_c)}"
            label_len = len(self.label2idx_c)
        else: raise ValueError(f"Invalid Task: {self.configs.train.task}")

        n1 = 3 * len(self.models) * label_len
        n2 = math.ceil(n1/2)
        self.l1 = t.nn.Linear(n1, n2).to(self.device)
        self.l2 = t.nn.Linear(n2, 2).to(self.device) #? Should this be 2 or label_len?

    def forward(self, batch, train=True):
        features = defaultdict(list)
        perform_task_a = 'a' in self.configs.train.task
        perform_task_b = 'b' in self.configs.train.task
        perform_task_c = 'c' in self.configs.train.task
        if any([k for k in batch['rewire_id'] if k not in self.memoize]):
            for i, model in enumerate(self.models):
                model.eval()
                pred, loss = model(batch, train=False)
                metrics = self.metrics[i]
                for rewire_id in batch['rewire_id']:
                    if perform_task_a:
                        logits = list(pred[rewire_id]['scores']['sexist'].values())
                        models_metrics = []
                        for k in self.label2idx_a.keys():
                            models_metrics.append(metrics['eval_metric']['a'][k]['precision'])
                            models_metrics.append(metrics['eval_metric']['a'][k]['recall'])
                        
                        features[rewire_id].extend(logits + models_metrics)
                        
                    if perform_task_b:
                        logits = list(pred[rewire_id]['scores']['category'].values())
                        models_metrics = []
                        for k in self.label2idx_b.keys():
                            metric_dic = metrics['eval_metric']['b']
                            if k in metric_dic:
                                models_metrics.append(metric_dic[k]['precision'])
                                models_metrics.append(metric_dic[k]['recall'])
                            else:
                                models_metrics.append(0)
                                models_metrics.append(0)
                        #? Why are logits and metrics being appended to features?
                        #? Are they being used as input for the meta-classifier?
                        #? logits(5) + # keys(4) * 2 metrics = 13
                        features[rewire_id].extend(logits + models_metrics)
                    
                    if perform_task_c:
                        logits = list(pred[rewire_id]['scores']['vector'].values())
                        models_metrics = []
                        for k in self.label2idx_c.keys():
                            metric_dic = metrics['eval_metric']['c']
                            if k in metric_dic:
                                models_metrics.append(metric_dic[k]['precision'])
                                models_metrics.append(metric_dic[k]['recall'])
                            else:
                                models_metrics.append(0)
                                models_metrics.append(0)
                        
                        features[rewire_id].extend(logits + models_metrics)

            for rewire_id in batch['rewire_id']:
                self.memoize[rewire_id] = features[rewire_id]
        else:
            for rewire_id in batch['rewire_id']:
                features[rewire_id] = self.memoize[rewire_id]

        l1_op = self.l1(t.tensor(list(features.values())).to(self.device))
        x = t.relu(l1_op)
        pred_a = self.l2(x) #? Should this output be different for each task?

        loss = 0
        if train:
            if perform_task_a:
                actual_a = t.tensor([self.label2idx_a[l] for l in batch['label_sexist']]).to(self.device)
                loss_a = self.loss_a(pred_a, actual_a)
                loss += loss_a
            if perform_task_b:
                actual_b = t.tensor([self.label2idx_b[l] for l in batch['label_category']]).to(self.device)
                # loss_b = self.loss_b(pred_b, actual_b)
                loss_b = self.loss_b(pred_a, actual_b)
                loss += loss_b
            if perform_task_c:
                actual_c = t.tensor([self.label2idx_c[l] for l in batch['label_vector']]).to(self.device)
                # loss_c = self.loss_c(pred_c, actual_c)
                loss_c = self.loss_b(pred_a, actual_c)
                loss += loss_c

        labels = {}
        pred_a_ids = t.argmax(pred_a, dim=1)
        for i in range(len(pred_a_ids)):
            labels[batch['rewire_id'][i]] = {
                'sexist': self.idx2label_a[pred_a_ids[i].item()] if perform_task_a else None,
                'category': self.idx2label_b[pred_a_ids[i].item()] if perform_task_b else None,
                'vector': self.idx2label_c[pred_a_ids[i].item()] if perform_task_c else None,
                'scores': {
                    'sexist': pred[batch['rewire_id'][i]]['scores']['sexist'],
                    'category': pred[batch['rewire_id'][i]]['scores']['category'], #! This is a dictionary output, not just the confidence of the output
                    'vector': pred[batch['rewire_id'][i]]['scores']['vector'] 
                },
                'confidence': {
                    'sexist': pred[batch['rewire_id'][i]]['confidence']['sexist'],
                    'category': pred[batch['rewire_id'][i]]['confidence']['category'],
                    'vector': pred[batch['rewire_id'][i]]['confidence']['vector']
                },
                'confidence_s': {
                    'sexist': pred[batch['rewire_id'][i]]['confidence_s']['sexist'],
                    'category': pred[batch['rewire_id'][i]]['confidence_s']['category'],
                    'vector': pred[batch['rewire_id'][i]]['confidence_s']['vector']
                },
                'uncertainity': {
                    'sexist': pred[batch['rewire_id'][i]]['uncertainity']['sexist'],
                    'category': pred[batch['rewire_id'][i]]['uncertainity']['category'],
                    'vector': pred[batch['rewire_id'][i]]['uncertainity']['vector']
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
