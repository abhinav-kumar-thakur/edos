import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder
from src.models.bert import BertClassifier
from src.models.bert_focal_loss import BertClassifier_fl
from src.models.unifiedQA import UnifiedQAClassifier
from src.config_reader import read_json_configs
import random
from typing import List
from ...trainer.edos_trainer import EDOSTrainer
from ...logger import Logger
from .ensemble import Ensemble
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict
import os
import pickle
import torch as t

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

class XGBoostEnsembler(Ensemble):
    def __init__(self, configs, logger:Logger, device='cpu',load_path=None):
        super().__init__()

        self.device = device
        self.logger = logger
        self.configs = configs
        
        self.models = []
        self.metrics = []
        self.xgb = xgb.XGBClassifier()

        for model_config in self.configs.model.xgboost_classifier.models:
            c = read_json_configs(os.path.join('./configs', model_config['config']))
            m = read_json_configs(model_config['metrics']).configs
            model = get_model(c, model_config['path'], self.device)
            self.models.append(model)
            self.metrics.append(m)
        
        self.label2idx_a = self.get_label_index_a()
        self.idx2label_a = {v: k for k, v in self.label2idx_a.items()}

    def get_features(self, batch, train=False):
        features = defaultdict(list)
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
        
        return features
    
    def fit(self, dataloader):
        inputs = []
        y_values = [] 

        for batch in tqdm(dataloader):
            features = self.get_features(batch, train=True)
            inputs.extend(features.values())
            y_values.extend(batch['label_sexist'])
            
        # one hot encode labels
        for i in range(len(y_values)):
            y_values[i] = self.label2idx_a[y_values[i]]

        self.xgb.fit(inputs, y_values)
                
    def forward(self, batch, train=False):
        features = self.get_features(batch, train)
        pred = self.xgb.predict(list(features.values()))
        labels = {}
        for i in range(len(pred)):
            sexist_label = self.idx2label_a[pred[i]]
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

        return labels, None

    def save(self, path):
        self.xgb.save_model(path)
    
    def load(self, path):
        self.xgb.load_model(path)

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

