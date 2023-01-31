import os
import itertools
from collections import defaultdict

from src.config_reader import read_json_configs
from src.models.utils import get_model
from .ensemble import Ensemble


class WeightedVoting(Ensemble):
    def __init__(self, config, device='cpu'):
        super().__init__()
        self.device = device
        self.models = []
        self.metrics = []
        self.configs = config

        self.model = None
        for model_config in self.configs.model.weighted_voting.models:
            c = read_json_configs(os.path.join('./configs', model_config['config']))
            m = read_json_configs(model_config['metrics']).configs
            model = get_model(c, model_config['path'], self.device)
            self.models.append(model)
            self.metrics.append(m)

    def format_predictions(self, pred, metrics, rewire_id):
        return {
            'sexist':pred[rewire_id]['sexist'],
            'weight_a': metrics['eval_metric']['a']['macro avg']['f1-score'] if 'a' in self.configs.train.task else None,
            'category': pred[rewire_id]['category'],
            'weight_b': metrics['eval_metric']['b'][pred[rewire_id]['category']]['f1-score'] if 'b' in self.configs.train.task else None,
            'vector': pred[rewire_id]['vector'],
            'weight_c':  metrics['eval_metric']['c'][pred[rewire_id]['vector']]['f1-score'] if 'c' in self.configs.train.task else None
        }

    def forward(self, batch, train=False):
        predictions = defaultdict(list)
        for i, model in enumerate(self.models):
            model.eval()
            pred, loss = model(batch, train=False)
            metrics = self.metrics[i]
            for rewire_id in batch['rewire_id']:
                predictions[rewire_id].append(self.format_predictions(pred, metrics, rewire_id))

        ## Getting Vote
        labels = {}
        for rew_id, pred in predictions.items():
            label_dict_a, label_dict_b, label_dict_c = defaultdict(float), defaultdict(float), defaultdict(float)
            weight_sum_a, weight_sum_b, weight_sum_c = 0, 0, 0
            for p in pred:
                if 'a' in self.configs.train.task:
                    label_dict_a[p['sexist']] += p['weight_a']
                    weight_sum_a += p['weight_a']
                if 'b' in self.configs.train.task:
                    label_dict_b[p['category']] += p['weight_b']
                    weight_sum_b += p['weight_b']
                if 'c' in self.configs.train.task:
                    label_dict_c[p['vector']] += p['weight_c']
                    weight_sum_c += p['weight_c']

            label_a = max(label_dict_a, key=label_dict_a.get) if 'a' in self.configs.train.task else None
            label_b = max(label_dict_b, key=label_dict_b.get) if 'b' in self.configs.train.task else None
            label_c = max(label_dict_c, key=label_dict_c.get) if 'c' in self.configs.train.task else None
            labels[rew_id] = {
                'sexist': label_a,
                'category': label_b,
                'vector': label_c,
                'confidence_s': {
                    'sexist': label_dict_a[label_a]/weight_sum_a if 'a' in self.configs.train.task else None,
                    'category': label_dict_b[label_b]/weight_sum_b if 'b' in self.configs.train.task else None,
                    'vector': label_dict_c[label_c]/weight_sum_c if 'c' in self.configs.train.task else None
                }
            }

        return labels, None
